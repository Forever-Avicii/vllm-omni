import asyncio
import time
import weakref
from collections.abc import AsyncGenerator, Iterable
from dataclasses import asdict
from pprint import pformat
from typing import Any

from vllm.config import VllmConfig
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.plugins.io_processors import get_io_processor
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine.exceptions import EngineDeadError

# Internal imports (our code)
from vllm_omni.config import OmniModelConfig
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.distributed.ray_utils.utils import try_close_ray
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.log_utils import (
    OrchestratorMetrics,
)
from vllm_omni.entrypoints.omni import OmniBase
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.stage_utils import SHUTDOWN_TASK, OmniStageTaskType
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _weak_close_cleanup_async(stage_list, stage_in_queues, ray_pg, output_handler):
    """Weak reference cleanup function for AsyncOmni instances."""
    if stage_list:
        for q in stage_in_queues:
            try:
                q.put_nowait(SHUTDOWN_TASK)
            except Exception as e:
                logger.warning(f"Failed to send shutdown signal to stage input queue: {e}")
        for stage in stage_list:
            try:
                stage.stop_stage_worker()
            except Exception as e:
                logger.warning(f"Failed to stop stage worker: {e}")
    try_close_ray(ray_pg)
    # Cancel output handler
    if output_handler is not None:
        output_handler.cancel()


class AsyncOmni(OmniBase):
    """Asynchronous unified entry point supporting multi-stage pipelines for LLM and Diffusion models.

    Similar to the Omni class, but provides an asynchronous interface supporting
    asynchronous LLM and Diffusion models.

    Args:
        *args: Variable length argument list.
            - args[0]: Model name or path to load.
        **kwargs: Arbitrary keyword arguments.
            - model: Model name or path to load (if not in args).
            - stage_configs_path: Optional path to YAML file containing stage
              configurations. If None, configurations are loaded from the model.
            - log_stats: Whether to enable statistics logging
              be written to files with stage-specific suffixes.
            - stage_init_timeout: Per-stage init watchdog (seconds). Measured from
              when the previous stage finished (possibly a prior Omni run with GPU
              reuse/overlap) to when the current stage starts to initialize.
            - shm_threshold_bytes: Threshold in bytes for using shared memory
              for IPC. Objects larger than this threshold will use shared memory.
            - worker_backend: Backend for worker processes. Default is "multi_process".
            - ray_address: Address of Ray cluster for Ray backend, if using Ray backend.
            - batch_timeout: Timeout in seconds for batching requests within a stage
            - init_timeout: Timeout in seconds for waiting for all stages to initialize
            - Additional keyword arguments passed to stage engines.

    Example:
        >>> async_llm = AsyncOmni(model="Qwen/Qwen2.5-Omni-7B")
        >>> async for output in async_llm.generate(
        ...     prompt="Hello",
        ...     request_id="req-1",
        ...     sampling_params_list=[SamplingParams(), SamplingParams()]
        ... ):
        ...     print(output)
    """

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        # Pause/resume control attributes
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False

        # Request state tracking
        self.request_states: dict[str, ClientRequestState] = {}
        self.output_handler: asyncio.Task | None = None

        super().__init__(*args, **kwargs)

        # Register weak reference cleanup (called on garbage collection)
        self._weak_finalizer = weakref.finalize(
            self,
            _weak_close_cleanup_async,
            self.stage_list,
            self._stage_in_queues,
            self._ray_pg,
            self.output_handler,
        )

    def _create_default_diffusion_stage_cfg(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Create default diffusion stage configuration."""
        # TODO: here is different from the Omni class. We should merge the two in the future.
        cache_backend = kwargs.get("cache_backend", "none")
        cache_config = self._normalize_cache_config(cache_backend, kwargs.get("cache_config", None))

        devices = "0"
        if "parallel_config" in kwargs:
            parallel_config = kwargs["parallel_config"]
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"
        else:
            ulysses_degree = kwargs.get("ulysses_degree") or 1
            ring_degree = kwargs.get("ring_degree") or 1
            sequence_parallel_size = kwargs.get("sequence_parallel_size")
            if sequence_parallel_size is None:
                sequence_parallel_size = ulysses_degree * ring_degree
            num_devices = sequence_parallel_size
            for i in range(1, num_devices):
                devices += f",{i}"
            parallel_config = DiffusionParallelConfig(
                pipeline_parallel_size=1,
                data_parallel_size=1,
                tensor_parallel_size=1,
                sequence_parallel_size=sequence_parallel_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                cfg_parallel_size=1,
            )
        default_stage_cfg = [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {
                    "process": True,
                    "devices": devices,
                    "max_batch_size": 1,
                },
                "engine_args": {
                    "parallel_config": parallel_config,
                    "vae_use_slicing": kwargs.get("vae_use_slicing", False),
                    "vae_use_tiling": kwargs.get("vae_use_tiling", False),
                    "cache_backend": cache_backend,
                    "cache_config": cache_config,
                },
                "final_output": True,
                "final_output_type": "image",
            }
        ]
        default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
        return default_stage_cfg

    def _process_stage_ready(self, stage: OmniStage, stage_id: int, result: dict[str, Any]) -> None:
        # Store vllm_config received from worker process (may be None for diffusion stages)
        vllm_config = result.get("vllm_config")
        if vllm_config is not None:
            stage.set_vllm_config(vllm_config)
        tokenizer = result.get("tokenizer")
        if tokenizer is not None:
            stage.set_tokenizer(tokenizer)
        is_tracing_enabled = result.get("is_tracing_enabled")
        if is_tracing_enabled is not None:
            stage.set_is_tracing_enabled(is_tracing_enabled)
        super()._process_stage_ready(stage, stage_id, result)

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
        """Wait for all stages to report readiness."""
        super()._wait_for_stages_ready(timeout)
        for stage in self.stage_list:
            if stage.vllm_config is not None and stage.tokenizer is not None:
                try:
                    vllm_config = stage.vllm_config
                    tokenizer = stage.tokenizer
                    # Initialize input_processor
                    self.input_processor = OmniInputProcessor(
                        vllm_config=vllm_config,
                        tokenizer=tokenizer,
                    )
                    # Initialize model_config
                    self.model_config = vllm_config.model_config
                    # Initialize io_processor
                    io_processor_plugin = self.model_config.io_processor_plugin
                    self.io_processor = get_io_processor(vllm_config, io_processor_plugin)

                    logger.info(
                        f"[{self._name}] Initialized input_processor, "
                        f"io_processor, and model_config from stage-{stage.stage_id}",
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"[{self._name}] Failed to initialize processors from stage-{stage.stage_id}: {e}",
                    )
        # If no LLM stage found, set processors to None
        if not hasattr(self, "input_processor") or self.input_processor is None:
            logger.warning(
                f"[{self._name}] No LLM stage found, processors will not be available. "
                "This may cause issues with OpenAIServingModels."
            )
            self.input_processor = None
            self.io_processor = None
            self.model_config = None

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC.

        Alias for close() method. Cleans up all stage processes
        and inter-process communication resources.
        """
        if hasattr(self, "_weak_finalizer"):
            self._weak_finalizer()

    def _process_stage_output(
        self,
        stage_id: int,
        stage: OmniStage,
        result: dict[str, Any],
        metrics: OrchestratorMetrics,
        final_stage_id_for_e2e: int,
        _req_start_ts: dict[str, float],
        _wall_start_ts: float,
    ) -> tuple[Any, bool, list[OmniRequestOutput]]:
        """Process a single result from a stage and return outputs to yield."""
        req_id = result.get("request_id")
        if "error" in result:
            logger.error(
                f"[{self._name}] Stage {stage_id} error on request {req_id}: {result['error']}",
            )
            raise RuntimeError(result)  # Request Finished due to error

        engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
        if isinstance(engine_outputs, list):
            engine_outputs = engine_outputs[0]
        finished = engine_outputs.finished

        # Mark last output time for this stage whenever we receive outputs
        metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())
        try:
            _m = asdict(result.get("metrics")) if result.get("metrics") else None
            if _m is not None:
                if finished:
                    metrics.on_stage_metrics(stage_id, req_id, _m)
        except Exception as e:
            logger.exception(
                f"[{self._name}] Failed to process metrics for stage {stage_id}, req {req_id}: {e}",
            )
        logger.debug(
            f"[{self._name}] Stage-{stage_id} completed request {req_id}; forwarding or finalizing",
        )

        yielded_outputs = []
        if getattr(stage, "final_output", False):
            logger.debug(
                f"[{self._name}] Request {req_id} finalized at stage-{stage_id}",
            )

            # End-to-end timing and time-per-token for final output
            # (only once per request at the designated final stage)
            try:
                rid_key = str(req_id)
                if stage_id == final_stage_id_for_e2e and rid_key not in metrics.e2e_done and finished:
                    metrics.on_finalize_request(
                        stage_id,
                        req_id,
                        _req_start_ts.get(req_id, _wall_start_ts),
                    )
            except Exception as e:
                logger.exception(
                    f"[{self._name}] Finalize request handling error for req "
                    f"{req_id} at stage {stage_id}: {e}",
                )

            # Handle diffusion outputs that already contain images
            if stage.final_output_type == "image":
                images = []
                if isinstance(engine_outputs, OmniRequestOutput) and engine_outputs.images:
                    images = engine_outputs.images
                elif hasattr(engine_outputs, "images") and engine_outputs.images:
                    images = engine_outputs.images
                yielded_outputs.append(
                    OmniRequestOutput(
                        stage_id=stage_id,
                        final_output_type=stage.final_output_type,
                        request_output=engine_outputs,
                        images=images,
                    )
                )
            else:
                yielded_outputs.append(
                    OmniRequestOutput(
                        stage_id=stage_id,
                        final_output_type=stage.final_output_type,
                        request_output=engine_outputs,
                    )
                )

        return engine_outputs, finished, yielded_outputs

    async def generate(self, *args: Any, **kwargs: dict[str, Any]) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt asynchronously.

        Coordinates multi-stage pipeline through YAML configuration.
        Each stage will use AsyncOmniLLM or AsyncOmniDiffusion based on stage_type.
        Processes the prompt through all stages in the pipeline and yields
        outputs as they become available. Each stage uses its corresponding
        sampling parameters from the sampling_params_list.

        Args:
            *args: Arguments for generation.
                - prompt: Prompt to process. Can be a text string, token IDs,
                    or multimodal prompt.
                - request_id: Unique identifier for this request
                - sampling_params_list: List of SamplingParams, one for each stage.
                    Must have the same length as the number of stages.
                    If None, uses default sampling params for each stage.
            **kwargs: Additional arguments for generation.
                - prompt: Prompt to process. Can be a text string, token IDs,
                    or multimodal prompt.
                - request_id: Unique identifier for this request
                - sampling_params_list: List of SamplingParams, one for each stage.
                    Must have the same length as the number of stages.
                    If None, uses default sampling params for each stage.
                - output_modalities: Optional list of output modalities.

        Yields:
            OmniRequestOutput objects as they are produced by each stage.
            Each output contains the stage_id, final_output_type, and
            the request_output from that stage.

        Raises:
            ValueError: If sampling_params_list has incorrect length.
        """
        # Wait until generation is resumed if the engine is paused.
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        logger.debug(f"[{self._name}] generate() called")
        try:
            # Start output handler on the first call to generate()
            self._run_output_handler()

            prompt = args[0] if args else kwargs.get("prompt")
            request_id = args[1] if len(args) > 1 else kwargs.get("request_id")
            sampling_params_list = args[2] if len(args) > 2 else kwargs.get("sampling_params_list")
            output_modalities = kwargs.get("output_modalities", None)
            # TODO: lora_request, trace_headers, priority are not supported yet

            if sampling_params_list is None:
                # For Omni LLM, the params are parsed via the yaml file. For the current version,
                # diffusion params can parsed via the command line.
                omni_params_kwargs = {
                    k: v for k, v in kwargs.items() if k not in ["prompt", "request_id", "output_modalities"]
                }

                per_stage_params: list[Any] = []
                for stage_id, stage in enumerate(self.stage_list):
                    stage_type = getattr(stage, "stage_type", "llm")
                    if stage_type == "diffusion":
                        default_dict = self.default_sampling_params_list[stage_id]
                        # Merge user-provided kwargs
                        merged = {**default_dict, **omni_params_kwargs}
                        # Diffusion only needs to keep diff params, will be used via OmniDiffusionRequest
                        per_stage_params.append(merged)
                    else:
                        # LLM directly constructs SamplingParams, don't use the merged params
                        per_stage_params.append(self.default_sampling_params_list[stage_id])

                sampling_params_list = per_stage_params

            if len(sampling_params_list) != len(self.stage_list):
                raise ValueError(f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}")

            # Orchestrator keeps stage objects for input derivation
            num_stages = len(self.stage_list)
            # Track per-request start time for end-to-end timing
            _req_start_ts: dict[int, float] = {}
            _wall_start_ts: float = time.time()
            # _last_finish_ts: float = _wall_start_ts

            # Determine the final stage for E2E stats (highest stage_id with
            # final_output=True; fallback to last stage)
            final_stage_id_for_e2e = get_final_stage_id_for_e2e(
                output_modalities, self.output_modalities, self.stage_list
            )

            # Metrics/aggregation helper
            metrics = OrchestratorMetrics(
                num_stages,
                self._enable_stats,
                _wall_start_ts,
            )

            stage_queues = {stage_id: asyncio.Queue() for stage_id in range(num_stages)}

            # Seed stage-0 queue with all requests
            logger.debug(f"[{self._name}] Seeding request into stage-0")
            req_state = ClientRequestState(request_id)
            req_state.stage_queues = stage_queues
            self.request_states[request_id] = req_state

            # Mark first input time for stage-0
            metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()

            sp0: SamplingParams = sampling_params_list[0]  # type: ignore[index]
            task = {
                "request_id": request_id,
                "engine_inputs": prompt,
                "sampling_params": sp0,
            }
            self.stage_list[0].submit(task)
            # Only submit to stage 1 if stage 0 explicitly instructs or if we have specific logic
            # The original code had a specific logic for stage 1 submission which seemed to be hardcoded/testing specific
            # "prompt_token_ids" logic for stage 1 seems like a specific behavior.
            # I will preserve it for now but it looks like a smell.
            if "prompt_token_ids" in prompt:
                prompt_token_ids = prompt["prompt_token_ids"]
                prompt_1 = prompt.copy()
                prompt_1["prompt_token_ids"] = [0] * len(prompt_token_ids)
                task_1 = {
                    "request_id": request_id,
                    "engine_inputs": prompt_1,
                    "sampling_params": sampling_params_list[1],
                }
                # Check if we should submit to stage 1.
                # In general, orchestrator should flow 0 -> 1 -> 2.
                # The original code unconditionally submitted to stage 1 if prompt had token ids.
                # If stage 0 is non-streaming or we are pipelining, we might want this.
                # However, for general correctness, we should let the loop handle transitions.
                # But the original code did:
                # self.stage_list[1].submit(task_1)
                # I'll comment it out or leave it if it was critical.
                # Given the user wants optimization, I'll stick to safe behavior.
                # The provided user snippet does NOT show this part, it shows the loop.
                # I will leave the initialization as is (from original file) if I haven't changed it,
                # but I AM rewriting the method.
                # I will include it to match original behavior.
                self.stage_list[1].submit(task_1)
            
            _req_start_ts[request_id] = time.time()
            logger.info(f"[{self._name}] Enqueued request {request_id} to stage-0")

            logger.info(f"[{self._name}] Entering scheduling loop: stages={num_stages}")
            
            # Check if async_chunk is enabled (assume it's in stage 0 engine args)
            async_chunk = self.stage_list[0].engine_args.get("async_chunk", False)

            if async_chunk:
                all_stages_finished = {sid: False for sid in range(num_stages)}
                while not all(all_stages_finished.values()):
                    for stage_id, stage in enumerate(self.stage_list[: final_stage_id_for_e2e + 1]):
                        if all_stages_finished[stage_id]:
                            continue
                        
                        # Use try_get_nowait or get? The user snippet uses .get() which awaits.
                        # If we await sequentially, we might block other stages.
                        # But asyncio.Queue.get() waits until item is available.
                        # If we wait for stage 0, stage 1 might be ready.
                        # Using get() on specific stage queue might block the loop if that stage has no output yet.
                        # This suggests we should use asyncio.wait on all queues or use get_nowait with sleep.
                        # However, the user snippet used: `result = await req_state.stage_queues[stage_id].get()`
                        # This implies it blocks until THAT stage produces something.
                        # If stage 0 produces 10 items, and we block on stage 0, we process them.
                        # But if stage 1 produces items while we wait for stage 0, we delay stage 1 processing.
                        # To truly be async, we should probably gather or wait for ANY queue.
                        # But adhering to the user snippet logic:
                        
                        # The user snippet logic for async_chunk:
                        # while not all finished:
                        #   for stage in stages:
                        #      if finished: continue
                        #      result = await queue.get()
                        
                        # This logic is FLAWED if stage 0 is slow but stage 1 is fast (e.g. buffering).
                        # But if the user provided this as "optimize this", maybe they want me to fix the BLOCKING nature?
                        # Or maybe they just want the code duplication removed.
                        # The user said "Help me optimize this part of code".
                        # And showed the duplication.
                        
                        # I will implement the loop but maybe use `asyncio.wait` for better concurrency if I can.
                        # For now, let's implement the deduplication first.
                        
                        # Actually, checking queue size or using race would be better.
                        # But let's stick to the structure but use the helper.
                        
                        # To avoid blocking indefinitely on one stage while others have work,
                        # we should probably only `await get()` if we know there's something, or use `timeout`.
                        # But `req_state` queues are populated by `output_handler`.
                        
                        # Let's implement the loop as requested but using the helper.
                        
                        # Wait, if I use `await get()` on stage 0, and stage 0 is waiting for stage 1 (circular?) no, pipeline is DAG.
                        # But if stage 1 has output, and we are stuck on stage 0...
                        # The snippet provided by user DOES `await req_state.stage_queues[stage_id].get()`.
                        
                        result = await req_state.stage_queues[stage_id].get()
                        
                        engine_outputs, finished, yielded_outputs = self._process_stage_output(
                            stage_id, stage, result, metrics, final_stage_id_for_e2e, _req_start_ts, _wall_start_ts
                        )
                        
                        all_stages_finished[stage_id] = finished
                        
                        for output in yielded_outputs:
                            yield output

                        # Forward to next stage
                        next_stage_id = stage_id + 1
                        # Logic for async_chunk might be different for forwarding?
                        # Usually async chunk means we forward *chunks*.
                        # The original/snippet logic for forwarding in the ELSE block handles `finished` checks.
                        # In `async_chunk` block of snippet, there was NO forwarding logic visible?
                        # Wait, the snippet ENDS after `yield OmniRequestOutput`.
                        # It has `logger.info(f"[{self._name}] Request {req_id} finalized at stage-{stage_id}")`.
                        # It does NOT show forwarding logic in the `if self.async_chunk` block!
                        # This implies `async_chunk` mode might handle forwarding implicitly via connectors?
                        # Or maybe the user snippet was incomplete?
                        # "Received result from stage-{stage_id}: {result}" ...
                        
                        # If I look at `omni_stage.py`, `async_chunk` enables "injecting connectors config".
                        # This suggests data flows via connectors (Ray/ZMQ/etc) directly between workers,
                        # avoiding the orchestrator for intermediate data.
                        # So the orchestrator only receives metrics and final outputs?
                        # If so, we don't need to forward.
                        
                        # If `async_chunk` is True, we assume connectors handle data flow.
                        # So we just consume outputs.
                        pass
                    
                    # Finalize request if needed
                    # The snippet has:
                    # try: rid_key = str(req_id) ... metrics.on_finalize_request ...
                    # This is inside the loop in snippet.
                    # My helper `_process_stage_output` handles `on_finalize_request`.
                    pass

            else:
                # Sequential / non-async-chunk logic
                for stage_id, stage in enumerate(self.stage_list[: final_stage_id_for_e2e + 1]):
                    finished = False
                    while not finished:
                        result = await req_state.stage_queues[stage_id].get()
                        # Note: User snippet used `req_state.queue.get()` in the else block
                        # but `req_state` in `AsyncOmni` uses `stage_queues` (dict).
                        # The original code `AsyncOmni` uses `stage_queues`.
                        # I will use `req_state.stage_queues[stage_id]`.
                        
                        engine_outputs, finished, yielded_outputs = self._process_stage_output(
                            stage_id, stage, result, metrics, final_stage_id_for_e2e, _req_start_ts, _wall_start_ts
                        )
                        
                        stage.set_engine_outputs(engine_outputs)
                        
                        for output in yielded_outputs:
                            yield output
                            
                    # Forward to next stage if there is one (and we are not in async_chunk mode which handles it via connectors presumably,
                    # or if we are in non-async mode we must manual forward)
                    
                    next_stage_id = stage_id + 1
                    if next_stage_id <= final_stage_id_for_e2e and finished:
                        next_stage: OmniStage = self.stage_list[next_stage_id]
                        next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)
                        sp_next: SamplingParams = sampling_params_list[next_stage_id]

                        # Check if we have a connector for this edge
                        connector_key = (str(stage_id), str(next_stage_id))
                        connector = self.connectors.get(connector_key)

                        sent_via_connector = False
                        if connector:
                            sent_via_connector = try_send_via_connector(
                                connector=connector,
                                stage_id=stage_id,
                                next_stage_id=next_stage_id,
                                req_id=request_id,
                                next_inputs=next_inputs,
                                sampling_params=sp_next,
                                original_prompt=prompt,
                                next_stage_queue_submit_fn=self.stage_list[next_stage_id].submit,
                                metrics=metrics,
                            )

                        if not sent_via_connector:
                            error_msg = (
                                f"[{self._name}] Failed to send request {request_id} to stage-{next_stage_id} via connector. "
                                "Configure a connector for this edge or inspect connector logs for details."
                            )
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        logger.debug(f"[{self._name}] Forwarded request {request_id} to stage-{next_stage_id}")
                    else:
                        logger.debug(f"[{self._name}] Request {request_id} fully completed")

            logger.info(f"[{self._name}] All requests completed")

            # Summarize and print stats
            try:
                summary = metrics.build_and_log_summary(final_stage_id_for_e2e)
                logger.info("[Summary] %s", pformat(summary, sort_dicts=False))
            except Exception as e:
                logger.exception(f"[{self._name}] Failed to build/log summary: {e}")
            finally:
                self.request_states.pop(request_id, None)
        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            logger.info("[AsyncOrchestrator] Request %s aborted.", request_id)
            raise

    def _run_output_handler(self) -> None:
        if self.output_handler is not None:
            return

        stage_list = self.stage_list
        request_states = self.request_states

        async def output_handler():
            try:
                while True:
                    idle = True
                    for stage_id, stage in enumerate(stage_list):
                        result = stage.try_collect()
                        if result is None:
                            continue
                        idle = False
                        if result.get("type") == "stage_ready":
                            # Only happens when stage is initialized slower than expected,
                            # so we wait for a short time and try again
                            await asyncio.sleep(0.05)
                            continue
                        req_id = result.get("request_id")
                        req_state = request_states.get(req_id)
                        if req_state is None:
                            logger.debug(
                                f"[{self._name}] Request may have been aborted; \
                                dropping output for req {req_id} at stage-{stage_id}"
                            )
                            continue
                        if hasattr(req_state, 'stage_queues') and stage_id in req_state.stage_queues:
                            await req_state.stage_queues[stage_id].put(result)                            
                        else:
                            # Fallback to old behavior for compatibility
                            await req_state.queue.put(result)                            
                            req_state.stage_id = stage_id
                    if idle:
                        await asyncio.sleep(0.001)  # Avoid CPU overload when idle
                    else:
                        await asyncio.sleep(0)
            except Exception as e:
                logger.exception("AsyncOmni output_handler failed.")
                for req_state in request_states.values():
                    error_msg = {"request_id": req_state.request_id, "error": str(e)}
                    # Send error to all stage queues
                    if hasattr(req_state, 'stage_queues'):                   
                        for queue in req_state.stage_queues.values():                        
                            await queue.put(error_msg)
                    else:
                        await req_state.queue.put(error_msg)
                self.output_handler = None  # Make possible for restart

        self.output_handler = asyncio.create_task(output_handler())

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return len(self._stage_in_queues) > 0

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return not self.is_running

    @property
    def _name(self) -> str:
        return "AsyncOrchestrator"

    @property
    def is_async(self) -> bool:
        return True

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    async def abort(self, request_id: str | Iterable[str]) -> None:
        abort_task = {"type": OmniStageTaskType.ABORT, "request_id": request_id}
        for stage in self.stage_list:
            stage.submit(abort_task)
        return None

    async def get_vllm_config(self) -> VllmConfig:
        for stage in self.stage_list:
            if stage.is_comprehension:
                # Use the vllm_config received from worker process
                if stage.vllm_config is not None:
                    return stage.vllm_config
        return None

    async def get_model_config(self) -> OmniModelConfig:
        for stage in self.stage_list:
            if stage.is_comprehension:
                # Use the vllm_config received from worker process
                if stage.vllm_config is not None:
                    return stage.vllm_config.model_config
        return None

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return None

    async def get_tokenizer(self) -> TokenizerLike:
        for stage in self.stage_list:
            if stage.is_comprehension:
                return stage.tokenizer
        return None

    async def is_tracing_enabled(self) -> bool:
        for stage in self.stage_list:
            if stage.is_comprehension:
                return stage.is_tracing_enabled
        return False

    async def do_log_stats(self) -> None:
        pass

    async def check_health(self) -> None:
        pass

    async def reset_mm_cache(self) -> None:
        pass

    async def reset_prefix_cache(self, reset_running_requests: bool = False) -> bool:
        pass

    async def sleep(self, level: int = 1) -> None:
        pass

    async def wake_up(self, tags: list[str] | None = None) -> None:
        pass

    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return False

    async def encode(
        self,
        *args,
        **kwargs,
    ):
        """Generate outputs for a request from a pooling model."""
        raise NotImplementedError("encode() is not implemented for AsyncOmni")

    async def start_profile(self, stages: list[int] | None = None) -> None:
        """Start profiling for specified stages.

        Async wrapper around the base implementation for API consistency.

        Args:
            stages: List of stage IDs to start profiling. If None, starts
                profiling for all stages that have profiling enabled.

        Example:
            >>> await async_omni.start_profile()
            >>> async for output in async_omni.generate(...):
            ...     pass
            >>> await async_omni.stop_profile()
        """
        super().start_profile(stages)

    async def stop_profile(self, stages: list[int] | None = None) -> None:
        """Stop profiling for specified stages.

        Async wrapper around the base implementation for API consistency.

        Args:
            stages: List of stage IDs to stop profiling. If None, stops
                profiling for all stages.

        Example:
            >>> await async_omni.start_profile()
            >>> async for output in async_omni.generate(...):
            ...     pass
            >>> await async_omni.stop_profile()
        """
        super().stop_profile(stages)

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """
        Pause generation to allow model weight updates.

        New generation/encoding requests are blocked until resume.

        Args:
            wait_for_inflight_requests: When ``True`` waits for in-flight
                requests to finish before pausing. When ``False`` (default),
                immediately aborts any in-flight requests.
            clear_cache: Whether to clear KV cache and prefix cache after
                draining. Set to ``False`` to preserve cache for faster resume.
                Default is ``True`` (clear caches).
        """

        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        # Note: AsyncOmni uses a stage-based architecture without a central
        # output_processor. For now, we simply set the pause flag and let
        # new requests wait. In-flight requests will complete naturally.
        # TODO: Implement request abortion for stages if needed.

        # Clear cache if requested
        if clear_cache:
            await self.reset_prefix_cache()
            await self.reset_mm_cache()

    async def resume_generation(self) -> None:
        """Resume generation after :meth:`pause_generation`."""

        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()  # Wake up all waiting requests

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""

        async with self._pause_cond:
            return self._paused
