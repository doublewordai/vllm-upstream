# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    is_dbo_breakable_cudagraph_enabled,
)
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    BatchDescriptor,
    DPMetadata,
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.deep_gemm import set_num_sms as deep_gemm_set_num_sms
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import weak_ref_tensors
from vllm.v1.worker.sm_control import (
    get_all2all_manager_for_sm_control,
    get_ubatch_comm_sms,
)
from vllm.v1.worker.ubatching import UBatchContext, make_ubatch_contexts
from vllm.v1.worker.ubatch_utils import ensure_tensor_alignment

logger = init_logger(__name__)


def _cat_ubatch_outputs(
    sorted_results: list,
) -> "torch.Tensor | tuple[torch.Tensor, ...]":
    """Concatenate per-ubatch model outputs along the batch dim.

    Most models return a single hidden-states tensor per ubatch. Target
    models running with auxiliary output (e.g. EAGLE3 speculative decoding,
    which collects aux hidden states for the drafter) return a tuple of
    tensors instead. Fan out over tuple components so `torch.cat` sees
    matching shapes and the caller receives the same structure the model
    produced for a single ubatch (#40769).
    """
    if sorted_results and isinstance(sorted_results[0], tuple):
        return tuple(torch.cat(parts, dim=0) for parts in zip(*sorted_results))
    return torch.cat(sorted_results, dim=0)


@dataclass
class UbatchMetadata:
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    num_tokens: int


@dataclass
class CUDAGraphMetaData:
    cudagraph: torch.cuda.CUDAGraph
    ubatch_metadata: UbatchMetadata
    outputs: Any | None = None


@dataclass
class BreakableCUDAGraphMetaData:
    captures: list[BreakableCUDAGraphCapture]
    ubatch_metadata: list[UbatchMetadata]
    outputs: list[Any] | None = None
    capture_log_emitted: bool = False
    replay_log_emitted: bool = False
    fallback_log_count: int = 0


class SMControlContextManager:
    def __init__(
        self,
        comm_sms: int,
        set_comm_sms: Callable[[int], None],
        set_compute_sms: Callable[[int], None],
    ):
        """
        Context manager for controlling SM (Streaming Multiprocessor)
        allocation. Upon entering the context, it sets the number of SMs
        allocated for communication and computation to comm_sms and
        total_sms - comm_sms respectively. Upon exiting, it restores the
        allocation to use all available SMs (i.e. total_sms).

        Args:
            comm_sms (int): The number of SMs to allocate for communication.
                (The remainder will be used for computation.)
            set_comm_sms (Callable[[int], None]):
                A function that sets the number of SMs for communication.
            set_compute_sms (Callable[[int], None]):
                A function that sets the number of SMs for computation.
        """

        assert current_platform.is_cuda() or current_platform.is_rocm(), (
            "SM/CU control is supported on CUDA and ROCm platforms"
        )
        device = torch.accelerator.current_device_index()
        total_sms = num_compute_units(device)

        assert comm_sms < total_sms
        self.total_sms = total_sms
        self.compute_sms = total_sms - comm_sms
        self.comm_sms = comm_sms
        self.set_comm_sms = set_comm_sms
        self.set_compute_sms = set_compute_sms

    def __enter__(self):
        self.set_comm_sms(self.comm_sms)
        self.set_compute_sms(self.compute_sms)

    def __exit__(self, exc_type, exc_value, traceback):
        self.set_comm_sms(self.total_sms)
        self.set_compute_sms(self.total_sms)


class UBatchWrapper:
    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.cuda.device,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.cuda.Stream(device=device)
        # Ubatch threads plus the main thread
        self.ready_barrier = threading.Barrier(
            self.vllm_config.parallel_config.num_ubatches + 1
        )

        self.cudagraphs: dict[int, CUDAGraphMetaData] = {}
        self.breakable_cudagraphs: dict[
            tuple[BatchDescriptor, ...], BreakableCUDAGraphMetaData
        ] = {}
        self.breakable_decision_log_counts: dict[str, int] = {}
        parallel_config = vllm_config.parallel_config
        self.use_breakable_cudagraphs = (
            runtime_mode is not CUDAGraphMode.NONE
            and is_dbo_breakable_cudagraph_enabled()
            and parallel_config.use_ubatching
            and parallel_config.all2all_backend == "deepep_high_throughput"
            and parallel_config.data_parallel_size > 1
        )
        self.breakable_graph_pools: dict[int, Any] = {}

        self.cudagraph_wrapper = None
        if runtime_mode is not CUDAGraphMode.NONE and not self.use_breakable_cudagraphs:
            self.cudagraph_wrapper = CUDAGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode
            )

        self.sm_control = self._create_sm_control_context(vllm_config)
        self.device = device
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

    @property
    def graph_pool(self):
        if self.cudagraph_wrapper is not None:
            return self.cudagraph_wrapper.graph_pool
        if self.use_breakable_cudagraphs:
            return None
        return None

    @graph_pool.setter
    def graph_pool(self, graph_pool) -> None:
        if self.cudagraph_wrapper is not None:
            self.cudagraph_wrapper.graph_pool = graph_pool

    def clear_graphs(self) -> None:
        self.cudagraphs.clear()
        self.breakable_cudagraphs.clear()
        self.breakable_graph_pools.clear()
        if self.cudagraph_wrapper is not None:
            self.cudagraph_wrapper.clear_graphs()

    def _get_breakable_graph_pool(self, ubatch_id: int) -> Any:
        graph_pool = self.breakable_graph_pools.get(ubatch_id)
        if graph_pool is None:
            graph_pool = current_platform.graph_pool_handle()
            self.breakable_graph_pools[ubatch_id] = graph_pool
        return graph_pool

    def _log_breakable_decision(self, message: str, *args: Any) -> None:
        if not envs.VLLM_DBO_DEBUG_LOGGING:
            return
        count = self.breakable_decision_log_counts.get(message, 0)
        if count < 16:
            logger.info(message, *args)
        elif count == 16:
            logger.info(
                "Suppressing further DBO breakable CUDA graph decision logs "
                "for: %s",
                message,
            )
        self.breakable_decision_log_counts[message] = count + 1

    @staticmethod
    def _create_sm_control_context(vllm_config: VllmConfig):
        device = torch.accelerator.current_device_index()
        total_sms = num_compute_units(device)
        all2all_manager = get_all2all_manager_for_sm_control(vllm_config)
        comm_sms = get_ubatch_comm_sms(vllm_config, total_sms, all2all_manager)

        set_comm_sms = lambda sms: None
        if comm_sms > 0 and all2all_manager is not None:
            set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)

        # TODO(lucas): support other kernels besides DeepGEMM
        set_compute_sms = lambda sms: None
        if has_deep_gemm() and comm_sms > 0:
            set_compute_sms = lambda sms: deep_gemm_set_num_sms(sms)

        return SMControlContextManager(
            comm_sms=comm_sms,
            set_comm_sms=set_comm_sms,
            set_compute_sms=set_compute_sms,
        )

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of "
                f"cudagraph wrapper: {self._runnable_str}"
            )
        raise AttributeError

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    @staticmethod
    def _copy_tensor_tree(dst: Any, src: Any) -> bool:
        return UBatchWrapper._copy_tensor_tree_mismatch(dst, src) is None

    @staticmethod
    def _describe_tree_value(value: Any) -> str:
        if isinstance(value, torch.Tensor):
            return (
                f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"device={value.device})"
            )
        if value is None:
            return "None"
        return type(value).__name__

    @staticmethod
    def _is_flashmla_sched_meta(value: Any) -> bool:
        return type(value).__name__ == "FlashMLASchedMeta"

    @staticmethod
    def _copy_flashmla_sched_meta_mismatch(
        dst: Any,
        src: Any,
        path: str,
    ) -> str | None:
        """Refresh FlashMLA planner tensors without treating config as input.

        FlashMLASchedMeta is populated lazily by the first FlashMLA call in a
        step. Its tensor buffers are graph inputs that must stay at captured
        addresses, but its Python config/planner fields are ephemeral and may be
        None before the first runtime attention layer uses them.
        """
        if dst is None or src is None:
            return None
        if not (
            UBatchWrapper._is_flashmla_sched_meta(dst)
            and UBatchWrapper._is_flashmla_sched_meta(src)
        ):
            return (
                f"{path}: FlashMLASchedMeta type mismatch "
                f"(captured={UBatchWrapper._describe_tree_value(dst)}, "
                f"runtime={UBatchWrapper._describe_tree_value(src)})"
            )

        for attr in ("tile_scheduler_metadata", "num_splits"):
            if not hasattr(dst, attr) or not hasattr(src, attr):
                continue
            src_value = getattr(src, attr)
            dst_value = getattr(dst, attr)
            if src_value is None:
                continue
            if dst_value is None:
                return f"{path}.{attr}: captured scheduler tensor is None"
            mismatch = UBatchWrapper._copy_tensor_tree_mismatch(
                dst_value, src_value, f"{path}.{attr}"
            )
            if mismatch is not None:
                return mismatch

        if hasattr(dst, "have_initialized"):
            setattr(dst, "have_initialized", False)
        return None

    @staticmethod
    def _copy_tensor_tree_mismatch(dst: Any, src: Any, path: str = "root") -> str | None:
        if dst is src:
            return None
        if (
            UBatchWrapper._is_flashmla_sched_meta(dst)
            or UBatchWrapper._is_flashmla_sched_meta(src)
        ):
            return UBatchWrapper._copy_flashmla_sched_meta_mismatch(dst, src, path)
        if dst is None or src is None:
            if dst is src:
                return None
            return (
                f"{path}: one side is None "
                f"(captured={UBatchWrapper._describe_tree_value(dst)}, "
                f"runtime={UBatchWrapper._describe_tree_value(src)})"
            )
        if isinstance(dst, torch.Tensor) or isinstance(src, torch.Tensor):
            if not isinstance(dst, torch.Tensor) or not isinstance(src, torch.Tensor):
                return (
                    f"{path}: tensor/non-tensor mismatch "
                    f"(captured={UBatchWrapper._describe_tree_value(dst)}, "
                    f"runtime={UBatchWrapper._describe_tree_value(src)})"
                )
            if dst.shape != src.shape or dst.dtype != src.dtype:
                return (
                    f"{path}: tensor metadata mismatch "
                    f"(captured=shape{tuple(dst.shape)} dtype={dst.dtype}, "
                    f"runtime=shape{tuple(src.shape)} dtype={src.dtype})"
                )
            dst.copy_(src, non_blocking=True)
            return None
        if isinstance(dst, IntermediateTensors) or isinstance(src, IntermediateTensors):
            if not isinstance(dst, IntermediateTensors) or not isinstance(
                src, IntermediateTensors
            ):
                return (
                    f"{path}: IntermediateTensors type mismatch "
                    f"(captured={UBatchWrapper._describe_tree_value(dst)}, "
                    f"runtime={UBatchWrapper._describe_tree_value(src)})"
                )
            return UBatchWrapper._copy_tensor_tree_mismatch(
                dst.tensors, src.tensors, f"{path}.tensors"
            )
        if isinstance(dst, dict) or isinstance(src, dict):
            if not isinstance(dst, dict) or not isinstance(src, dict):
                return (
                    f"{path}: dict type mismatch "
                    f"(captured={UBatchWrapper._describe_tree_value(dst)}, "
                    f"runtime={UBatchWrapper._describe_tree_value(src)})"
                )
            for key, src_value in src.items():
                if key not in dst:
                    return f"{path}: runtime key {key!r} missing from captured dict"
                mismatch = UBatchWrapper._copy_tensor_tree_mismatch(
                    dst[key], src_value, f"{path}[{key!r}]"
                )
                if mismatch is not None:
                    return mismatch
            return None
        if isinstance(dst, (list, tuple)) or isinstance(src, (list, tuple)):
            if not isinstance(dst, type(src)) or len(dst) != len(src):
                captured_len = len(dst) if isinstance(dst, (list, tuple)) else "n/a"
                runtime_len = len(src) if isinstance(src, (list, tuple)) else "n/a"
                return (
                    f"{path}: sequence mismatch "
                    f"(captured={type(dst).__name__}[{captured_len}], "
                    f"runtime={type(src).__name__}[{runtime_len}])"
                )
            for i, (dst_value, src_value) in enumerate(zip(dst, src)):
                mismatch = UBatchWrapper._copy_tensor_tree_mismatch(
                    dst_value, src_value, f"{path}[{i}]"
                )
                if mismatch is not None:
                    return mismatch
            return None
        if is_dataclass(dst) or is_dataclass(src):
            if not is_dataclass(dst) or not is_dataclass(src):
                return (
                    f"{path}: dataclass type mismatch "
                    f"(captured={UBatchWrapper._describe_tree_value(dst)}, "
                    f"runtime={UBatchWrapper._describe_tree_value(src)})"
                )
            for field in fields(dst):
                # These are large static config/module references or scalar
                # dispatch keys. Tensor metadata below is refreshed separately.
                if field.name in (
                    "no_compile_layers",
                    "batch_descriptor",
                    "all_moe_layers",
                    "additional_kwargs",
                ):
                    continue
                if not hasattr(src, field.name):
                    return (
                        f"{path}.{field.name}: field missing from runtime "
                        f"{type(src).__name__}"
                    )
                dst_value = getattr(dst, field.name)
                src_value = getattr(src, field.name)
                if isinstance(dst_value, (int, bool, float, str, slice)) or isinstance(
                    src_value, (int, bool, float, str, slice)
                ):
                    if not isinstance(dst_value, type(src_value)):
                        return (
                            f"{path}.{field.name}: scalar type mismatch "
                            f"(captured={UBatchWrapper._describe_tree_value(dst_value)}, "
                            f"runtime={UBatchWrapper._describe_tree_value(src_value)})"
                        )
                    if field.name == "max_seq_len":
                        setattr(dst, field.name, src_value)
                        continue
                    if dst_value != src_value:
                        return (
                            f"{path}.{field.name}: scalar mismatch "
                            f"(captured={dst_value!r}, runtime={src_value!r})"
                        )
                    continue
                mismatch = UBatchWrapper._copy_tensor_tree_mismatch(
                    dst_value,
                    src_value,
                    f"{path}.{field.name}",
                )
                if mismatch is not None:
                    return mismatch
            return None
        return None

    @staticmethod
    def _breakable_cudagraph_key(
        ubatch_metadata: list[UbatchMetadata],
    ) -> tuple[BatchDescriptor, ...]:
        key = []
        for metadata in ubatch_metadata:
            batch_descriptor = metadata.context.forward_context.batch_descriptor
            if batch_descriptor is None:
                batch_descriptor = BatchDescriptor(num_tokens=metadata.num_tokens)
            key.append(batch_descriptor)
        return tuple(key)

    @classmethod
    def _refresh_breakable_metadata(
        cls,
        captured: list[UbatchMetadata],
        runtime: list[UbatchMetadata],
    ) -> str | None:
        if len(captured) != len(runtime):
            return (
                "ubatch count mismatch "
                f"(captured={len(captured)}, runtime={len(runtime)})"
            )
        for ubatch_id, (captured_meta, runtime_meta) in enumerate(zip(captured, runtime)):
            if captured_meta.num_tokens != runtime_meta.num_tokens:
                return (
                    f"ubatch[{ubatch_id}].num_tokens mismatch "
                    f"(captured={captured_meta.num_tokens}, "
                    f"runtime={runtime_meta.num_tokens})"
                )
            for attr in (
                "input_ids",
                "positions",
                "inputs_embeds",
                "intermediate_tensors",
            ):
                mismatch = cls._copy_tensor_tree_mismatch(
                    getattr(captured_meta, attr),
                    getattr(runtime_meta, attr),
                    f"ubatch[{ubatch_id}].{attr}",
                )
                if mismatch is not None:
                    return mismatch
            captured_context = captured_meta.context.forward_context
            runtime_context = runtime_meta.context.forward_context
            if captured_context.batch_descriptor != runtime_context.batch_descriptor:
                return (
                    f"ubatch[{ubatch_id}].batch_descriptor mismatch "
                    f"(captured={captured_context.batch_descriptor}, "
                    f"runtime={runtime_context.batch_descriptor})"
                )
            mismatch = cls._copy_tensor_tree_mismatch(
                captured_context.attn_metadata,
                runtime_context.attn_metadata,
                f"ubatch[{ubatch_id}].attn_metadata",
            )
            if mismatch is not None:
                return mismatch
            mismatch = cls._copy_tensor_tree_mismatch(
                captured_context.slot_mapping,
                runtime_context.slot_mapping,
                f"ubatch[{ubatch_id}].slot_mapping",
            )
            if mismatch is not None:
                return mismatch
            mismatch = cls._copy_tensor_tree_mismatch(
                captured_context.dp_metadata,
                runtime_context.dp_metadata,
                f"ubatch[{ubatch_id}].dp_metadata",
            )
            if mismatch is not None:
                return mismatch
            captured_context.moe_layer_index = runtime_context.moe_layer_index
        return None

    @staticmethod
    def _reset_ubatch_contexts(ubatch_metadata: list[UbatchMetadata]) -> None:
        for metadata in ubatch_metadata:
            context = metadata.context
            context.cpu_wait_event.clear()
            context.cpu_signal_event.clear()
            context.recv_hook = None

    def _capture_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        """
        Capture a cudagraph for a microbatched run.

        The logic here is somewhat complicated because we need to make sure that
        each of the ubatch threads initialize the cuda context before we start
        the graph capture.

        The flow is as follows:
        1. The main thread starts up each ubatch thread. Each thread will
        initialize its cuda context (torch.cuda.current_blas_handle())
        before going to sleep upon entering the ubatch_context.

        2. The main thread starts the graph capture and wakes up the first
        ubatch thread.

        3. Each ubatch thread runs the model to completion and returns the
        completed output tensors back to the main thread.

        4. The main thread stores the captured cudagraph along with its metadata
        and returns
        """

        @torch.inference_mode()
        def _capture_ubatch_thread(results, ubatch_metadata):
            torch.accelerator.set_device_index(self.device)
            ubatch_context = ubatch_metadata.context
            with torch.cuda.stream(ubatch_context.compute_stream):
                _ = torch.cuda.current_blas_handle()
            with torch.cuda.stream(ubatch_context.comm_stream):
                _ = torch.cuda.current_blas_handle()
            with ubatch_context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )

            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = ubatch_metadata[0].num_tokens + ubatch_metadata[1].num_tokens

        # Ubatches will manually manage the forward context, so we override
        # it to None here so we can have it restored correctly later
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_capture_ubatch_thread,
                    args=(
                        results,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready

            # Capture the cudagraph
            cudagraph_metadata = CUDAGraphMetaData(
                cudagraph=torch.cuda.CUDAGraph(),
                ubatch_metadata=ubatch_metadata,
            )
            if self.graph_pool is not None:
                set_graph_pool_id(self.graph_pool)
            else:
                set_graph_pool_id(current_platform.graph_pool_handle())

            # Sync offloader's copy stream before capture.
            # Ensure any pre-capture prefetches from offloader are complete.
            get_offloader().sync_prev_onload()

            with torch.cuda.graph(
                cudagraph_metadata.cudagraph,
                stream=compute_stream,
                pool=self.graph_pool,
            ):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for position, value in sorted(results)]
                result = _cat_ubatch_outputs(sorted_results)
                cudagraph_metadata.outputs = result
                # Join offloader's copy stream after forward to avoid unjoined
                # stream error. The last layer's start_prefetch forks copy_stream,
                # but wait_prefetch only happens in the next forward pass.
                get_offloader().join_after_forward()
            self.cudagraphs[num_tokens] = cudagraph_metadata
        return cudagraph_metadata.outputs

    def _capture_breakable_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        @torch.inference_mode()
        def _capture_breakable_ubatch_thread(results, captures, errors, metadata):
            try:
                torch.accelerator.set_device_index(self.device)
                ubatch_context = metadata.context
                with torch.cuda.stream(ubatch_context.compute_stream):
                    _ = torch.cuda.current_blas_handle()
                with torch.cuda.stream(ubatch_context.comm_stream):
                    _ = torch.cuda.current_blas_handle()

                graph_pool = self._get_breakable_graph_pool(ubatch_context.id)
                capture = BreakableCUDAGraphCapture(pool=graph_pool)
                set_graph_pool_id(graph_pool)

                with ubatch_context:
                    get_offloader().sync_prev_onload()
                    with capture:
                        model_output = model(
                            input_ids=metadata.input_ids,
                            positions=metadata.positions,
                            intermediate_tensors=metadata.intermediate_tensors,
                            inputs_embeds=metadata.inputs_embeds,
                        )
                        get_offloader().join_after_forward()
                        model_output = weak_ref_tensors(model_output)

                captures[metadata.context.id] = capture
                results.append((metadata.context.id, model_output))
            except BaseException as exc:
                errors.append(exc)
                self.ready_barrier.abort()

        results: list[tuple[int, torch.Tensor]] = []
        captures: list[BreakableCUDAGraphCapture | None] = [None] * len(ubatch_metadata)
        errors: list[BaseException] = []
        num_tokens = sum(metadata.num_tokens for metadata in ubatch_metadata)

        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_capture_breakable_ubatch_thread,
                    args=(results, captures, errors, metadata),
                )
                ubatch_threads.append(thread)
                thread.start()
            barrier_error = None
            try:
                self.ready_barrier.wait()
                ubatch_metadata[0].context.cpu_wait_event.set()
            except threading.BrokenBarrierError as exc:
                barrier_error = exc
            finally:
                for thread in ubatch_threads:
                    thread.join()

        if errors:
            raise RuntimeError("Breakable DBO CUDA graph capture failed") from errors[0]
        if barrier_error is not None:
            raise RuntimeError(
                "Breakable DBO CUDA graph capture did not reach the ubatch barrier"
            ) from barrier_error
        if len(results) != len(ubatch_metadata):
            raise RuntimeError(
                "Breakable DBO CUDA graph capture produced incomplete outputs: "
                f"got {len(results)} of {len(ubatch_metadata)} ubatches"
            )
        if not all(capture is not None for capture in captures):
            raise RuntimeError("Breakable DBO CUDA graph capture produced no graph")

        sorted_results = [value for position, value in sorted(results)]
        output = _cat_ubatch_outputs(sorted_results)
        key = self._breakable_cudagraph_key(ubatch_metadata)
        breakable_metadata = BreakableCUDAGraphMetaData(
            captures=[capture for capture in captures if capture is not None],
            ubatch_metadata=ubatch_metadata,
            outputs=sorted_results,
        )
        if not breakable_metadata.capture_log_emitted:
            logger.info(
                "Captured DBO breakable CUDA graph for %d tokens: ubatch_tokens=%s "
                "descriptors=%s graphs=%s eager_breaks=%s",
                num_tokens,
                [metadata.num_tokens for metadata in ubatch_metadata],
                key,
                [capture.num_graphs for capture in breakable_metadata.captures],
                [capture.num_eager_breaks for capture in breakable_metadata.captures],
            )
            breakable_metadata.capture_log_emitted = True
        self.breakable_cudagraphs[key] = breakable_metadata
        return output

    def _replay_breakable_ubatches(
        self,
        cudagraph_metadata: BreakableCUDAGraphMetaData,
        runtime_ubatch_metadata: list[UbatchMetadata],
    ) -> torch.Tensor | None:
        metadata_mismatch = self._refresh_breakable_metadata(
            cudagraph_metadata.ubatch_metadata,
            runtime_ubatch_metadata,
        )
        if metadata_mismatch is not None:
            if cudagraph_metadata.fallback_log_count < 8:
                logger.warning(
                    "DBO breakable CUDA graph metadata changed; falling back to "
                    "eager ubatch execution for this step. First mismatch: %s",
                    metadata_mismatch,
                )
                cudagraph_metadata.fallback_log_count += 1
            elif cudagraph_metadata.fallback_log_count == 8:
                logger.warning(
                    "Suppressing further DBO breakable CUDA graph fallback logs "
                    "for this token shape."
                )
                cudagraph_metadata.fallback_log_count += 1
            return None

        if not cudagraph_metadata.replay_log_emitted:
            logger.info(
                "Replaying DBO breakable CUDA graph: ubatch_tokens=%s "
                "descriptors=%s graphs=%s eager_breaks=%s",
                [metadata.num_tokens for metadata in cudagraph_metadata.ubatch_metadata],
                self._breakable_cudagraph_key(cudagraph_metadata.ubatch_metadata),
                [capture.num_graphs for capture in cudagraph_metadata.captures],
                [capture.num_eager_breaks for capture in cudagraph_metadata.captures],
            )
            cudagraph_metadata.replay_log_emitted = True

        if cudagraph_metadata.fallback_log_count:
            logger.info(
                "DBO breakable CUDA graph replay resumed after %d fallback(s) "
                "for this token shape.",
                cudagraph_metadata.fallback_log_count,
            )
            cudagraph_metadata.fallback_log_count = 0

        self._reset_ubatch_contexts(cudagraph_metadata.ubatch_metadata)

        @torch.inference_mode()
        def _replay_breakable_ubatch_thread(results, ubatch_metadata, capture):
            torch.accelerator.set_device_index(self.device)
            with ubatch_metadata.context:
                capture.replay()
            assert cudagraph_metadata.outputs is not None
            output = cudagraph_metadata.outputs[ubatch_metadata.context.id]
            results.append((ubatch_metadata.context.id, output))

        results: list[tuple[int, torch.Tensor]] = []
        get_offloader().sync_prev_onload()
        with override_forward_context(None):
            ubatch_threads = []
            for metadata, capture in zip(
                cudagraph_metadata.ubatch_metadata,
                cudagraph_metadata.captures,
            ):
                thread = threading.Thread(
                    target=_replay_breakable_ubatch_thread,
                    args=(results, metadata, capture),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()
            cudagraph_metadata.ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()

        sorted_results = [value for position, value in sorted(results)]
        return _cat_ubatch_outputs(sorted_results)

    def _run_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        @torch.inference_mode()
        def _ubatch_thread(results, errors, model, metadata):
            try:
                with metadata.context:
                    model_output = model(
                        input_ids=metadata.input_ids,
                        positions=metadata.positions,
                        intermediate_tensors=metadata.intermediate_tensors,
                        inputs_embeds=metadata.inputs_embeds,
                    )
                results.append((metadata.context.id, model_output))
            except BaseException as exc:
                errors.append(exc)
                self.ready_barrier.abort()

        results: list[tuple[int, torch.Tensor]] = []
        errors: list[BaseException] = []

        # Ubatch threads will manually manage the forward context, so we
        # override it to None here so we can have it restored correctly
        # after both threads have finished
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_ubatch_thread,
                    args=(
                        results,
                        errors,
                        model,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            barrier_error = None
            try:
                self.ready_barrier.wait()  # Wait for both threads to be ready
                ubatch_metadata[0].context.cpu_wait_event.set()
            except threading.BrokenBarrierError as exc:
                barrier_error = exc
            finally:
                for thread in ubatch_threads:
                    thread.join()

        if errors:
            raise RuntimeError("DBO ubatch execution failed") from errors[0]
        if barrier_error is not None:
            raise RuntimeError("DBO ubatch execution did not reach the barrier") from (
                barrier_error
            )
        if len(results) != len(ubatch_metadata):
            raise RuntimeError(
                "DBO ubatch execution produced incomplete outputs: "
                f"got {len(results)} of {len(ubatch_metadata)} ubatches"
            )
        sorted_results = [value for position, value in sorted(results)]
        result = _cat_ubatch_outputs(sorted_results)
        return result

    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        slot_mapping,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
        cudagraph_runtime_mode,
    ) -> list[UbatchMetadata]:
        # Create one forward context per ubatch
        forward_contexts = []
        # slot_mapping can be None, an empty dict (from create_forward_context
        # converting None to {}), or a list of dicts (one per ubatch)
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        for i, ubatch_slice in enumerate(ubatch_slices):
            ubatch_num_reqs = (
                ubatch_slice.request_slice.stop - ubatch_slice.request_slice.start
            )
            ubatch_batch_descriptor = None
            if batch_descriptor is not None:
                ubatch_batch_descriptor = BatchDescriptor(
                    num_tokens=ubatch_slice.num_tokens,
                    num_reqs=ubatch_num_reqs
                    if batch_descriptor.num_reqs is not None
                    else None,
                    uniform=batch_descriptor.uniform,
                    has_lora=batch_descriptor.has_lora,
                    num_active_loras=batch_descriptor.num_active_loras,
                )
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    batch_descriptor=ubatch_batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=slot_mapping[i] if has_slot_mapping else None,
                )
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop
                    - ubatch_slice.token_slice.start,
                )
            )

        return ubatch_metadata

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        sliced_input_ids = (
            ensure_tensor_alignment(input_ids[tokens_slice])
            if input_ids is not None
            else None
        )
        # if we are using mrope. Mrope adds an additional dimension to the
        # positions tensor
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_positions = ensure_tensor_alignment(sliced_positions)
        sliced_inputs_embeds = (
            inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        )
        sliced_intermediate_tensors = (
            intermediate_tensors[tokens_slice]
            if intermediate_tensors is not None
            else None
        )

        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:
            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched  cudagraph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the cudagraph wrapper will try to capture a cudagraph
            # for this shape during a normal run.
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE

            with self.sm_control:
                if self.use_breakable_cudagraphs:
                    self._log_breakable_decision(
                        "DBO breakable CUDA graph bypass: no ubatch slices. "
                        "runtime_mode=%s batch_descriptor=%s",
                        cudagraph_runtime_mode,
                        batch_descriptor,
                    )
                    return self.runnable(*args, **kwargs)
                if cudagraph_runtime_mode in (
                    CUDAGraphMode.NONE,
                    CUDAGraphMode.PIECEWISE,
                ):
                    return self.runnable(*args, **kwargs)
                else:
                    assert self.cudagraph_wrapper is not None
                    return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        slot_mapping = forward_context.slot_mapping
        num_tokens = sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices)
        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]
        compute_stream = torch.cuda.current_stream()

        dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        assert dp_metadata is not None
        ubatch_dp_metadata = []
        for ubatch_slice in ubatch_slices:
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            ubatch_num_tokens_across_dp = torch.tensor(
                [ubatch_slice.num_tokens] * dp_size, device="cpu", dtype=torch.int32
            )
            ubatch_dp_metadata.append(
                DPMetadata.make(
                    self.vllm_config.parallel_config,
                    ubatch_slice.num_tokens,
                    ubatch_num_tokens_across_dp,
                )
            )

        if self.use_breakable_cudagraphs and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            runtime_ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
            )
            breakable_key = self._breakable_cudagraph_key(runtime_ubatch_metadata)
            if breakable_key not in self.breakable_cudagraphs:
                self._log_breakable_decision(
                    "DBO breakable CUDA graph missing startup capture; "
                    "capturing lazily. runtime_mode=%s total_tokens=%d "
                    "descriptors=%s",
                    cudagraph_runtime_mode,
                    num_tokens,
                    breakable_key,
                )
                with self.sm_control:
                    return self._capture_breakable_ubatches(
                        runtime_ubatch_metadata,
                        self.runnable,
                    )
            with self.sm_control:
                replay_output = self._replay_breakable_ubatches(
                    self.breakable_cudagraphs[breakable_key],
                    runtime_ubatch_metadata,
                )
                if replay_output is not None:
                    return replay_output

            eager_ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._run_ubatches(eager_ubatch_metadata, self.runnable)
        elif self.use_breakable_cudagraphs:
            self._log_breakable_decision(
                "DBO breakable CUDA graph bypass: runtime mode is not FULL. "
                "runtime_mode=%s total_tokens=%d batch_descriptor=%s "
                "ubatch_tokens=%s",
                cudagraph_runtime_mode,
                num_tokens,
                batch_descriptor,
                [ubatch_slice.num_tokens for ubatch_slice in ubatch_slices],
            )
            eager_ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._run_ubatches(eager_ubatch_metadata, self.runnable)
        elif (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._capture_ubatches(ubatch_metadata, self.runnable)
        elif (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            cudagraph_metadata = self.cudagraphs[num_tokens]
            # Sync offloader before replay - ensures any external dependencies
            # from pre-capture prefetches are satisfied.
            get_offloader().sync_prev_onload()
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
        else:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._run_ubatches(ubatch_metadata, self.runnable)
