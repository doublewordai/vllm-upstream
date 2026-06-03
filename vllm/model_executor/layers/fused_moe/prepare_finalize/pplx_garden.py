# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Callable
import os
import threading
import time

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.all2all import PplxGardenAll2AllHandle
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_maybe_run_recv_hook,
)

logger = init_logger(__name__)

_PPLX_EVENT_TRACE_LIMIT = 256
_pplx_event_trace: deque[str] = deque(maxlen=_PPLX_EVENT_TRACE_LIMIT)
_pplx_event_trace_lock = threading.Lock()


def _pplx_debug_timeout_seconds() -> float:
    timeout = os.environ.get("VLLM_PPLX_GARDEN_DEBUG_STATE_TIMEOUT_S")
    if timeout is None:
        timeout = os.environ.get("PPLX_GARDEN_DEBUG_STATE_TIMEOUT_S")
    if timeout is None:
        return 0.0
    try:
        return max(float(timeout), 0.0)
    except ValueError:
        logger.warning_once(
            "Ignoring invalid PPLX debug timeout %r; expected seconds.",
            timeout,
        )
        return 0.0


def _pplx_capture_trace_enabled() -> bool:
    return os.environ.get("PPLX_GARDEN_CAPTURE_TRACE") == "1"


def _pplx_event_trace_enabled() -> bool:
    return os.environ.get("PPLX_GARDEN_EVENT_TRACE") == "1"


def _tensor_trace(tensor: torch.Tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} ptr=0x{tensor.data_ptr():x} "
        f"contiguous={tensor.is_contiguous()}"
    )


def _tensor_event(tensor: torch.Tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} ptr=0x{tensor.data_ptr():x}"
    )


def _record_pplx_event(phase: str, **fields: object) -> None:
    if not _pplx_event_trace_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    event = f"{time.monotonic():.6f} phase={phase} {field_text}"
    with _pplx_event_trace_lock:
        _pplx_event_trace.append(event)


def dump_pplx_event_trace(where: str) -> None:
    with _pplx_event_trace_lock:
        events = list(_pplx_event_trace)
    if not events:
        logger.warning("PPLX Garden event trace at %s: empty", where)
        return
    logger.warning(
        "PPLX Garden event trace at %s: last %d events", where, len(events)
    )
    for event in events:
        logger.warning("PPLX Garden event: %s", event)


class PplxGardenPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using PPLX Garden's CXI/RDMA P2P all-to-all.

    This first integration intentionally targets the GH200/CXI path we are
    benchmarking: unquantized activations, TP=1, and async dispatch/combine.
    """

    def __init__(
        self,
        *,
        handle: PplxGardenAll2AllHandle | None = None,
        handle_factory: Callable[[], PplxGardenAll2AllHandle] | None = None,
        max_tokens_per_expert: int | None = None,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        assert handle is not None or handle_factory is not None
        self._handle = handle
        self._handle_factory = handle_factory
        self._handle_lock = threading.Lock()
        self._max_tokens_per_expert = max_tokens_per_expert
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        self.num_local_experts = num_local_experts
        self._dispatch_handles: dict[int, object] = {}

    def _log_debug_state(self, where: str) -> None:
        try:
            state = self.handle.get_debug_state()
        except Exception:
            logger.exception("Failed to read PPLX Garden debug state at %s", where)
            return
        logger.warning("PPLX Garden debug state at %s: %s", where, state)

    def _call_with_debug_state(
        self,
        where: str,
        fn: Callable[[], object],
    ) -> object:
        timer: threading.Timer | None = None
        timeout_s = _pplx_debug_timeout_seconds()
        if timeout_s > 0:
            timer = threading.Timer(timeout_s, self._log_debug_state, args=(where,))
            timer.daemon = True
            timer.start()
        try:
            return fn()
        except Exception:
            self._log_debug_state(f"{where} exception")
            raise
        finally:
            if timer is not None:
                timer.cancel()

    @property
    def handle(self) -> PplxGardenAll2AllHandle:
        handle = self._handle
        if handle is not None:
            return handle

        with self._handle_lock:
            handle = self._handle
            if handle is None:
                assert self._handle_factory is not None
                handle = self._handle_factory()
                self._handle = handle
        return handle

    @property
    def max_tokens_per_expert(self) -> int:
        if self._max_tokens_per_expert is not None:
            return self._max_tokens_per_expert
        return self.handle.max_tokens_per_expert

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def supports_async(self) -> bool:
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        hook, receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=defer_input_quant,
        )
        hook()
        return receiver()

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable[[], None], mk.ReceiverType]:
        del expert_map, num_experts
        if quant_config.quant_dtype is not None and not defer_input_quant:
            raise NotImplementedError(
                "pplx_garden currently dispatches unquantized activations only."
            )
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        ubatch_id = dbo_current_ubatch_id()
        assert ubatch_id not in self._dispatch_handles, (
            f"stale PPLX Garden dispatch handle for ubatch {ubatch_id}"
        )
        _record_pplx_event(
            "dispatch_enqueue_start",
            ubatch=ubatch_id,
            capturing=torch.cuda.is_current_stream_capturing(),
            a1=_tensor_event(a1),
            topk_ids=_tensor_event(topk_ids),
            topk_weights=_tensor_event(topk_weights),
            live_handles=tuple(sorted(self._dispatch_handles)),
        )
        if _pplx_capture_trace_enabled():
            logger.warning(
                "PPLX Garden capture trace: phase=dispatch_enqueue_start "
                "ubatch=%s capturing=%s a1=(%s) topk_ids=(%s) "
                "topk_weights=(%s) live_dispatch_handles=%s",
                ubatch_id,
                torch.cuda.is_current_stream_capturing(),
                _tensor_trace(a1),
                _tensor_trace(topk_ids),
                _tensor_trace(topk_weights),
                sorted(self._dispatch_handles),
            )
        original_topk_ids = topk_ids.to(torch.uint32).contiguous()
        original_topk_weights = (
            torch.ones_like(topk_weights)
            if apply_router_weight_on_input
            else topk_weights
        ).to(torch.float32).contiguous()

        expert_num_tokens = torch.empty(
            (self.num_local_experts,), dtype=torch.int32, device=a1.device
        )
        expert_x = torch.empty(
            (
                self.num_local_experts,
                self.max_tokens_per_expert,
                a1.shape[1],
            ),
            dtype=a1.dtype,
            device=a1.device,
        )
        dp_x = a1.contiguous()
        try:
            dispatch_handle = self.handle.dispatch_async(
                out_expert_num_tokens=expert_num_tokens,
                out_expert_x=expert_x,
                out_expert_x_scale=None,
                dp_x=dp_x,
                dp_x_scale=None,
                indices=original_topk_ids,
                weights=original_topk_weights,
            )
        except Exception:
            dump_pplx_event_trace(f"dispatch enqueue ubatch={ubatch_id} exception")
            self._log_debug_state(f"dispatch enqueue ubatch={ubatch_id} exception")
            raise
        self._dispatch_handles[ubatch_id] = dispatch_handle
        _record_pplx_event(
            "dispatch_enqueue_done",
            ubatch=ubatch_id,
            capturing=torch.cuda.is_current_stream_capturing(),
            dispatch_handle_id=hex(id(dispatch_handle)),
            expert_num_tokens=_tensor_event(expert_num_tokens),
            expert_x=_tensor_event(expert_x),
            dp_x=_tensor_event(dp_x),
            indices=_tensor_event(original_topk_ids),
            weights=_tensor_event(original_topk_weights),
            live_handles=tuple(sorted(self._dispatch_handles)),
        )
        if _pplx_capture_trace_enabled():
            logger.warning(
                "PPLX Garden capture trace: phase=dispatch_enqueue_done "
                "ubatch=%s capturing=%s dispatch_handle_id=0x%x "
                "expert_num_tokens=(%s) expert_x=(%s) dp_x=(%s) "
                "indices=(%s) weights=(%s) live_dispatch_handles=%s",
                ubatch_id,
                torch.cuda.is_current_stream_capturing(),
                id(dispatch_handle),
                _tensor_trace(expert_num_tokens),
                _tensor_trace(expert_x),
                _tensor_trace(dp_x),
                _tensor_trace(original_topk_ids),
                _tensor_trace(original_topk_weights),
                sorted(self._dispatch_handles),
            )

        recv_done = False

        def recv_dispatch() -> None:
            nonlocal recv_done
            if recv_done:
                return
            _record_pplx_event(
                "dispatch_recv_start",
                ubatch=ubatch_id,
                capturing=torch.cuda.is_current_stream_capturing(),
                dispatch_handle_id=hex(id(dispatch_handle)),
            )
            if _pplx_capture_trace_enabled():
                logger.warning(
                    "PPLX Garden capture trace: phase=dispatch_recv_start "
                    "ubatch=%s capturing=%s dispatch_handle_id=0x%x",
                    ubatch_id,
                    torch.cuda.is_current_stream_capturing(),
                    id(dispatch_handle),
                )
            self._call_with_debug_state(
                f"dispatch recv ubatch={ubatch_id}",
                dispatch_handle.recv,
            )
            recv_done = True
            _record_pplx_event(
                "dispatch_recv_done",
                ubatch=ubatch_id,
                capturing=torch.cuda.is_current_stream_capturing(),
                dispatch_handle_id=hex(id(dispatch_handle)),
            )
            if _pplx_capture_trace_enabled():
                logger.warning(
                    "PPLX Garden capture trace: phase=dispatch_recv_done "
                    "ubatch=%s capturing=%s dispatch_handle_id=0x%x",
                    ubatch_id,
                    torch.cuda.is_current_stream_capturing(),
                    id(dispatch_handle),
                )

        def receiver() -> mk.PrepareResultType:
            recv_dispatch()
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
            )
            return (
                expert_x,
                None,
                expert_tokens_meta,
                None,
                None,
            )

        return recv_dispatch, receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        hook, receiver = self.finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        hook()
        receiver()

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable[[], None], Callable[[], None]]:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the PPLX Garden "
            "combine kernel."
        )
        del apply_router_weight_on_input, topk_weights, topk_ids
        ubatch_id = dbo_current_ubatch_id()
        assert ubatch_id in self._dispatch_handles
        dispatch_handle = self._dispatch_handles[ubatch_id]

        if fused_expert_output.ndim == 3:
            assert fused_expert_output.shape[0] == self.num_local_experts
            assert fused_expert_output.shape[1] == self.max_tokens_per_expert
        if fused_expert_output.dtype != output.dtype:
            raise TypeError(
                "PPLX Garden combine expected expert output dtype to match "
                f"output dtype, got expert_y={fused_expert_output.dtype}, "
                f"output={output.dtype}"
            )
        if fused_expert_output.shape[-1] != output.shape[-1]:
            raise ValueError(
                "PPLX Garden combine expected expert output hidden size to "
                f"match output hidden size, got expert_y={fused_expert_output.shape[-1]}, "
                f"output={output.shape[-1]}"
            )
        logger.info_once(
            "PPLX Garden combine: output=%s %s expert_y=%s %s",
            tuple(output.shape),
            output.dtype,
            tuple(fused_expert_output.shape),
            fused_expert_output.dtype,
        )
        if (
            os.environ.get("PPLX_GARDEN_TRACE") == "1"
            and torch.cuda.is_current_stream_capturing()
        ):
            logger.warning(
                "PPLX Garden dispatch counts trace omitted during CUDA graph "
                "capture to avoid CPU/GPU copies."
            )
        elif os.environ.get("PPLX_GARDEN_TRACE") == "1":
            counts = dispatch_handle.out_expert_num_tokens.detach().cpu()
            logger.warning(
                "PPLX Garden dispatch counts: max=%s sum=%s "
                "limit_per_expert=%s counts=%s",
                int(counts.max().item()) if counts.numel() else 0,
                int(counts.sum().item()),
                self.max_tokens_per_expert,
                counts.tolist(),
            )

        if os.environ.get("PPLX_GARDEN_DEBUG_CLONE_EXPERT_Y") == "1":
            expert_y_send = fused_expert_output.contiguous().clone()
        else:
            expert_y_send = fused_expert_output.contiguous()
        if expert_y_send.ndim == 3:
            expert_y_send = expert_y_send.view(-1, expert_y_send.shape[-1])
        if os.environ.get("PPLX_GARDEN_DEBUG_SYNC_BEFORE_COMBINE") == "1":
            torch.cuda.synchronize(output.device)
        _record_pplx_event(
            "combine_enqueue_start",
            ubatch=ubatch_id,
            capturing=torch.cuda.is_current_stream_capturing(),
            dispatch_handle_id=hex(id(dispatch_handle)),
            output=_tensor_event(output),
            fused_expert_output=_tensor_event(fused_expert_output),
            expert_y_send=_tensor_event(expert_y_send),
            live_handles=tuple(sorted(self._dispatch_handles)),
        )
        if _pplx_capture_trace_enabled():
            logger.warning(
                "PPLX Garden capture trace: phase=combine_enqueue_start "
                "ubatch=%s capturing=%s dispatch_handle_id=0x%x "
                "output=(%s) fused_expert_output=(%s) expert_y_send=(%s) "
                "live_dispatch_handles=%s",
                ubatch_id,
                torch.cuda.is_current_stream_capturing(),
                id(dispatch_handle),
                _tensor_trace(output),
                _tensor_trace(fused_expert_output),
                _tensor_trace(expert_y_send),
                sorted(self._dispatch_handles),
            )
        dbo_maybe_run_recv_hook()
        try:
            combine_handle = self.handle.combine_async(
                out_tokens=output,
                dispatch_handle=dispatch_handle,
                expert_y=expert_y_send,
            )
        except Exception:
            dump_pplx_event_trace(f"combine enqueue ubatch={ubatch_id} exception")
            self._log_debug_state(f"combine enqueue ubatch={ubatch_id} exception")
            self._dispatch_handles.pop(ubatch_id, None)
            raise
        _record_pplx_event(
            "combine_enqueue_done",
            ubatch=ubatch_id,
            capturing=torch.cuda.is_current_stream_capturing(),
            dispatch_handle_id=hex(id(dispatch_handle)),
            combine_handle_id=hex(id(combine_handle)),
            live_handles=tuple(sorted(self._dispatch_handles)),
        )
        if _pplx_capture_trace_enabled():
            logger.warning(
                "PPLX Garden capture trace: phase=combine_enqueue_done "
                "ubatch=%s capturing=%s dispatch_handle_id=0x%x "
                "combine_handle_id=0x%x live_dispatch_handles=%s",
                ubatch_id,
                torch.cuda.is_current_stream_capturing(),
                id(dispatch_handle),
                id(combine_handle),
                sorted(self._dispatch_handles),
            )

        recv_done = False

        def recv_combine() -> None:
            nonlocal recv_done
            if recv_done:
                return
            _record_pplx_event(
                "combine_recv_start",
                ubatch=ubatch_id,
                capturing=torch.cuda.is_current_stream_capturing(),
                dispatch_handle_id=hex(id(dispatch_handle)),
                combine_handle_id=hex(id(combine_handle)),
            )
            if _pplx_capture_trace_enabled():
                logger.warning(
                    "PPLX Garden capture trace: phase=combine_recv_start "
                    "ubatch=%s capturing=%s dispatch_handle_id=0x%x "
                    "combine_handle_id=0x%x",
                    ubatch_id,
                    torch.cuda.is_current_stream_capturing(),
                    id(dispatch_handle),
                    id(combine_handle),
                )
            self._call_with_debug_state(
                f"combine recv ubatch={ubatch_id}",
                combine_handle.recv,
            )
            recv_done = True
            self._dispatch_handles.pop(ubatch_id, None)
            _record_pplx_event(
                "combine_recv_done",
                ubatch=ubatch_id,
                capturing=torch.cuda.is_current_stream_capturing(),
                dispatch_handle_id=hex(id(dispatch_handle)),
                combine_handle_id=hex(id(combine_handle)),
                live_handles=tuple(sorted(self._dispatch_handles)),
            )
            if _pplx_capture_trace_enabled():
                logger.warning(
                    "PPLX Garden capture trace: phase=combine_recv_done "
                    "ubatch=%s capturing=%s dispatch_handle_id=0x%x "
                    "combine_handle_id=0x%x live_dispatch_handles=%s",
                    ubatch_id,
                    torch.cuda.is_current_stream_capturing(),
                    id(dispatch_handle),
                    id(combine_handle),
                    sorted(self._dispatch_handles),
                )

        return recv_combine, lambda: None
