# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.all2all import PplxGardenAll2AllHandle
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
)

logger = logging.getLogger(__name__)
_GPU_ROUTE_THRESHOLD = 1_000_000


class PplxGardenPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    _next_kernel_slot: int = 0

    """
    Prepare/Finalize using PPLX Garden's CXI/RDMA P2P all-to-all.

    This first integration intentionally targets the GH200/CXI path we are
    benchmarking: unquantized activations, TP=1, and synchronous dispatch/combine.
    """

    def __init__(
        self,
        handle: PplxGardenAll2AllHandle,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_local_experts: int,
        rank_expert_offset: int,
    ) -> None:
        super().__init__()
        self.handle = handle
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        self.num_local_experts = num_local_experts
        self.rank_expert_offset = rank_expert_offset
        self._original_topk_ids: dict[int, torch.Tensor] = {}
        self._original_topk_weights: dict[int, torch.Tensor] = {}
        self._kernel_slots: dict[int, int] = {}
        self._async_enabled = os.environ.get("VLLM_PPLX_ENABLE_DBO", "0").lower() in (
            "1",
            "true",
            "yes",
        )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def _claim_kernel_for_ubatch(self, ubatch_id: int):
        if hasattr(self.handle, "acquire_kernel_slot"):
            slot, kernel = self.handle.acquire_kernel_slot()
            self._kernel_slots[ubatch_id] = slot
            return kernel
        if hasattr(self.handle, "kernels"):
            slot = PplxGardenPrepareAndFinalize._next_kernel_slot % len(self.handle.kernels)
            PplxGardenPrepareAndFinalize._next_kernel_slot += 1
            self._kernel_slots[ubatch_id] = slot
            return self.handle.kernels[slot]
        return self.handle.kernel

    def _kernel_for_ubatch(self, ubatch_id: int):
        slot = self._kernel_slots.get(ubatch_id)
        if slot is not None and hasattr(self.handle, "kernels"):
            return self.handle.kernels[slot]
        if hasattr(self.handle, "kernel_for_ubatch"):
            return self.handle.kernel_for_ubatch(ubatch_id)
        return self.handle.kernel

    def _release_kernel_for_ubatch(self, ubatch_id: int) -> None:
        slot = self._kernel_slots.pop(ubatch_id, None)
        if slot is not None and hasattr(self.handle, "release_kernel_slot"):
            self.handle.release_kernel_slot(slot)

    def supports_async(self) -> bool:
        return self._async_enabled

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
        prepare_ret = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=defer_input_quant,
        )
        if isinstance(prepare_ret, tuple):
            hook, receiver = prepare_ret
        else:
            hook, receiver = None, prepare_ret
        if hook is not None:
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
    ) -> tuple[Callable[[], None] | None, mk.ReceiverType] | mk.ReceiverType:
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
        kernel = self._claim_kernel_for_ubatch(ubatch_id)
        original_topk_ids = topk_ids.to(torch.uint32).contiguous()
        original_topk_weights = (
            torch.ones_like(topk_weights)
            if apply_router_weight_on_input
            else topk_weights
        ).to(torch.float32).contiguous()
        self._original_topk_ids[ubatch_id] = original_topk_ids
        self._original_topk_weights[ubatch_id] = original_topk_weights

        expert_num_tokens = torch.empty(
            (self.num_local_experts,), dtype=torch.int32, device=a1.device
        )
        expert_x = torch.empty(
            (self.handle.max_recv_tokens, a1.shape[1]), dtype=a1.dtype, device=a1.device
        )
        dp_x = a1.contiguous()
        kernel.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=None,
            dp_x=dp_x,
            dp_x_scale=None,
            indices=original_topk_ids,
            weights=original_topk_weights,
            do_send=True,
            do_recv=False,
        )
        send_done_event = torch.cuda.Event()
        send_done_event.record(torch.cuda.current_stream(a1.device))

        recv_done = False

        def hook() -> None:
            nonlocal recv_done
            if recv_done:
                return
            torch.cuda.current_stream(a1.device).wait_event(send_done_event)
            kernel.dispatch(
                out_expert_num_tokens=expert_num_tokens,
                out_expert_x=expert_x,
                out_expert_x_scale=None,
                dp_x=dp_x,
                dp_x_scale=None,
                indices=original_topk_ids,
                weights=original_topk_weights,
                do_send=False,
                do_recv=True,
            )
            recv_done = True

        def receiver() -> mk.PrepareResultType:
            hook()
            active_rows = int(expert_num_tokens.sum().item())
            active_expert_x = expert_x[:active_rows]
            fake_topk_ids, fake_topk_weights = self._make_local_expert_routing(
                expert_num_tokens, active_rows, topk_ids.dtype
            )
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
            )
            return (
                active_expert_x,
                None,
                expert_tokens_meta,
                fake_topk_ids,
                fake_topk_weights,
            )

        return hook, receiver

    def _make_local_expert_routing(
        self, expert_num_tokens: torch.Tensor, rows: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dtype not in (torch.int32, torch.int64):
            dtype = torch.int64
        fake_topk_ids = torch.full(
            (rows, 1),
            self.rank_expert_offset,
            dtype=dtype,
            device=expert_num_tokens.device,
        )
        fake_topk_weights = torch.zeros(
            (rows, 1), dtype=torch.float32, device=expert_num_tokens.device
        )

        if rows > _GPU_ROUTE_THRESHOLD:
            offset = 0
            for expert, count in enumerate(expert_num_tokens.detach().cpu().tolist()):
                end = offset + int(count)
                if end > offset:
                    fake_topk_ids[offset:end, 0] = self.rank_expert_offset + expert
                    fake_topk_weights[offset:end, 0] = 1
                offset = end
            return fake_topk_ids, fake_topk_weights

        row_ids = torch.arange(rows, dtype=torch.int64, device=expert_num_tokens.device)
        expert_offsets = torch.cumsum(expert_num_tokens.to(torch.int64), dim=0)
        valid_rows = row_ids < expert_offsets[-1]
        local_experts = torch.searchsorted(expert_offsets, row_ids + 1)
        fake_topk_ids[:, 0] = (local_experts + self.rank_expert_offset).to(dtype)
        fake_topk_weights[:, 0] = valid_rows.to(torch.float32)
        return fake_topk_ids, fake_topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        finalize_ret = self.finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        if isinstance(finalize_ret, tuple):
            hook, receiver = finalize_ret
        else:
            hook, receiver = None, finalize_ret
        if hook is not None:
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
    ) -> tuple[Callable[[], None] | None, Callable[[], None]] | Callable[[], None]:
        del apply_router_weight_on_input, topk_weights, topk_ids, weight_and_reduce_impl
        ubatch_id = dbo_current_ubatch_id()
        kernel = self._kernel_for_ubatch(ubatch_id)
        assert ubatch_id in self._original_topk_ids
        assert ubatch_id in self._original_topk_weights
        original_topk_ids = self._original_topk_ids[ubatch_id]
        original_topk_weights = self._original_topk_weights[ubatch_id]

        if fused_expert_output.ndim == 3:
            assert fused_expert_output.shape[1] == 1
            expert_y = fused_expert_output[:, 0, :]
        else:
            expert_y = fused_expert_output

        expert_y_send = expert_y.contiguous()
        kernel.combine(
            out_tokens=output,
            indices=original_topk_ids,
            weights=original_topk_weights,
            expert_y=expert_y_send,
            do_send=True,
            do_recv=False,
        )
        send_done_event = torch.cuda.Event()
        send_done_event.record(torch.cuda.current_stream(output.device))

        recv_done = False

        def hook() -> None:
            nonlocal recv_done
            if recv_done:
                return
            torch.cuda.current_stream(output.device).wait_event(send_done_event)
            kernel.combine(
                out_tokens=output,
                indices=original_topk_ids,
                weights=original_topk_weights,
                expert_y=expert_y_send,
                do_send=False,
                do_recv=True,
            )
            recv_done = True
            self._original_topk_ids.pop(ubatch_id, None)
            self._original_topk_weights.pop(ubatch_id, None)
            self._release_kernel_for_ubatch(ubatch_id)

        return hook, lambda: None
