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


class PplxGardenPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
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
        self._dispatch_handles: dict[int, object] = {}
        self._async_enabled = os.environ.get("VLLM_PPLX_ENABLE_DBO", "0").lower() in (
            "1",
            "true",
            "yes",
        )

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
                self.handle.max_tokens_per_expert,
                a1.shape[1],
            ),
            dtype=a1.dtype,
            device=a1.device,
        )
        dp_x = a1.contiguous()
        dispatch_handle = self.handle.dispatch_async(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=None,
            dp_x=dp_x,
            dp_x_scale=None,
            indices=original_topk_ids,
            weights=original_topk_weights,
        )
        self._dispatch_handles[ubatch_id] = dispatch_handle

        def hook() -> None:
            dispatch_handle.recv()

        def receiver() -> mk.PrepareResultType:
            hook()
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

        return hook, receiver

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
        assert ubatch_id in self._dispatch_handles
        dispatch_handle = self._dispatch_handles[ubatch_id]

        if fused_expert_output.ndim == 3:
            assert fused_expert_output.shape[0] == self.num_local_experts
            assert fused_expert_output.shape[1] == self.handle.max_tokens_per_expert

        expert_y_send = fused_expert_output.contiguous()
        combine_handle = self.handle.combine_async(
            out_tokens=output,
            dispatch_handle=dispatch_handle,
            expert_y=expert_y_send,
        )

        def hook() -> None:
            combine_handle.recv()
            self._dispatch_handles.pop(ubatch_id, None)

        return hook, lambda: None
