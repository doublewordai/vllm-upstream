# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare/finalize for the megakernel MoE backend (github.com/doublewordai/megakernel).

One kernel launch per MoE layer performs the dispatch over CXI, the fused GEMM1 + SwiGLU, GEMM2
and the combine, so prepare only quantizes this rank's tokens (FP8, per-128-group scales) and
finalize only hands the reduced output back.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input


def megakernel_transport_kwargs(moe: FusedMoEConfig) -> dict:
    """The transport is keyed by the layer geometry: every MoE layer of a model shares one."""
    return dict(
        hidden_size=moe.hidden_dim,
        top_k=moe.experts_per_token,
        num_local_experts=moe.num_local_experts,
        max_tokens_per_rank=moe.max_num_tokens,
    )


class MegakernelPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    def __init__(self, transport) -> None:
        super().__init__()
        self.transport = transport

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.transport.TMAX

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.transport.R

    def output_is_reduced(self) -> bool:
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
        assert not apply_router_weight_on_input, (
            "the megakernel applies the router weights in the combine"
        )
        assert not defer_input_quant
        block_shape = list(quant_config.block_shape) if quant_config.block_shape else None
        assert quant_config.quant_dtype in (torch.float8_e4m3fn, "fp8") and block_shape == [128, 128], (
            "the megakernel dispatches FP8 activations with per-128-group scales; got "
            f"quant_dtype={quant_config.quant_dtype!r} block_shape={quant_config.block_shape!r}"
        )
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            None,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=False,
            block_shape=quant_config.block_shape,
            is_scale_swizzled=False,
        )
        assert a1q_scale is not None
        return a1q, a1q_scale.contiguous(), None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        # The kernel wrote the weighted, reduced output for this rank's tokens.
        if output.data_ptr() != fused_expert_output.data_ptr():
            output.copy_(fused_expert_output)
