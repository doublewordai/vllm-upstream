# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare/finalize for the megakernel MoE backend (github.com/doublewordai/megakernel).

One kernel launch per MoE layer performs the dispatch over CXI, the fused GEMM1 + SwiGLU, GEMM2
and the combine, so prepare only quantizes this rank's tokens (FP8, per-128-group scales) and
finalize only hands the reduced output back.
"""

import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input


ACT_FORMAT = os.environ.get("MEGAKERNEL_ACT_FORMAT", "fp8")   # fp8 | int8 | int8+bf16 (int8 dispatch, bf16 SwiGLU) | bf16 (bf16 dispatch, no activation quant); non-fp8: MXFP4 weights
INT8_QUANT = os.environ.get("MEGAKERNEL_INT8_QUANT", "round")   # round (ours) | vllm (per_token_group_quant_int8, truncating)


def megakernel_transport_kwargs(moe: FusedMoEConfig) -> dict:
    """The transport is keyed by the layer geometry: every MoE layer of a model shares one.
    MEGAKERNEL_COMBINE_FORMAT (fp8 default, int8, or bf16) picks the combine payload precision; int8 is fp8-speed at ~bf16 accuracy."""
    return dict(
        hidden_size=moe.hidden_dim,
        top_k=moe.experts_per_token,
        num_local_experts=moe.num_local_experts,
        max_tokens_per_rank=moe.max_num_tokens,
        combine_format=os.environ.get("MEGAKERNEL_COMBINE_FORMAT", "fp8"),
        dispatch_format=("bf16" if os.environ.get("MEGAKERNEL_ACT_FORMAT", "fp8") == "bf16" else "int8"),
    )


def _quant_int8_groups(x: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-(token, group) int8 with round-to-nearest: scale = amax / 127, q = round(x / scale)."""
    T, H = x.shape
    g = x.float().view(T, H // group, group)
    scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 127.0
    q = torch.round(g / scale).clamp(-127, 127).to(torch.int8).view(T, H)
    return q, scale.view(T, H // group)


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
        # The kernel dispatches per-128-group activations whatever the expert weight format: fp8 e4m3
        # by default, int8 with MEGAKERNEL_ACT_FORMAT=int8 (MXFP4 weights only); the quant config
        # only carries the weight scales.
        if ACT_FORMAT == "bf16":
            # bf16 dispatch: the kernel takes bf16 activation rows directly (no per-128 quantisation).
            a1q = a1.to(torch.bfloat16).contiguous()
            a1q_scale = torch.ones((a1.shape[0], a1.shape[1] // 128), device=a1.device, dtype=torch.float32)
        elif ACT_FORMAT != "fp8":
            # vLLM's per_token_group_quant_int8 truncates toward zero (measured gain 0.991 on real activations, a
            # -0.5 LSB bias that becomes a -1.8 % per-layer MoE output gain); quantise with round-to-nearest.
            # MEGAKERNEL_INT8_QUANT=vllm keeps the vLLM op (for reproducing its effects).
            if INT8_QUANT == "vllm":
                from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_group_quant_int8
                a1q, a1q_scale = per_token_group_quant_int8(a1, 128)
            elif hasattr(self.transport.C, "quant_int8_groups") and a1.dtype == torch.bfloat16 and a1.is_contiguous():
                a1q, a1q_scale = self.transport.C.quant_int8_groups(a1)   # fused, one warp per (token, group)
            else:
                a1q, a1q_scale = _quant_int8_groups(a1, 128)
        else:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                None,
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=False,
                block_shape=[128, 128],
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
