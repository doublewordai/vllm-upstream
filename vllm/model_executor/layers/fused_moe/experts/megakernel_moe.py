# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 block-quantized MoE experts computed by the megakernel (github.com/doublewordai/megakernel):
one persistent kernel per layer per GPU that also performs the dispatch and the combine over CXI.
Selected with ``--enable-expert-parallel --all2all-backend megakernel`` on GH200 nodes.

MEGAKERNEL_W13_FORMAT / MEGAKERNEL_W2_FORMAT (``fp8``, default, or ``mxfp4``) choose the weight
format the kernel reads: ``mxfp4`` re-quantizes the fp8 checkpoint experts at load into the
kernel's packed MXFP4 (``megakernel.weights.pack_mxfp4``), halving the expert weight bytes.
"""

import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.megakernel import (
    ACT_FORMAT,
    megakernel_transport_kwargs,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform

W13_FORMAT = os.environ.get("MEGAKERNEL_W13_FORMAT", "fp8")
W2_FORMAT = os.environ.get("MEGAKERNEL_W2_FORMAT", "fp8")


def _dequant_blocks(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp8 [N, K] with per-128x128 block scales [N/128, K/128] -> fp32 [N, K]."""
    N, K = w.shape
    return (
        w.float().view(N // 128, 128, K // 128, 128) * s.view(N // 128, 1, K // 128, 1)
    ).reshape(N, K)


def _pack_experts(w: torch.Tensor, scale: torch.Tensor):
    """[G, N, K] fp8 block-quantized experts -> packed MXFP4 (bq [G, N/2, K] uint8, residual
    exponents sfq [G, K/128, N, 4] uint8, block scales sfb [G, N/128, K/128] fp32)."""
    from megakernel.weights import pack_mxfp4, quantize_mxfp4

    G, N, K = w.shape
    bq = torch.empty((G, N // 2, K), dtype=torch.uint8, device=w.device)
    sfq = torch.empty((G, K // 128, N, 4), dtype=torch.uint8, device=w.device)
    sfb = torch.empty_like(scale)
    for g in range(G):
        codes, e8m0 = quantize_mxfp4(_dequant_blocks(w[g], scale[g]))
        bq[g], sfq[g], sfb[g] = pack_mxfp4(codes, e8m0, int8=(ACT_FORMAT == "int8"))
    return bq, sfq, sfb


def convert_to_megakernel_format(layer, w13, w2, w13_scale, w2_scale):
    """Weight preparation after loading: gate/up rows interleaved per 128-row block (the fused
    GEMM1 + SwiGLU works on one tile), scales contiguous fp32, and the experts re-quantized to
    packed MXFP4 when the format asks for it (residual exponents kept on the layer)."""
    from megakernel import interleave_gate_up_inplace

    interleave_gate_up_inplace(w13, w13_scale)
    w13_scale = w13_scale.contiguous().float()
    w2_scale = w2_scale.contiguous().float()
    if W13_FORMAT == "mxfp4":
        w13, layer.megakernel_w13_sfq, w13_scale = _pack_experts(w13, w13_scale)
    if W2_FORMAT == "mxfp4":
        w2, layer.megakernel_w2_sfq, w2_scale = _pack_experts(w2, w2_scale)
    return w13, w2, w13_scale, w2_scale


_kernels: dict = {}


def _shared_kernel(transport, intermediate_size: int, w13_format: str, w2_format: str):
    """One kernel object (activation slabs, flags, cubin) per geometry, shared by every MoE layer
    of the process: the layers run one at a time and only the weights differ."""
    key = (id(transport), intermediate_size, w13_format, w2_format, ACT_FORMAT)
    if key not in _kernels:
        from megakernel import Megakernel

        _kernels[key] = Megakernel(
            transport, intermediate_size, w2_format=w2_format, w13_format=w13_format, act_format=ACT_FORMAT
        )
    return _kernels[key]


def _unpack_nibbles(w: torch.Tensor) -> torch.Tensor:
    """MXFP4 [.., K/2] uint8 (element 2j in the low nibble) -> codes [.., K] uint8."""
    return torch.stack((w & 0x0F, w >> 4), dim=-1).reshape(*w.shape[:-1], w.shape[-1] * 2)


def _repack_experts(codes: torch.Tensor, e8m0: torch.Tensor):
    """[G, N, K] E2M1 codes + [G, N, K/32] E8M0 -> the kernel's packed MXFP4 (bq, sfq, sfb), no
    re-quantization: only the per-expert exponent clamp of megakernel.weights.pack_mxfp4."""
    from megakernel.weights import pack_mxfp4

    G, N, K = codes.shape
    bq = torch.empty((G, N // 2, K), dtype=torch.uint8, device=codes.device)
    sfq = torch.empty((G, K // 128, N, 4), dtype=torch.uint8, device=codes.device)
    sfb = torch.empty((G, N // 128, K // 128), dtype=torch.float32, device=codes.device)
    for g in range(G):
        bq[g], sfq[g], sfb[g] = pack_mxfp4(codes[g], e8m0[g], int8=(ACT_FORMAT == "int8"))
    return bq, sfq, sfb


def convert_mxfp4_to_megakernel_format(layer, w13, w2, w13_scale, w2_scale):
    """MXFP4 checkpoint experts (vLLM layout: [G, N, K/2] packed nibbles, [G, N, K/32] E8M0 as
    uint8) -> the kernel's packed format, gate/up rows interleaved per 128-row block.  Returns
    (w13, w2, w13_scale, w2_scale) with the block scales as fp32; residual exponents on the layer."""
    c13 = _unpack_nibbles(w13.data)
    e13 = w13_scale.data.view(torch.uint8)
    I = c13.shape[1] // 2
    il = torch.empty_like(c13); il[:, 0::2] = c13[:, :I]; il[:, 1::2] = c13[:, I:]
    ie = torch.empty_like(e13); ie[:, 0::2] = e13[:, :I]; ie[:, 1::2] = e13[:, I:]
    w13, layer.megakernel_w13_sfq, w13_scale = _repack_experts(il, ie)
    w2, layer.megakernel_w2_sfq, w2_scale = _repack_experts(_unpack_nibbles(w2.data), w2_scale.data.view(torch.uint8))
    return w13, w2, w13_scale, w2_scale


class MegakernelExperts(mk.FusedMoEExpertsModular):
    """fp8 block-quantized experts (optionally re-quantized to MXFP4 at load, see the module doc)."""

    w13_format = W13_FORMAT
    w2_format = W2_FORMAT

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config, quant_config)
        manager = get_ep_group().device_communicator.all2all_manager
        transport = manager.get_handle(megakernel_transport_kwargs(moe_config))
        self.kernel = _shared_kernel(
            transport, moe_config.intermediate_size_per_partition, self.w13_format, self.w2_format
        )
        self.w13_sfq, self.w2_sfq = getattr(quant_config, "megakernel_sfq", (None, None))
        assert (self.w13_sfq is not None) == (self.w13_format == "mxfp4")
        assert (self.w2_sfq is not None) == (self.w2_format == "mxfp4")

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and current_platform.is_device_capability(90)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return weight_key == kFp8Static128BlockSym and activation_key == kFp8Dynamic128Sym

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.use_megakernel_kernels

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Packed MXFP4 weights carry two rows per byte row: sizes come from the kernel geometry.
        return w1.shape[0], a1.shape[0], 2 * self.kernel.I, self.kernel.t.H, topk_ids.shape[1]

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The kernel keeps its own activation slabs; only the reduced output is handed back.
        return ((0,), (0,), (M, K))

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert not apply_router_weight_on_input
        assert a1q_scale is not None
        assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        # topk_ids are global expert ids: the kernel routes rank = id // num_local_experts.
        self.kernel.forward(
            hidden_states,
            a1q_scale,
            topk_ids,
            topk_weights,
            hidden_states.shape[0],
            w1,
            self.quant_config.w1_scale,
            w2,
            self.quant_config.w2_scale,
            w2_sfq=self.w2_sfq,
            w13_sfq=self.w13_sfq,
            out=output,
        )


class MegakernelMxfp4Experts(MegakernelExperts):
    """MXFP4 checkpoint experts (DeepSeek-V4: E2M1 codes with UE8M0 group-32 scales), repacked at
    load into the kernel's packed MXFP4; activations still fp8 per-128-group."""

    w13_format = "mxfp4"
    w2_format = "mxfp4"

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

        return weight_key == kMxfp4Static and activation_key == kFp8Dynamic128Sym
