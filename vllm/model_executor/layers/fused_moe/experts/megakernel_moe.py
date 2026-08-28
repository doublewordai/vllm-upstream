# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 block-quantized MoE experts computed by the megakernel (github.com/doublewordai/megakernel):
one persistent kernel per layer per GPU that also performs the dispatch and the combine over CXI.
Selected with ``--enable-expert-parallel --all2all-backend megakernel`` on GH200 nodes.
"""

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


class MegakernelExperts(mk.FusedMoEExpertsModular):
    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config, quant_config)
        from megakernel import Megakernel

        manager = get_ep_group().device_communicator.all2all_manager
        transport = manager.get_handle(megakernel_transport_kwargs(moe_config))
        self.kernel = Megakernel(transport, moe_config.intermediate_size_per_partition)

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
            out=output,
        )
