# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.utils.platform_utils import num_compute_units


_ZERO_COMM_SM_BACKENDS = frozenset({"deepep_low_latency", "nixl_ep"})


def get_all2all_manager_for_sm_control(vllm_config: VllmConfig) -> Any | None:
    if not vllm_config.parallel_config.enable_expert_parallel:
        return None

    try:
        from vllm.distributed import get_ep_group

        ep_group = get_ep_group()
    except AssertionError:
        return None

    device_communicator = ep_group.device_communicator
    if device_communicator is None:
        return None
    return device_communicator.all2all_manager


def get_ubatch_comm_sms(
    vllm_config: VllmConfig,
    total_sms: int,
    all2all_manager: Any | None = None,
) -> int:
    if not vllm_config.parallel_config.use_ubatching:
        return 0

    comm_sms = envs.VLLM_DBO_COMM_SMS
    if comm_sms <= 0:
        return 0

    if all2all_manager is None:
        all2all_manager = get_all2all_manager_for_sm_control(vllm_config)

    if all2all_manager is not None:
        max_sms_used = all2all_manager.max_sms_used()
        if max_sms_used is not None:
            comm_sms = min(comm_sms, max_sms_used)
    elif vllm_config.parallel_config.all2all_backend in _ZERO_COMM_SM_BACKENDS:
        comm_sms = 0

    assert 0 <= comm_sms < total_sms, (
        f"Invalid ubatch comm SM count: comm_sms={comm_sms}, "
        f"total_sms={total_sms}"
    )
    return comm_sms


def get_ubatch_compute_sms(
    vllm_config: VllmConfig,
    device_index: int | None,
    all2all_manager: Any | None = None,
) -> int:
    total_sms = num_compute_units(device_index)
    comm_sms = get_ubatch_comm_sms(vllm_config, total_sms, all2all_manager)
    return total_sms - comm_sms
