# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unquantized FlashInfer TRT-LLM MoE rounds the intermediate size up before
the expert weights are allocated."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from .utils import make_dummy_moe_config


class TestUnquantizedTrtLlmPrePadding:
    @staticmethod
    def _make_method(
        intermediate: int,
        backend: UnquantizedMoeBackend = UnquantizedMoeBackend.FLASHINFER_TRTLLM,
    ) -> UnquantizedFusedMoEMethod:
        moe_config = make_dummy_moe_config(
            num_experts=2,
            hidden_dim=64,
            intermediate_size=intermediate,
        )
        method = object.__new__(UnquantizedFusedMoEMethod)
        method.moe = moe_config
        method.unquantized_backend = backend
        method.moe_kernel = None
        return method

    @pytest.mark.parametrize(
        "backend,original,expected",
        [
            (UnquantizedMoeBackend.FLASHINFER_TRTLLM, 1344, 1408),
            (UnquantizedMoeBackend.FLASHINFER_TRTLLM, 1408, 1408),
            (UnquantizedMoeBackend.TRITON, 1344, 1344),
        ],
    )
    def test_rounds_intermediate_before_weight_allocation(
        self,
        backend: UnquantizedMoeBackend,
        original: int,
        expected: int,
    ):
        method = self._make_method(original, backend)

        hidden, intermediate = method.maybe_roundup_sizes(
            hidden_size=64,
            intermediate_size_per_partition=original,
            act_dtype=torch.bfloat16,
            moe_parallel_config=method.moe.moe_parallel_config,
        )

        assert hidden == 64
        assert intermediate == expected
