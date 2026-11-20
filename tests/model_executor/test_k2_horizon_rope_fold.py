# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models.k2_horizon import is_rope_weights_folding_supported


def test_is_rope_weights_folding_supported(default_vllm_config):
    dense = SimpleNamespace(quant_method=UnquantizedLinearMethod())
    quantized = SimpleNamespace(quant_method=object())

    assert is_rope_weights_folding_supported(dense) is True
    assert is_rope_weights_folding_supported(quantized) is False
    assert is_rope_weights_folding_supported(SimpleNamespace()) is False
