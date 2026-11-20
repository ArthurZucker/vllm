# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers' quantization plan with vLLM's formats registered in it."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING

from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.model_executor.models.transformers.conversion import (
    VllmFormatQuantizer,
    dequantizable,
    quantization_plan,
    register_vllm_formats,
)

register_vllm_formats()
VLLM_ONLY = [
    m for m in QUANTIZATION_METHODS if AUTO_QUANTIZER_MAPPING[m] is VllmFormatQuantizer
]


@pytest.mark.parametrize("method", VLLM_ONLY)
def test_vllm_only_format_kept_serialized(method):
    """A format only vLLM reads: its tensors reach vLLM's kernels as they are"""
    hf_quantizer = quantization_plan(
        {"quant_method": method, "group_size": 128}, serialized=True
    )
    assert type(hf_quantizer.quantizer) is VllmFormatQuantizer
    assert hf_quantizer.quantizer.load_serialized


def test_modelopt_mixed_with_vllm_only_algos():
    """ModelOpt's per-layer algos Transformers lacks join the plan, one per layer"""
    layers = {
        "q_proj": "NVFP4",
        "k_proj": "FP8",
        "v_proj": "INT8_PB_WO",
        "o_proj": "FP8_PER_CHANNEL_PER_TOKEN",
    }
    quant_cfg = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            f"model.layers.0.self_attn.{proj}": {"quant_algo": algo}
            for proj, algo in layers.items()
        },
    }
    config = LlamaConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        vocab_size=128,
    )
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    hf_quantizer = quantization_plan(quant_cfg, serialized=True)
    hf_quantizer.update_weight_conversions([], model)
    for proj, algo in layers.items():
        quantizer = hf_quantizer.get_quantizer(f"model.layers.0.self_attn.{proj}")
        assert getattr(quantizer.quantization_config, "quant_algo", algo) == algo


def test_transformers_only_format_dequantized():
    """A format only Transformers reads (MLX's affine, `metal`): its ops dequantize"""
    assert dequantizable({"quant_method": "metal", "bits": 4, "group_size": 32})
    assert not dequantizable({"quant_method": "fp8"})  # vLLM's kernels take it
