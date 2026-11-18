# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers' loader (converters + quantization plan) on a stream of weights."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.integrations.metal_quantization import _affine_quantize_tensor
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING

from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.model_executor.models.transformers.conversion import (
    VllmFormatQuantizer,
    convert_weights,
    dequantizable,
    quantization_plan,
    register_vllm_formats,
    streamable_converters,
)

register_vllm_formats()
VLLM_ONLY = [
    m for m in QUANTIZATION_METHODS if AUTO_QUANTIZER_MAPPING[m] is VllmFormatQuantizer
]


def _convert(model_type, weights):
    converters = streamable_converters(get_checkpoint_conversion_mapping(model_type))
    return dict(convert_weights(weights, converters))


def test_split_fused_qkv():
    # HRM's checkpoint fuses gate, q, k and v
    gqkv = torch.randn(16, 4)
    out = _convert(
        "hrm_text",
        [
            ("model.layers.0.attn.gqkv_proj.weight", gqkv),
            ("model.layers.0.input_layernorm.weight", torch.ones(4)),
        ],
    )
    prefix = "model.layers.0.self_attn."
    for i, name in enumerate(["gate_proj", "q_proj", "k_proj", "v_proj"]):
        torch.testing.assert_close(
            out[f"{prefix}{name}.weight"], gqkv[4 * i : 4 * i + 4]
        )
    assert "model.layers.0.input_layernorm.weight" in out  # untouched
    assert len(out) == 5


def test_concatenate_waits_for_every_source():
    # OLMo-Hybrid's checkpoint has one conv per q, k, v, interleaved with other keys
    prefix = "model.layers.0.linear_attn."
    q, k, v = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)
    stream = [
        (f"{prefix}q_conv1d.weight", q),
        ("model.norm.weight", torch.ones(3)),
        (f"{prefix}k_conv1d.weight", k),
        (f"{prefix}v_conv1d.weight", v),
    ]
    out = list(
        convert_weights(
            stream,
            streamable_converters(get_checkpoint_conversion_mapping("olmo_hybrid")),
        )
    )
    # The unrelated key is not held back behind the pending group
    assert [name for name, _ in out] == ["model.norm.weight", f"{prefix}conv1d.weight"]
    torch.testing.assert_close(out[1][1], torch.cat([q, k, v]))


def test_expert_merges_left_to_fused_moe():
    converters = streamable_converters(get_checkpoint_conversion_mapping("mixtral"))
    assert converters == []
    name = "model.layers.0.mlp.experts.3.gate_proj.weight"
    assert _convert("mixtral", [(name, torch.ones(1))]).keys() == {name}


@pytest.mark.parametrize("method", VLLM_ONLY)
def test_vllm_only_format_kept_serialized(method):
    """A format only vLLM reads: in our plan, its tensors reach vLLM's kernels as
    they are, through the model's converters (the fused qkv split carries the scale)"""
    hf_quantizer = quantization_plan(
        {"quant_method": method, "group_size": 128}, serialized=True
    )
    assert type(hf_quantizer.quantizer) is VllmFormatQuantizer
    conversions = hf_quantizer.update_weight_conversions(
        get_checkpoint_conversion_mapping("hrm_text"), None
    )
    weight = torch.randint(-128, 127, (16, 4), dtype=torch.int8)
    scale = torch.rand(16, 1)
    out = dict(
        convert_weights(
            [
                ("model.layers.0.attn.gqkv_proj.weight", weight),
                ("model.layers.0.attn.gqkv_proj.weight_scale", scale),
                ("model.norm.weight", torch.ones(4)),
            ],
            streamable_converters(conversions),
            hf_quantizer=hf_quantizer,
        )
    )
    q = "model.layers.0.self_attn.q_proj."
    torch.testing.assert_close(out[f"{q}weight"], weight[4:8])
    torch.testing.assert_close(out[f"{q}weight_scale"], scale[4:8])
    assert len(out) == 9


def test_modelopt_mixed_with_vllm_only_algos():
    """ModelOpt's per-layer algos we lack (INT8_PB_WO, FP8_PER_CHANNEL_PER_TOKEN)
    join our plan; every tensor keeps its checkpoint name for vLLM's kernels"""
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
    conversions = hf_quantizer.update_weight_conversions([], model)
    for proj, algo in layers.items():
        quantizer = hf_quantizer.get_quantizer(f"model.layers.0.self_attn.{proj}")
        assert getattr(quantizer.quantization_config, "quant_algo", algo) == algo

    stream = [
        (f"model.layers.0.self_attn.{proj}.{name}", torch.ones(1))
        for proj in layers
        for name in ("weight", "weight_scale", "input_scale")
    ]
    out = list(
        convert_weights(
            stream, streamable_converters(conversions), config, hf_quantizer
        )
    )
    assert [name for name, _ in out] == [name for name, _ in stream]


def test_transformers_only_format_dequantized():
    """A format only Transformers reads (MLX's affine, `metal`): vLLM has no kernels,
    our ops dequantize it and vLLM loads bf16"""
    quant_cfg = {"quant_method": "metal", "bits": 4, "group_size": 32}
    assert dequantizable(quant_cfg)
    assert not dequantizable({"quant_method": "fp8"})  # vLLM's kernels take it

    hf_quantizer = quantization_plan(quant_cfg, serialized=False)
    conversions = hf_quantizer.update_weight_conversions([], None)
    weight = torch.randn(8, 64, dtype=torch.bfloat16)
    packed, scales, biases = _affine_quantize_tensor(weight, 32, 4)
    scales, biases = scales.bfloat16(), biases.bfloat16()  # as MLX saves them
    q = "model.layers.0.self_attn.q_proj."
    stream = [
        (f"{q}weight", packed),
        ("model.norm.weight", torch.ones(64)),
        (f"{q}scales", scales),
        (f"{q}qbiases", biases),
    ]
    out = dict(
        convert_weights(
            stream,
            streamable_converters(conversions),
            hf_quantizer=hf_quantizer,
            checkpoint_keys=[name for name, _ in stream],
        )
    )
    assert out.keys() == {"model.norm.weight", f"{q}weight"}
    assert out[f"{q}weight"].dtype == torch.bfloat16
    torch.testing.assert_close(out[f"{q}weight"], weight, atol=0.3, rtol=0.1)
