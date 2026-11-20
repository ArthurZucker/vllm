# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint name mapping and weight loading for the ROCm Muse Spark model."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from transformers.core_model_loading import WeightConverter, rename_source_key

import vllm.models.muse_spark.amd.model as amd_model
from vllm.model_executor.model_loader import ops_loader
from vllm.model_executor.model_loader.conversion_mapping import (
    _CONVERSION_MAPPING,
    GATE_UP,
    QKV,
    _layer_experts,
)
from vllm.model_executor.model_loader.ops_loader import load_with_ops
from vllm.models.muse_spark.amd.model import MuseSparkForCausalLM
from vllm.models.muse_spark.amd.multimodal import MuseSparkForConditionalGeneration
from vllm.platforms import current_platform


def test_rocm_mapper_retains_quark_dense_and_individual_expert_scales() -> None:
    """Quark scales survive the HF name mapping; q stacks into qkv_proj."""
    layer = "model.language_model.layers.0"
    expert_scale = f"{layer}.mlp.experts.7.gate_proj.weight_scale"
    dense_scale = f"{layer}.self_attn.o_proj.weight_scale"
    stacked_dense_scale = f"{layer}.self_attn.q_proj.weight_scale"

    # what `get_conversion_mapping` gives the model: its experts' (RoutedExperts'
    # checkpoint names), its entry, then the fused projections
    experts = SimpleNamespace(
        ckpt_gate_proj_name="gate_proj",
        ckpt_up_proj_name="up_proj",
        ckpt_down_proj_name="down_proj",
    )
    model = SimpleNamespace(
        hf_to_vllm_mapper=MuseSparkForConditionalGeneration.hf_to_vllm_mapper,
        modules=lambda: [experts],
    )
    entry = _CONVERSION_MAPPING["MuseSparkForConditionalGeneration"]
    transforms = [*_layer_experts(model), *entry(model), QKV, GATE_UP]
    renamings = [t for t in transforms if not isinstance(t, WeightConverter)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]

    amd_mapped = [
        rename_source_key(name, renamings, converters)
        for name in (expert_scale, dense_scale, stacked_dense_scale)
    ]

    assert [name for name, _ in amd_mapped] == [
        "model.layers.0.feed_forward.experts.7.gate_proj.weight_scale",
        "model.layers.0.self_attn.o_proj.weight_scale",
        "model.layers.0.self_attn.qkv_proj.weight_scale",
    ]
    # the q shard: the first source of the fused qkv_proj
    assert QKV.source_patterns.index(amd_mapped[2][1]) == 0


def test_load_weights_moves_per_row_qk_scales_with_the_interleaved_rows() -> None:
    """A PTPC-FP8 q/k projection dequantizes unchanged across the row interleave."""
    head_dim, q_rows, k_rows, hidden = 8, 4 * 8, 2 * 8, 16
    torch.manual_seed(0)
    weight = torch.randn(k_rows, hidden)
    scale = torch.rand(k_rows) + 0.5
    block_scale = torch.rand(k_rows, hidden // 4)
    input_scale = torch.tensor(0.25)
    # Same length as the k rows but indexed by input channel: must not move.
    per_channel_input = torch.rand(k_rows)
    o_proj = torch.randn(hidden, q_rows)
    layer = "model.language_model.layers.3.self_attn"

    model = object.__new__(MuseSparkForCausalLM)
    model.model = SimpleNamespace(
        config=SimpleNamespace(
            head_dim=head_dim, num_attention_heads=4, num_key_value_heads=2
        )
    )
    weights = [
        (f"{layer}.k_proj.weight", weight),
        (f"{layer}.k_proj.weight_scale", scale),
        (f"{layer}.k_proj.weight_scale_block", block_scale),
        (f"{layer}.k_proj.input_scale", input_scale),
        (f"{layer}.k_proj.input_channel_scale", per_channel_input),
        (f"{layer}.o_proj.weight", o_proj),
    ]
    # what `load_with_ops` runs on the stream before its conversions
    for transform in getattr(model, "checkpoint_transforms", []):
        weights = transform(weights)
    seen = dict(weights)

    def interleave(t: torch.Tensor) -> torch.Tensor:
        return (
            t.reshape(-1, 2, head_dim // 2, *t.shape[1:])
            .transpose(1, 2)
            .reshape(t.shape)
        )

    assert torch.equal(seen[f"{layer}.k_proj.weight"], interleave(weight))
    assert torch.equal(seen[f"{layer}.k_proj.weight_scale"], interleave(scale))
    assert torch.equal(
        seen[f"{layer}.k_proj.weight_scale_block"], interleave(block_scale)
    )
    assert torch.equal(
        seen[f"{layer}.k_proj.weight"] * seen[f"{layer}.k_proj.weight_scale"][:, None],
        interleave(weight * scale[:, None]),
    )
    assert seen[f"{layer}.k_proj.input_scale"] is input_scale
    assert seen[f"{layer}.k_proj.input_channel_scale"] is per_channel_input
    assert seen[f"{layer}.o_proj.weight"] is o_proj


def test_public_entry_point_selects_rocm_model(monkeypatch) -> None:
    import vllm.models.muse_spark as muse_spark

    original_model = muse_spark.MuseSparkForCausalLM
    original_omni_model = muse_spark.MuseSparkForConditionalGeneration
    with monkeypatch.context() as patch:
        patch.setattr(current_platform, "is_rocm", lambda: True)
        reloaded = importlib.reload(muse_spark)
        assert reloaded.MuseSparkForCausalLM.__module__.endswith(".amd.model")
        assert reloaded.MuseSparkForConditionalGeneration.__module__.endswith(
            ".amd.multimodal"
        )

    reloaded = importlib.reload(muse_spark)
    assert reloaded.MuseSparkForCausalLM is original_model
    assert reloaded.MuseSparkForConditionalGeneration is original_omni_model


def test_router_weight_reload_refolds_and_preserves_planes(monkeypatch) -> None:
    from vllm.models.muse_spark.amd.model import MuseSparkMoE

    moe = MuseSparkMoE.__new__(MuseSparkMoE)
    torch.nn.Module.__init__(moe)
    moe.router = torch.nn.Linear(4, 2, bias=False)
    moe.router_output_multiplier = 0.25
    moe.moe_router = SimpleNamespace(router_output_multiplier=None)
    monkeypatch.setattr(amd_model, "_supports_router_bf16x3_device", lambda: True)
    monkeypatch.setattr(amd_model, "supports_router_bf16x3", lambda *args: True)
    # single rank: the router is replicated
    monkeypatch.setattr(ops_loader, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(ops_loader, "get_tensor_model_parallel_world_size", lambda: 1)

    weight_ptr = moe.router.weight.data_ptr()
    planes_ptr = None
    for value in (1.0, 3.0):
        raw = torch.full_like(moe.router.weight, value)
        load_with_ops(moe, iter([("router.weight", raw)]))
        moe.fold_router_output_multiplier()
        expected = raw * 0.25
        torch.testing.assert_close(moe.router.weight, expected)
        torch.testing.assert_close(moe._router_planes.float().sum(0), expected)
        assert moe.router.weight.data_ptr() == weight_ptr
        if planes_ptr is None:
            planes_ptr = moe._router_planes.data_ptr()
        assert moe._router_planes.data_ptr() == planes_ptr
        assert moe.moe_router.router_output_multiplier is None

        # Repeated materialization without loading must not rescale the weights.
        moe.fold_router_output_multiplier()
        torch.testing.assert_close(moe.router.weight, expected)
        assert moe._router_planes.data_ptr() == planes_ptr


@pytest.mark.parametrize(
    "sharing, bypass", [(False, False), (True, False), (False, True)]
)
def test_aiter_kv_workspace_only_for_cache_readers(
    monkeypatch, default_vllm_config, sharing, bypass
) -> None:
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionImpl,
        AiterFlashAttentionMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import AttentionSpec

    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.impl = AiterFlashAttentionImpl.__new__(AiterFlashAttentionImpl)
    layer.impl.sliding_window = None
    layer.impl.num_kv_heads = 1
    layer.impl.head_size = 64
    layer.kv_sharing_target_layer_name = "source" if sharing else None
    layer.kv_cache_update_bypass = bypass
    monkeypatch.setattr(
        default_vllm_config.compilation_config,
        "static_forward_context",
        {"attn": layer},
    )
    monkeypatch.setattr(
        default_vllm_config,
        "model_config",
        SimpleNamespace(
            dtype=torch.bfloat16,
            get_num_attention_heads=lambda _: 1,
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: 64,
        ),
    )
    spec = AttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    builder = AiterFlashAttentionMetadataBuilder(
        spec, ["attn"], default_vllm_config, torch.device("cpu")
    )
    assert (builder.kv_sharing_workspace is not None) == (sharing or bypass)


@pytest.mark.parametrize(
    "expert_parallel,reduce_dtype,supported",
    [(True, "fp32", True), (False, "bf16", True), (False, "fp32", False)],
)
def test_token_sharded_stream_needs_ep_or_a_bf16_tp_reduction(
    monkeypatch, expert_parallel: bool, reduce_dtype: str, supported: bool
) -> None:
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE", reduce_dtype)
    monkeypatch.setattr(envs, "VLLM_ROCM_MUSE_SP_MIN_TOKENS", 1024)
    parallel_config = SimpleNamespace(
        enable_expert_parallel=expert_parallel,
        tensor_parallel_size=8,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        use_sequence_parallel_moe=False,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(mlp_layer_types=["sparse"] * 4)
        ),
    )
    assert amd_model.token_sharded_stream_supported(vllm_config) is supported
