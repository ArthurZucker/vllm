# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How each model's checkpoint becomes its params: Transformers `WeightTransform`s.

One explicit list per model, keyed by its vLLM class name (or HF `model_type`), like
`transformers.conversion_mapping`. A `WeightRenaming` renames keys, a `WeightConverter`
runs Transformers ops on them. A pattern without a param suffix (`self_attn.q_proj`)
takes all of that module's tensors (weight, bias, scales), each into its own target
(`qkv_proj.weight`, `qkv_proj.weight_scale`, ...).

Tensor parallelism is not written here: every source is sharded on read, as its
target param is (see `ops_loader.VllmShard`), before the ops run.

To add a model: copy the closest entry, run it, read the "match no param" warning.
"""

from transformers.core_model_loading import (
    Concatenate,
    MergeModulelist,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)

from vllm.model_executor.model_loader.weight_ops import OutputConcatenate


def fused_attention(prefix: str = "self_attn") -> list[WeightTransform]:
    return [
        WeightConverter(
            [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            f"{prefix}.qkv_proj",
            operations=[OutputConcatenate()],
        )
    ]


def fused_mlp(prefix: str = "mlp") -> list[WeightTransform]:
    return [
        WeightConverter(
            [f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            f"{prefix}.gate_up_proj",
            operations=[OutputConcatenate()],
        )
    ]


def experts(
    prefix: str = "mlp.experts",
    gate: str = "gate_proj",
    up: str = "up_proj",
    down: str = "down_proj",
) -> list[WeightTransform]:
    """Per-expert checkpoint tensors into the fused `w13_*` / `w2_*` params"""
    return [
        WeightConverter(
            [f"{prefix}.*.{gate}.weight", f"{prefix}.*.{up}.weight"],
            f"{prefix}.w13_weight",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            f"{prefix}.*.{down}.weight",
            f"{prefix}.w2_weight",
            operations=[MergeModulelist(dim=0)],
        ),
    ]


DENSE = [*fused_attention(), *fused_mlp()]
MOE = [*fused_attention(), *experts()]

_CONVERSION_MAPPING: dict[str, list[WeightTransform]] = {
    "LlamaForCausalLM": DENSE,
    "MistralForCausalLM": DENSE,
    "Qwen2ForCausalLM": DENSE,
    "Qwen3ForCausalLM": DENSE,
    "Qwen2MoeForCausalLM": [*MOE, *fused_mlp("mlp.shared_expert")],
    "Qwen3MoeForCausalLM": MOE,
    "OlmoeForCausalLM": MOE,
    "MixtralForCausalLM": [
        *fused_attention(),
        *experts("block_sparse_moe.experts", "w1", "w3", "w2"),
    ],
    "GptOssForCausalLM": [
        WeightRenaming("self_attn", "attn"),
        WeightRenaming("embed_tokens", "embedding"),
        *fused_attention("attn"),
        # Experts fused on disk (MXFP4 blocks + scales), gate/up interleaved: their
        # contiguous TP slice keeps both, so they are only renamed
        WeightRenaming("mlp.experts.gate_up_proj_blocks", "mlp.experts.w13_weight"),
        WeightRenaming(
            "mlp.experts.gate_up_proj_scales", "mlp.experts.w13_weight_scale"
        ),
        WeightRenaming("mlp.experts.gate_up_proj_bias", "mlp.experts.w13_bias"),
        WeightRenaming("mlp.experts.down_proj_blocks", "mlp.experts.w2_weight"),
        WeightRenaming("mlp.experts.down_proj_scales", "mlp.experts.w2_weight_scale"),
        WeightRenaming("mlp.experts.down_proj_bias", "mlp.experts.w2_bias"),
    ],
}


def get_conversion_mapping(model) -> list[WeightTransform]:
    """The model's conversions, by class name then HF `model_type`; none: as is"""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    for key in (type(model).__name__, model_type):
        if key in _CONVERSION_MAPPING:
            return _CONVERSION_MAPPING[key]
    return []
