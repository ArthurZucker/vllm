# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How each model's checkpoint becomes its params: Transformers `WeightTransform`s.

One explicit list per architecture, like `transformers.conversion_mapping`. Renamings
apply in order, then at most one converter: the first whose source matches. A pattern
without a param suffix (`.q_proj`) takes all of that module's tensors (weight, bias,
scales), each into its own target (`.qkv_proj.weight`, `.qkv_proj.weight_scale`, ...).

Tensor parallelism is not written here: every source is sharded on read, as its
target param is (`ops_loader.VllmShard`), before the ops run.

To add a model: copy the closest entry, load it, read the "matching no param" warning.
"""

import os
import re
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

from torch import nn
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import (
    Concatenate,
    MergeModulelist,
    Transpose,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)

from vllm.model_executor.model_loader.weight_ops import (
    Deinterleave,
    ExpertRows,
    OutputConcatenate,
    RepeatLastSource,
    SwapHalves,
    ZeroMissingSource,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


def _names(pattern: str) -> str:
    """Module names, no escaping: `.` is a dot, unless a quantifier follows (`.*`)"""
    return re.sub(r"(?<!\\)\.(?![*+?{])", r"\.", pattern)


def rename(pattern: str, new: str) -> WeightRenaming:
    return WeightRenaming(_names(pattern), new)


def drop(pattern: str) -> WeightRenaming:
    """A checkpoint tensor the model does not load"""
    return WeightRenaming(_names(pattern), "_dropped_")


class Fused(WeightConverter):
    """`WeightConverter` concatenating its sources into a fused param; `shards[i]` is
    how many of the param's shards source `i` fills (Qwen3.5's `in_proj_qkv`: 3)"""

    __slots__ = ("shards",)


def fused(target: str, *sources: str | tuple[str, int]) -> WeightConverter:
    """`sources` concatenated into the fused `target`, in its shard order; a source
    filling several shards is `(pattern, shards)`"""
    # module names, not regexes: `.k.` must not match the `ock` of `blocks`
    patterns = [re.escape(s if isinstance(s, str) else s[0]) for s in sources]
    converter = Fused(patterns, target, operations=[OutputConcatenate()])
    converter.shards = [1 if isinstance(s, str) else s[1] for s in sources]
    return converter


# Transformers' experts (`experts.gate_up_proj`, `[E, 2I, H]`: gate then up) are vLLM's
# fused experts, which `FusedMoE` keeps in its `routed_experts`
TRANSFORMERS_EXPERTS = {
    "experts.gate_up_proj": "experts.routed_experts.w13_weight",
    "experts.down_proj": "experts.routed_experts.w2_weight",
}
# Experts fused on disk (Transformers v5, gpt-oss' MXFP4 blocks), with their tensors'
# suffixes: checkpoint -> vLLM
EXPERT_SUFFIXES = {
    "_blocks": "",
    "_scales": "_scale",
    "_scale_inv": "_scale_inv",
    "_bias": "",
    "": "",
}
FUSED_EXPERTS = [
    rename(
        rf"{re.escape(hf)}{re.escape(suffix)}$",
        (vllm.removesuffix("_weight") + "_bias") if suffix == "_bias" else vllm + new,
    )
    for hf, vllm in TRANSFORMERS_EXPERTS.items()
    for suffix, new in EXPERT_SUFFIXES.items()
]


def to_vllm(conversions: list[WeightTransform]) -> list[WeightTransform]:
    """Transformers' conversions onto vLLM's params: their converters merging
    per-expert checkpoints (`MergeModulelist`) retargeted to the fused experts, and the
    renames of experts already fused on disk"""
    out: list[WeightTransform] = [*FUSED_EXPERTS]
    for conversion in conversions:
        targets = list(conversion._original_target_patterns)
        for hf, vllm in TRANSFORMERS_EXPERTS.items():
            targets = [target.replace(hf, vllm) for target in targets]
        if targets == conversion._original_target_patterns:
            out.append(conversion)
        else:
            kwargs = (
                {"operations": conversion.operations}
                if isinstance(conversion, WeightConverter)
                else {}
            )
            out.append(
                conversion.copy(conversion._original_source_patterns, targets, **kwargs)
            )
    return out


def transformers_experts(model_type: str) -> list[WeightTransform]:
    """Transformers' own converters merging per-expert checkpoints, for `model_type`"""
    return [
        conversion
        for conversion in get_checkpoint_conversion_mapping(model_type) or []
        if any(
            isinstance(op, MergeModulelist)
            for op in getattr(conversion, "operations", [])
        )
    ]


# Mistral's own checkpoint format (`consolidated.safetensors`) onto the HF names
MISTRAL_FORMAT = [
    rename(r"^layers\.", "model.layers."),
    rename(r"^norm\.weight$", "model.norm.weight"),
    rename(r"^tok_embeddings\.weight$", "model.embed_tokens.weight"),
    rename(r"^output\.weight$", "lm_head.weight"),
    rename(r"\.attention_norm\.", ".input_layernorm."),
    rename(r"\.ffn_norm\.", ".post_attention_layernorm."),
    rename(r"\.attention\.wq_a\.", ".self_attn.q_a_proj."),
    rename(r"\.attention\.q_a_norm\.", ".self_attn.q_a_layernorm."),
    rename(r"\.attention\.wq_b\.", ".self_attn.q_b_proj."),
    rename(r"\.attention\.wkv_a_with_mqa\.", ".self_attn.kv_a_proj_with_mqa."),
    rename(r"\.attention\.kv_a_norm\.", ".self_attn.kv_a_layernorm."),
    rename(r"\.attention\.wkv_b\.", ".self_attn.kv_b_proj."),
    rename(r"\.attention\.wo\.", ".self_attn.o_proj."),
    rename(r"\.feed_forward\.w1\.", ".mlp.gate_proj."),
    rename(r"\.feed_forward\.w2\.", ".mlp.down_proj."),
    rename(r"\.feed_forward\.w3\.", ".mlp.up_proj."),
    rename(r"(\.\d+)\.gate\.weight$", r"\1.mlp.gate.weight"),
    rename(r"\.shared_experts\.w1\.", ".mlp.shared_experts.gate_proj."),
    rename(r"\.shared_experts\.w2\.", ".mlp.shared_experts.down_proj."),
    rename(r"\.shared_experts\.w3\.", ".mlp.shared_experts.up_proj."),
    rename(r"\.experts\.(\d+)\.w1\.", r".mlp.experts.\1.gate_proj."),
    rename(r"\.experts\.(\d+)\.w2\.", r".mlp.experts.\1.down_proj."),
    rename(r"\.experts\.(\d+)\.w3\.", r".mlp.experts.\1.up_proj."),
    rename(r"\.qscale_act$", ".input_scale"),
    rename(r"\.qscale_weight$", ".weight_scale"),
]

# ColBERT: the projection (`linear` / `colbert_linear`), then the encoder's checkpoint
# names under `model.`
COLBERT_LINEAR = rename(r"^(.*\.)?(colbert_)?linear\.weight$", "colbert_linear.weight")

# GPT-2 / Bloom style checkpoints, saved without the `transformer.` prefix
TRANSFORMER_PREFIX = rename(r"^(?!transformer\.|lm_head\.|score\.)", "transformer.")
GPT2 = [
    TRANSFORMER_PREFIX,
    drop(".attn.bias"),  # attention mask buffers (not `c_attn.bias`)
    drop(".attn.masked_bias"),
    *(
        WeightConverter(f".{proj}.weight", f".{proj}.weight", operations=[Transpose()])
        for proj in ("c_attn", "c_proj", "c_fc")
    ),
]
BLOOM = [
    TRANSFORMER_PREFIX,
    *(
        WeightConverter(
            f".query_key_value.{tensor}",
            f".query_key_value.{tensor}",
            operations=[Deinterleave(parts=3)],
        )
        for tensor in ("weight", "bias")
    ),
]

# Whisper's k_proj ships no bias: zeros in its slot of the fused bias
WHISPER_QKV_BIAS = WeightConverter(
    [f".self_attn.{p}_proj.bias" for p in "qkv"],
    ".self_attn.qkv_proj.bias",
    operations=[ZeroMissingSource(), OutputConcatenate()],
)
# Mistral Large 4's consolidated checkpoint; a layer's modules (`attention.wq`) onto
# vLLM's (`self_attn.q_proj`), `lerope.alpha` onto the rotary embedding
_L = r"^(model\.layers\.\d+)\."
MISTRAL_LARGE4 = [
    rename("^lerope.alpha$", "model.rotary_emb.lerope_alpha"),
    drop("^mtp_"),  # the draft model's
    rename(r"^tok_embeddings(?=\.|$)", "model.embed_tokens"),
    rename(r"^norm(?=\.|$)", "model.norm"),
    rename(r"^output(?=\.|$)", "lm_head"),
    rename("^layers.", "model.layers."),
    *(
        rename(_L + old + r"(?=\.|$)", r"\1." + new)
        for old, new in (
            ("attention_norm", "input_layernorm"),
            ("ffn_norm", "post_attention_layernorm"),
            (r"attention\.indexer\.ssmax_scaling", "self_attn.indexer.ssmax"),
            (r"attention\.ssmax_scaling", "self_attn.ssmax"),
            (r"attention\.wq", "self_attn.q_proj"),
            (r"attention\.wkv_a_with_mqa", "self_attn.kv_a_proj_with_mqa"),
            (r"attention\.kv_a_norm", "self_attn.kv_a_layernorm"),
            (r"attention\.wkv_b", "self_attn.kv_b_proj"),
            (r"attention\.wo", "self_attn.o_proj"),
            (r"attention\.indexer", "self_attn.indexer"),
            ("feed_forward", "mlp"),
            ("shared_experts", "mlp.shared_experts"),
            ("gate", "mlp.gate"),
            ("experts", "mlp.experts"),  # per expert: `w1` / `w2` / `w3` (MistralMoE)
        )
    ),
    rename(".qscale_weight$", ".weight_scale_inv"),
    rename(r"\.(mlp|shared_experts)\.w2\.", r".\1.down_proj."),
    fused(".mlp.gate_up_proj", ".mlp.w1", ".mlp.w3"),
    fused(".shared_experts.gate_up_proj", ".shared_experts.w1", ".shared_experts.w3"),
    fused(".indexer.wk_weights_proj", ".indexer.wk", ".indexer.weights_proj"),
]
QKV = fused(".qkv_proj", ".q_proj", ".k_proj", ".v_proj")
GATE_UP = fused(".gate_up_proj", ".gate_proj", ".up_proj")


def _cosmos3_edge() -> list[WeightTransform]:
    """Cosmos3 Edge's Diffusers checkpoint: block `N` holds attention and MLP, which the
    Nemotron-H language model splits into layers `2N` and `2N + 1`"""
    layers = {
        "embed_tokens.": "language_model.model.embed_tokens.",
        "norm.": "language_model.model.norm_f.",
        "model.visual.": "visual.encoder.",
        "model.projector.": "visual.projector.",
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
    }
    for n in range(28):
        attention, mlp = (
            f"language_model.model.layers.{2 * n}",
            f"language_model.model.layers.{2 * n + 1}",
        )
        layers[f"layers.{n}.input_layernorm."] = f"{attention}.norm."
        for proj in ("q", "k", "v", "o"):
            layers[f"layers.{n}.self_attn.{proj}_proj."] = (
                f"{attention}.mixer.{proj}_proj."
            )
        layers[f"layers.{n}.post_attention_layernorm."] = f"{mlp}.norm."
        layers[f"layers.{n}.mlp."] = f"{mlp}.mixer."
    return [
        *(
            rename(rf"\.self_attn\.to_{a}\.", f".self_attn.{b}_proj.")
            for a, b in (("q", "q"), ("k", "k"), ("v", "v"), ("out", "o"))
        ),
        rename(r"language_model\.embeddings", "language_model.embed_tokens"),
        *(rename("^" + re.escape(old), new) for old, new in layers.items()),
        fused(".qkv_proj.", ".q_proj.", ".k_proj.", ".v_proj."),
    ]


def _own_mapper(module: nn.Module) -> list[WeightTransform]:
    """The `WeightsMapper` a model builds for itself (from its config)"""
    return from_mapper(module.hf_to_vllm_mapper)


def _extends(base: str) -> Callable:
    """A class whose mapper extends another's: that class's entry, then its own"""
    return lambda module: [*_CONVERSION_MAPPING[base], *_own_mapper(module)]


def _gemma4(module: nn.Module) -> list[WeightTransform]:
    """`attention_k_eq_v` full-attention layers ship `k_proj` only: K is loaded into
    both the K and V shards. KV-shared layers only have `q_proj`, kept as is"""
    config = module.config
    shared = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
    k_eq_v = [
        i
        for i, layer_type in enumerate(config.layer_types[:shared])
        if getattr(config, "attention_k_eq_v", False) and layer_type == "full_attention"
    ]
    return [
        *(
            WeightConverter(
                [f"layers.{i}.self_attn.q_proj", f"layers.{i}.self_attn.k_proj"],
                f"layers.{i}.self_attn.qkv_proj",
                operations=[RepeatLastSource(), OutputConcatenate()],
            )
            for i in k_eq_v
        ),
        QKV,
        GATE_UP,
    ]


def _mimo_mtp(module: nn.Module) -> list[WeightTransform]:
    """MTP layers stored 0-indexed without the `mtp_block` wrapper"""
    start, layers = (
        module.config.num_hidden_layers,
        module.config.num_nextn_predict_layers,
    )
    blocks = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
    return [
        *(rename(rf"\.{block}\.", f".mtp_block.{block}.") for block in blocks),
        *(
            rename(rf"^model\.mtp_layers\.{i}\.", f"model.mtp_layers.{i + start}.")
            for i in range(layers)
        ),
        QKV,
        GATE_UP,
    ]


def _ernie_mtp(module: nn.Module) -> list[WeightTransform]:
    """MTP weights stored as a flat `mtp_*.0.` block, after the decoder layers"""
    layer = module.config.num_hidden_layers
    parts = ("mtp_emb_norm", "mtp_hidden_norm", "mtp_linear_proj", "mtp_block")
    return [
        *(
            rename(rf"model\.{part}\.0\.", f"model.layers.{layer}.{part}.")
            for part in parts
        ),
        QKV,
        GATE_UP,
    ]


# A class's entry: its transforms, or a function of one of its modules returning them
# (when they depend on the config, e.g. an MTP layer's index)
_CONVERSION_MAPPING: dict[str, "list[WeightTransform] | Callable"] = {
    "MistralLarge4ForCausalLM": MISTRAL_LARGE4,
    "Gemma4Model": _gemma4,
    "GPT2LMHeadModel": GPT2,
    "GPT2ForSequenceClassification": GPT2,
    "BloomForCausalLM": BLOOM,
    "ColBERTModel": [
        COLBERT_LINEAR,
        rename("^bert.", "model."),
    ],
    "ColBERTJinaRobertaModel": [
        COLBERT_LINEAR,
        rename("^roberta.", "model."),
        drop("^model.pooler."),
    ],
    # Sentence-Transformers checkpoints: the encoder without its prefix
    "ColBERTLfm2Model": [
        COLBERT_LINEAR,
        rename("^(?!model.|colbert_linear.)", "model."),
    ],
    # their mapper is built per instance by a helper (TODO: return transforms)
    "BeamModel": _own_mapper,
    "DFlashQwen3Model": _own_mapper,
    "DeepseekV41ForCausalLM": _own_mapper,
    "DeepseekV41LLMForCausalLM": _own_mapper,
    "DeepseekV4ForCausalLM": _own_mapper,
    "DeepseekV4ForConditionalGeneration": _own_mapper,
    "Dots3NoteForCausalLM": _own_mapper,
    "EmbeddingGemma2Model": _own_mapper,
    "K3DSparkForCausalLM": _own_mapper,
    "MiniMaxM3SparseForCausalLM": _own_mapper,
    "MuseSparkForConditionalGeneration": _extends("MuseSparkForCausalLM"),
    "Qwen4ExpMultiTokenPredictor": _extends("Qwen3_5Model"),
    "Step5ForConditionalGeneration": _extends("Step3p7ForConditionalGeneration"),
    "MiMoMTP": _mimo_mtp,
    "ErnieMTP": _ernie_mtp,
    "AIMv2Model": [
        fused(".fc13", ".fc1", ".fc3"),
    ],
    # TODO: old loader: stacked_unparsed
    "AXK1Model": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    "AfmoeForCausalLM": [
        rename(".router.gate.weight$", ".gate.weight"),
    ],
    "AliceAIForCausalLM": [
        drop("^mtp."),
    ],
    "AliceAIModel": [
        rename(".linear_attn.a_log_bias", ".linear_attn.A_log"),
        fused(
            ".linear_attn.in_proj_qkvgfab",
            ".linear_attn.q_proj",
            ".linear_attn.k_proj",
            ".linear_attn.v_proj",
            ".linear_attn.b_proj",
            ".linear_attn.f_a_proj",
        ),
    ],
    "Alpamayo2SuperForConditionalGeneration": [
        rename("^vlm.model.visual.", "visual."),
        rename("^vlm.lm_head.", "language_model.lm_head."),
        rename("^vlm.model.language_model.", "language_model.model."),
        drop("^expert."),
    ],
    # TODO: old loader: tensor_ops
    "BagelForConditionalGeneration": [
        drop("^vit_pos_embed.pos_embed"),
    ],
    # TODO: old loader: stacked_unparsed
    # TODO: old loader: tensor_ops
    "BailingMoeV25MTPModel": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    # TODO: old loader: tensor_ops
    "BailingMoeV25Model": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    # TODO: old loader: stacked_unparsed
    # TODO: old loader: tensor_ops
    "BailingMoeV3MTPModel": [
        rename(".attention.", ".self_attn."),
        rename("^attention.", "self_attn."),
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    "BailingMoeV3VLForConditionalGeneration": [
        rename(".attention.", ".self_attn."),
        rename("^attention.", "self_attn."),
        rename("^model.visual", "visual"),
        rename("^model.linear_proj", "linear_proj"),
        rename("^linear_proj.0", "linear_proj.linear_fc1"),
        rename("^linear_proj.2", "linear_proj.linear_fc2"),
        rename("^lm_head", "language_model.lm_head"),
        rename("^model.", "language_model.model."),
    ],
    "BeamForCausalLM": [
        rename(".self_attn.gate_proj", ".self_attn.output_gate_proj"),
    ],
    "BeeForConditionalGeneration": [
        rename("^model.language_model.", "language_model.model."),
        rename("^model.vision_tower.", "vision_tower."),
        rename("^model.multi_modal_projector.", "multi_modal_projector."),
        rename("^model.image_newline", "image_newline"),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    "BertEmbeddingModel": [
        rename("^bert.", "model."),
        rename("^roberta.", "model."),
        rename("^encoder.", "model.encoder."),
        rename("^embeddings.", "model.embeddings."),
    ],
    "BertForMaskedLM": [
        drop("cls.seq_relationship"),
        drop("cls.predictions.decoder.bias"),
        rename(
            "cls.predictions.transform.LayerNorm.gamma",
            "mlm_head.layer_norm.weight",
        ),
        rename(
            "cls.predictions.transform.LayerNorm.beta",
            "mlm_head.layer_norm.bias",
        ),
        rename("cls.predictions.transform.LayerNorm", "mlm_head.layer_norm"),
        rename("cls.predictions.transform.dense", "mlm_head.dense"),
        rename("cls.predictions.decoder", "mlm_head.decoder"),
        rename("cls.predictions.bias", "mlm_head.decoder.bias"),
    ],
    "BertModel": [
        rename("LayerNorm.gamma", "LayerNorm.weight"),
        rename("LayerNorm.beta", "LayerNorm.bias"),
        drop("^pooler."),
        fused(".self.qkv_proj", ".self.query", ".self.key", ".self.value"),
    ],
    "BertPoolingModel": [
        # BertModel's, keeping the pooler
        rename("LayerNorm.gamma", "LayerNorm.weight"),
        rename("LayerNorm.beta", "LayerNorm.bias"),
        fused(".self.qkv_proj", ".self.query", ".self.key", ".self.value"),
    ],
    "BertWithRope": [
        rename("^model.", ""),
    ],
    # TODO: old loader: tensor_ops
    "BlipVisionModel": [
        fused(".qkv", ".q_proj", ".k_proj", ".v_proj"),
    ],
    "CLIPEmbeddingModel": [
        drop(".position_ids"),
    ],
    "CanaryForConditionalGeneration": [
        rename(r"(decoder.layers.\d+.\w+_attn).o_proj.", r"\1.out_projection."),
        rename(r"(decoder.layers.\d+.mlp).fc1.", r"\1.dense_in."),
        rename(r"(decoder.layers.\d+.mlp).fc2.", r"\1.dense_out."),
        fused(".encoder_attn.kv_proj", ".encoder_attn.k_proj", ".encoder_attn.v_proj"),
    ],
    "ChatGLMModel": [
        rename(".word_embeddings", ""),
    ],
    "Cohere2MoeForCausalLM": [
        drop("^lm_head."),
    ],
    "Cohere2VisionForConditionalGeneration": [
        rename("^model.vision_tower.", "vision_tower."),
        rename("^model.multi_modal_projector.", "multi_modal_projector."),
        rename("^model.language_model.", "language_model.model."),
    ],
    "CohereASRModel": [
        fused(
            ".second_sub_layer.kv_proj",
            ".second_sub_layer.key_net",
            ".second_sub_layer.value_net",
        ),
        fused(
            ".first_sub_layer.qkv_proj",
            ".first_sub_layer.query_net",
            ".first_sub_layer.key_net",
            ".first_sub_layer.value_net",
        ),
    ],
    "CohereCompassForConditionalGeneration": [
        rename("^model.visual.", "visual."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.language_model.", "language_model.model."),
    ],
    "CohereForCausalLM": [
        drop("_quantizer."),
        drop("^lm_head"),
    ],
    "Cohere_VisionTransformer": [
        fused(".qkv.", ".q.", ".k.", ".v."),
    ],
    "ColModernVBertForRetrieval": [
        rename("^model.text_model.layers.", "text_layers."),
        rename("^model.text_model.embeddings.", "text_embeddings."),
        rename("^model.text_model.final_norm.", "text_final_norm."),
        rename("^model.connector.modality_projection.", "connector.proj."),
        rename("^model.custom_text_proj.", "custom_text_proj."),
        rename("^model.vision_model.vision_model.", "vision_model.vision_model."),
        rename("^model.", ""),
    ],
    # TODO: old loader: tensor_ops
    "ColPaliModel": [
        rename("^embedding_proj_layer.", "custom_text_proj."),
        rename("^vlm.vision_tower.", "vision_tower."),
        rename("^vlm.language_model.", "language_model."),
        rename("^vlm.multi_modal_projector.", "multi_modal_projector."),
        rename("^model.vision_tower.", "vision_tower."),
        rename("^model.language_model.", "language_model."),
        rename("^model.multi_modal_projector.", "multi_modal_projector."),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    # TODO: old loader: tensor_ops
    "ColQwen3Model": [
        rename("^vlm.model.visual.", "visual."),
        rename("^vlm.lm_head.", "language_model.lm_head."),
        rename("^vlm.model.language_model.", "language_model.model."),
        rename("^model.visual.", "visual."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.language_model.model.", "language_model.model."),
        rename("^model.language_model.", "language_model.model."),
    ],
    # TODO: old loader: tensor_ops
    "ColQwen3_5Model": [
        rename("^language_model.", "language_model.model."),
        drop("^mtp."),
    ],
    "ColUltraVBertForRetrieval": [
        rename("^vision_model.(?!vision_model.)", "vision_model.vision_model."),
    ],
    "Cosmos3EdgeForConditionalGeneration": _cosmos3_edge(),
    "Cosmos3ForConditionalGeneration": [
        rename("^(layers|embed_tokens|norm).", r"language_model.model.\1."),
        rename(
            "^(blocks.|merger.|patch_embed.|pos_embed.|deepstack_merger_list.)",
            r"visual.\1",
        ),
        drop("^audio_modality_embed(?:..*)?$"),
        drop("^action_modality_embed(?:..*)?$"),
        drop("_moe_gen"),
        drop(".add_q_proj."),
        drop(".add_k_proj."),
        drop(".add_v_proj."),
        drop(".to_add_out."),
        drop(".norm_added_q."),
        drop(".norm_added_k."),
        rename(".to_q.", ".q_proj."),
        rename(".to_k.", ".k_proj."),
        rename(".to_v.", ".v_proj."),
        rename(".to_out.", ".o_proj."),
        rename(".norm_q.", ".q_norm."),
        rename(".norm_k.", ".k_norm."),
        drop(".input_quantizer."),
        drop(".weight_quantizer."),
        drop(".output_quantizer."),
        drop("^proj_in."),
        drop("^proj_out."),
        drop("^time_embedder."),
        drop("^audio_proj_in."),
        drop("^audio_proj_out."),
        drop("^action_proj_in."),
        drop("^action_proj_out."),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    # TODO: old loader: tensor_ops
    "DSparkDeepseekV4ForCausalLM": [
        fused("attn.fused_wqa_wkv", "attn.wq_a", "attn.wkv"),
        fused(".gate_up_proj", ".w1", ".w3"),
    ],
    "DSpineQwen3Model": [
        rename(r"(?<![A-Za-z0-9_])midlayer(?=.|\*|$)", "layers.0"),
        rename(r"(?<![A-Za-z0-9_])encoder.output_norm_enc(?=.|\*|$)", "hidden_norm"),
        rename(r"(?<![A-Za-z0-9_])encoder.fc(?=.|\*|$)", "fc"),
        rename("transfer_embeddings", "transfer_codebook"),
    ],
    # TODO: old loader: stacked_unparsed
    "DeepSeekMTP": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    "DeepseekOCR2ForCausalLM": [
        rename("^model.embed_tokens.", "language_model.model.embed_tokens."),
        rename("^model.layers.", "language_model.model.layers."),
        rename("^model.norm.", "language_model.model.norm."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", ""),
    ],
    "DeepseekV2Eagle3Model": [
        rename("midlayer.", "layers.0."),
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    # TODO: old loader: stacked_unparsed
    "DeepseekV2Model": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    # TODO: old loader: stacked_unparsed
    "DeepseekV32MTP": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    # TODO: old loader: stacked_unparsed
    "DeepseekV32Model": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    "DeepseekVLV2ForCausalLM": [
        rename("^language.", "language_model."),
    ],
    "DiffusionGemmaForConditionalGeneration": [
        rename("^model.decoder.self_conditioning.", "self_conditioning."),
        rename("^model.decoder.", "model."),
        rename("^model.encoder.language_model.", "model."),
        rename("^model.encoder.vision_tower.", "vision_tower."),
        rename("^model.encoder.embed_vision.", "embed_vision."),
    ],
    "DotsOCRForCausalLM": [
        rename(".attn.qkv_proj.", ".attn.qkv."),
        rename(".attn.out_proj.", ".attn.proj."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
        fused(".fc13.", ".fc1.", ".fc3."),
    ],
    "EagleMistralLarge3ForCausalLM": [
        *MISTRAL_FORMAT,
    ],
    "Ernie4_5_MoeModel": [
        drop("mtp"),
    ],
    "Ernie4_5_VLMoeForConditionalGeneration": [
        rename("spatial_linear.0.", "spatial_linear1."),
        rename("spatial_linear.2.", "spatial_linear2."),
        rename("spatial_linear.3.", "spatial_norm."),
        rename("temporal_linear.0.", "temporal_linear1."),
        rename("temporal_linear.2.", "temporal_linear2."),
        rename("temporal_linear.3.", "temporal_norm."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
        rename("^language_model.model.resampler_model.", "resampler_model."),
    ],
    "ExaoneForCausalLM": [
        fused(".gate_up_proj", ".c_fc_0", ".c_fc_1"),
    ],
    "ExaoneMoeForCausalLM": [
        drop("^mtp."),
    ],
    "FalconH1ForCausalLM": [
        rename("A_log", "A"),
        rename("mamba", "mamba.mamba"),
    ],
    "FireRedASR2ForConditionalGeneration": [
        rename("llm.", "model.decoder."),
        rename("encoder.", "model.encoder.audio_encoder."),
        rename("encoder_projector.", "model.encoder_projector."),
        rename("net.0", "pre_layer_norm"),
        rename("net.1", "linear_expand"),
        rename("net.4", "linear_project"),
        drop("^model.encoder.audio_encoder.positional_encoding.pe"),
    ],
    # TODO: old loader: stacked_unparsed
    "FlashModel": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    "FunASREncoder": [
        fused(
            ".self_attn.qkv",
            ".self_attn.q_proj",
            ".self_attn.k_proj",
            ".self_attn.v_proj",
        ),
    ],
    "FunASRForConditionalGeneration": [
        rename("linear_q.", "q_proj."),
        rename("linear_k.", "k_proj."),
        rename("linear_v.", "v_proj."),
        rename("linear_out.", "out_proj."),
        rename("audio_adaptor.", "model.encoder.audio_adaptor."),
        rename("audio_encoder.", "model.encoder.audio_encoder."),
        rename("llm.model.", "model.decoder."),
        rename("llm.lm_head", "lm_head"),
    ],
    "FunAudioChatForConditionalGeneration": [
        drop("^audio_invert_tower."),
    ],
    "GLM4VForCausalLM": [
        rename(".word_embeddings", ""),
        fused(
            "linear_proj.merged_proj",
            "linear_proj.gate_proj",
            "linear_proj.dense_h_to_4h",
        ),
    ],
    "GPTJForCausalLM": [
        rename("(?<!.attn).attn.([qkv]_(?:scale|zero_point))$", r".attn.attn.\1"),
        drop("attn.bias"),
        drop("attn.masked_bias"),
    ],
    "Gemma3ForConditionalGeneration": [
        rename("^model.language_model.", "language_model.model."),
        rename("^model.vision_tower.", "vision_tower."),
        rename("^model.multi_modal_projector.", "multi_modal_projector."),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    "Gemma3nForCausalLM": [
        drop("embed_audio."),
        drop("embed_vision."),
        drop("audio_tower."),
        drop("vision_tower."),
    ],
    "Gemma3nForConditionalGeneration": [
        rename("^model.embed_audio.", "embed_audio."),
        rename("^model.embed_vision.", "embed_vision."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.vision_tower.", "vision_tower."),
        rename("^model.audio_tower.", "audio_tower."),
        rename("^model.multi_modal_projector.", "multi_modal_projector."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model", "language_model.model"),
    ],
    "Gemma3nTextModel": [
        # KV-shared layers only have `q_proj`: their q keeps its name (the converter's
        # target is not a param there), their K/V tensors load nowhere
        QKV,
        rename("^embed_tokens.", "self_decoder.embed_tokens."),
        rename("^embed_tokens_per_layer.", "self_decoder.embed_tokens_per_layer."),
        rename(
            "^per_layer_model_projection.", "self_decoder.per_layer_model_projection."
        ),
        rename(
            "^per_layer_projection_norm.", "self_decoder.per_layer_projection_norm."
        ),
        rename("^altup_projections.", "self_decoder.altup_projections."),
    ],
    "Gemma4ForCausalLM": [
        drop("audio_tower."),
        drop("vision_tower."),
        drop("embed_audio."),
        drop("embed_vision."),
        rename("^model.language_model.", "model."),
    ],
    "Gemma4ForConditionalGeneration": [
        rename("^model.vision_tower", "vision_tower"),
        rename("^model.embed_vision", "embed_vision"),
        rename("^model.audio_tower.", "audio_tower."),
        rename("^model.embed_audio.", "embed_audio."),
        rename("^model.language_model.", "language_model.model."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model", "language_model.model"),
    ],
    "Gemma4MTP": [
        rename("^pre_projection.", "model.pre_projection."),
        rename("^post_projection.", "model.post_projection."),
    ],
    "Gemma4UnifiedForConditionalGeneration": [
        rename("^model.embed_audio.", "embed_audio."),
        rename("^model.embed_vision.", "embed_vision."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.vision_embedder.", "vision_embedder."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model", "language_model.model"),
    ],
    # TODO: old loader: stacked_unparsed
    "Glm4MoeLiteMTP": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    # TODO: old loader: stacked_unparsed
    "Glm4MoeLiteModel": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    "Glm4vForConditionalGeneration": [
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.visual.", "visual."),
    ],
    "Glm4vVisionTransformer": [
        fused(".attn.qkv.", ".attn.q.", ".attn.k.", ".attn.v."),
    ],
    # TODO: old loader: stacked_unparsed
    "Glm5NextMTP": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    # TODO: old loader: stacked_unparsed
    "Glm5NextModel": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
        fused(
            ".in_proj_qkvbfg_a",
            ".q_proj",
            ".k_proj",
            ".v_proj",
            ".b_proj",
            ".f_a_proj",
            ".g_a_proj",
        ),
        fused(".conv1d", ".q_conv1d", ".k_conv1d", ".v_conv1d"),
    ],
    "Glm5NextVisionTransformer": [
        fused(".attn.qkv.", ".attn.q.", ".attn.k.", ".attn.v."),
    ],
    "GlmAsrEncoder": [
        WHISPER_QKV_BIAS,
    ],
    "GlmAsrForConditionalGeneration": [
        drop("^audio_tower.embed_positions"),
    ],
    "GptOssForCausalLM": [
        rename(".self_attn.", ".attn."),
        rename(".embed_tokens.weight$", ".embedding.weight"),
    ],
    "Granite4VisionForConditionalGeneration": [
        rename("^model.language_model.", "language_model.model."),
        rename("^model.layerwise_projectors.", "layerwise_projectors."),
        rename("^model.spatial_projectors.", "spatial_projectors."),
        rename("^model.image_newline", "image_newline"),
        rename("^model.vision_tower.", "vision_tower."),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    "GraniteMoeModel": [
        rename("moe.input_linear.weight$", "moe.experts.gate_up_proj"),
        rename("moe.output_linear.weight$", "moe.experts.down_proj"),
        rename(".router.layer.weight$", ".gate.weight"),
        rename(".router.weight$", ".gate.weight"),
    ],
    "GraniteSpeechForConditionalGeneration": [
        rename("^language_model.model.model.", "language_model.model."),
    ],
    "HCXVisionV2ForCausalLM": [
        drop("discrete_vision_model"),
        rename("^model.", ""),
        rename("^vision_model.", "visual."),
    ],
    "HYV4ForCausalLM": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    # TODO: old loader: stacked_unparsed
    # TODO: old loader: tensor_ops
    "HYV4MTP": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    "HrmTextForCausalLM": [
        rename(".attn.", ".self_attn."),
    ],
    "InklingForConditionalGeneration": [
        rename(".w13_dn", ".gate_up_proj"),
        rename(".w2_md", ".down_proj"),
        rename("^model.llm.layers.", "model.layers."),
        rename("^model.llm.embed_norm", "model.embed_norm"),
        rename("^model.llm.embed", "model.embed_tokens"),
        rename("^model.llm.norm", "model.norm"),
        rename("^model.llm.unembed", "lm_head"),
        rename("^language_model.layers.", "model.layers."),
        rename("^language_model.lm_head.", "lm_head."),
        rename("^model.audio.", "audio."),
        rename("^model.visual.", "visual.vision_encoder."),
        rename(".w13_weight.scale$", ".routed_experts.w13_weight_scale"),
        rename(".w13_weight.scale2$", ".routed_experts.w13_weight_scale_2"),
        rename(".w2_weight.scale$", ".routed_experts.w2_weight_scale"),
        rename(".w2_weight.scale2$", ".routed_experts.w2_weight_scale_2"),
        rename(
            ".w13_weight.input_global_scale$",
            ".routed_experts.w13_input_global_scale",
        ),
        rename(
            ".w13_weight.weight_global_scale$",
            ".routed_experts.w13_weight_global_scale",
        ),
        rename(".w13_weight.weight_packed$", ".routed_experts.w13_weight_packed"),
        rename(".w13_weight.weight_scale$", ".routed_experts.w13_weight_scale"),
        rename(
            ".w2_weight.input_global_scale$",
            ".routed_experts.w2_input_global_scale",
        ),
        rename(
            ".w2_weight.weight_global_scale$",
            ".routed_experts.w2_weight_global_scale",
        ),
        rename(".w2_weight.weight_packed$", ".routed_experts.w2_weight_packed"),
        rename(".w2_weight.weight_scale$", ".routed_experts.w2_weight_scale"),
        fused(
            ".attn.qkvr.",
            ".attn.wq_du.",
            ".attn.wk_dv.",
            ".attn.wv_dv.",
            ".attn.wr_du.",
        ),
    ],
    "InklingMTP": [
        rename(".w13_dn", ".gate_up_proj"),
        rename(".w2_md", ".down_proj"),
        rename("^model.llm.layers.", "model.layers."),
        # the target's embedding norm, applied before the depth layers
        rename("^model.llm.embed_norm.weight$", "model.backbone_embed_norm.weight"),
        drop("^model.llm.embed.weight$"),  # embedding and head: the target's
        rename("^model.llm.norm", "model.norm"),
        drop("^model.llm.unembed."),
        rename("^language_model.layers.", "model.layers."),
        rename("^language_model.lm_head.", "lm_head."),
        rename("^model.mtp.", "model."),
        rename(".w13_weight.scale$", ".routed_experts.w13_weight_scale"),
        rename(".w13_weight.scale2$", ".routed_experts.w13_weight_scale_2"),
        rename(".w2_weight.scale$", ".routed_experts.w2_weight_scale"),
        rename(".w2_weight.scale2$", ".routed_experts.w2_weight_scale_2"),
        rename(
            ".w13_weight.input_global_scale$",
            ".routed_experts.w13_input_global_scale",
        ),
        rename(
            ".w13_weight.weight_global_scale$",
            ".routed_experts.w13_weight_global_scale",
        ),
        rename(".w13_weight.weight_packed$", ".routed_experts.w13_weight_packed"),
        rename(".w13_weight.weight_scale$", ".routed_experts.w13_weight_scale"),
        rename(
            ".w2_weight.input_global_scale$",
            ".routed_experts.w2_input_global_scale",
        ),
        rename(
            ".w2_weight.weight_global_scale$",
            ".routed_experts.w2_weight_global_scale",
        ),
        rename(".w2_weight.weight_packed$", ".routed_experts.w2_weight_packed"),
        rename(".w2_weight.weight_scale$", ".routed_experts.w2_weight_scale"),
        fused(
            ".attn.qkvr.",
            ".attn.wq_du.",
            ".attn.wk_dv.",
            ".attn.wv_dv.",
            ".attn.wr_du.",
        ),
    ],
    "InternLM2ForCausalLM": [
        fused(".gate_up_proj", ".w1", ".w3"),
    ],
    "InternLM2Model": [
        fused(".gate_up_proj", ".w1", ".w3"),
    ],
    "InternS1ForConditionalGeneration": [
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.vision_tower.", "vision_tower."),
        rename("^model.multi_modal_projector.", "multi_modal_projector."),
    ],
    "InternS2MobiusModel": [
        fused(".in_proj_qkvz", (".in_proj_qkv", 3), ".in_proj_z"),
        fused(".in_proj_ba", ".in_proj_b", ".in_proj_a"),
        fused(".in_proj_ba", ".in_proj_b", ".in_proj_a"),
    ],
    "IsaacForConditionalGeneration": [
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.text_model.lm_head.", "language_model.lm_head."),
        rename("^model.text_model.", "language_model.model."),
        rename("^model.vision_embedding.0", "vision_embedding.transformer"),
        rename("^model.vision_embedding.1", "vision_embedding.linear_fc1"),
        rename("^model.vision_embedding.2", "vision_embedding.act"),
        rename("^model.vision_embedding.3", "vision_embedding.linear_fc2"),
        rename("^model.vision_embedding.", "vision_embedding."),
        rename("^model.lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
    ],
    "JambaForCausalLM": [
        rename(".self_attn.", "."),
        rename(".A_log", ".A"),
    ],
    "JinaEmbeddingsV5DecoderModel": [
        rename("^", "model."),
    ],
    "JinaEmbeddingsV5EncoderModel": [
        rename("^", "model."),
    ],
    "JinaForRanking": [
        drop("^lm_head."),
    ],
    "JinaRobertaModel": [
        rename("emb_ln", "embeddings.LayerNorm"),
        rename("mixer.Wqkv", "attn.qkv_proj"),
        rename("mixer.out_proj", "attn.out_proj"),
        rename("norm1", "attn_ln"),
        rename("mlp.fc1.", "mlp.up_proj."),
        rename("mlp.fc2", "mlp.down_proj"),
        rename("norm2", "mlp_ln"),
    ],
    "KimiK3ForConditionalGeneration": [
        rename("^language_model.layers.", "language_model.model.layers."),
        rename("^mm_projector.proj.0", "mm_projector.linear_1"),
        rename("^mm_projector.proj.2", "mm_projector.linear_2"),
    ],
    "Lfm2ForCausalLM": [
        rename(".conv.", ".short_conv."),
        fused(".w13", ".w1", ".w3"),
    ],
    "Lfm2MoeForCausalLM": [
        rename(".conv.", ".short_conv."),
    ],
    "Lfm2MoeModel": [
        rename(".conv.", ".short_conv."),
        rename("expert_bias", "gate.e_score_correction_bias"),
        fused(".feed_forward.w13.", ".feed_forward.w1.", ".feed_forward.w3."),
    ],
    "Lfm3AudioForConditionalGeneration": [
        rename("^lfm.model.", "language_model.model."),
        rename("^lfm.lm_head.", "language_model.lm_head."),
        rename("^encoder.lfm.lfm.", "audio_encoder.lfm."),
        rename("^encoder.downsample.", "audio_encoder.downsample."),
        rename("^mlp.model.0.", "audio_projector.layer_norm."),
        rename("^mlp.model.1.", "audio_projector.linear_1."),
        rename("^mlp.model.3.", "audio_projector.linear_2."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.audio_encoder.", "audio_encoder."),
        rename("^model.audio_projector.", "audio_projector."),
        rename("^lm_head.", "language_model.lm_head."),
        drop("^codebook_embedding."),
        drop("^depth_linear."),
        drop("^depth_embeddings."),
        drop("^depthformer."),
        drop("^mask_embedding"),
        drop("^model.codebook_embedding."),
        drop("^model.depth_decoder."),
    ],
    "Lfm3VlVisionModel": [
        fused(".qkv_proj.", ".q_proj.", ".k_proj.", ".v_proj."),
        fused(".w13.", ".w1.", ".w3."),
    ],
    "LightOnOCRForConditionalGeneration": [
        rename("^model.vision_encoder.", "vision_tower."),
        rename("^model.vision_projection.", "multi_modal_projector."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.language_model.", "language_model.model."),
    ],
    "LlamaNemotronVLChatModel": [
        drop("norm_mean"),
        drop("norm_std"),
    ],
    "LlamaNemotronVLForEmbedding": [
        rename("^language_model.layers.", "language_model.model.layers."),
        rename("^language_model.embed_tokens.", "language_model.model.embed_tokens."),
        rename("^language_model.norm.", "language_model.model.norm."),
        rename("^vision_model.encoder.", "vision_model.vision_model.encoder."),
        rename("^vision_model.embeddings.", "vision_model.vision_model.embeddings."),
        rename(
            "^vision_model.post_layernorm.",
            "vision_model.vision_model.post_layernorm.",
        ),
    ],
    "LlavaOnevision2ForConditionalGeneration": [
        rename("^model.language_model.", "language_model.model."),
        rename("^model.visual.", "visual."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
    ],
    "LongCatFlashMTP": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    "LongcatFlashNgramForCausalLM": [
        drop("^model.mtp."),
    ],
    "MBartDecoderNoPos": [
        fused(".encoder_attn.kv_proj", ".encoder_attn.k_proj", ".encoder_attn.v_proj"),
    ],
    "Mamba2ForCausalLM": [
        rename(".A_log", ".A"),
    ],
    "MambaForCausalLM": [
        rename(".A_log", ".A"),
    ],
    "MiMoForCausalLM": [
        drop("^model.mtp_layers."),
    ],
    "MingAudioEncoder": [
        rename("blocks.", "layers."),
        rename(".attn.query.", ".self_attn.q_proj."),
        rename(".attn.key.", ".self_attn.k_proj."),
        rename(".attn.value.", ".self_attn.v_proj."),
        rename(".attn.out.", ".self_attn.out_proj."),
        rename(".attn_ln.", ".self_attn_layer_norm."),
        rename(".mlp_ln.", ".final_layer_norm."),
        rename(".mlp.0.", ".mlp.fc1."),
        rename(".mlp.2.", ".mlp.fc2."),
        rename("ln_post.", "layer_norm."),
        rename("positional_embedding$", "embed_positions.weight"),
    ],
    "MiniCPMOBaseModel": [
        drop("^tts"),
    ],
    "MiniCPMV2_6": [
        drop("^apm."),
        drop("^audio"),
        drop("^tts"),
    ],
    "MiniCPMV4_6ForConditionalGeneration": [
        rename("^model.vision_tower.vit_merger.", "vit_merger."),
        rename("^model.vision_tower.", "vpm."),
        rename("^model.vpm.", "vpm."),
        rename("^model.vit_merger.", "vit_merger."),
        rename("^model.merger.", "merger."),
        rename("^model.language_model.", "language_model.model."),
        rename("^lm_head.", "language_model.lm_head."),
        drop("^mtp."),
    ],
    # TODO: old loader: stacked_unparsed
    "MiniMaxM3MTP": [
        fused(
            ".qkv_proj",
            ".q_proj",
            ".k_proj",
            ".v_proj",
            ".index_k_proj",
            ".index_q_proj",
        ),
    ],
    "MiniMaxM3SparseForConditionalGeneration": [
        rename(".mlp.fc1.", ".fc1."),
        rename(".mlp.fc2.", ".fc2."),
        rename("^multi_modal_projector.", "vision_tower.multi_modal_projector."),
        rename("^patch_merge_mlp.", "vision_tower.patch_merge_mlp."),
    ],
    "MiniMaxVLVisionModel": [
        fused(".qkv_proj.", ".q_proj.", ".k_proj.", ".v_proj."),
    ],
    "MistralLarge3ForCausalLM": [
        *MISTRAL_FORMAT,
    ],
    "MixtralForCausalLM": [
        rename(".gate_proj.", ".w1."),
        rename(".up_proj.", ".w3."),
        rename(".down_proj.", ".w2."),
    ],
    "ModernBertForSequenceClassification": [rename("^head.", "pooling.")],
    "JinaVLForSequenceClassification": [
        rename("^score.0.", "score.dense."),
        rename("^score.2.", "score.out_proj."),
        rename("^model.language_model.", "language_model.model."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
    ],
    "RadioMBartPrefixForCausalLM": [drop("^decoder.extra_(heads|proj).")],
    # Jina's reranker checkpoints (flash-attn BERT names) onto RoBERTa's
    "RobertaForSequenceClassification": [
        rename("emb_ln", "embeddings.LayerNorm"),
        rename("layers", "layer"),
        rename("mixer.Wqkv", "attention.self.qkv_proj"),
        rename("mixer.out_proj", "attention.output.dense"),
        rename("norm1", "attention.output.LayerNorm"),
        rename("mlp.fc1", "intermediate.dense"),
        rename("mlp.fc2", "output.dense"),
        rename("norm2", "output.LayerNorm"),
    ],
    "ModernBertForTokenClassification": [
        drop("^drop"),
    ],
    "ModernBertModel": [
        rename("^model.layers.", "encoder_layer.layers."),
        rename("^layers.", "encoder_layer.layers."),
        rename("^model.", ""),
    ],
    "Molmo2ForConditionalGeneration": [
        rename("image_pooling_2d.wq", "image_pooling_2d.q_proj"),
        rename("image_pooling_2d.wk", "image_pooling_2d.k_proj"),
        rename("image_pooling_2d.wv", "image_pooling_2d.v_proj"),
        rename("image_pooling_2d.wo", "image_pooling_2d.o_proj"),
        rename("image_projector.w1", "image_projector.gate_proj"),
        rename("image_projector.w3", "image_projector.up_proj"),
        rename("image_projector.w2", "image_projector.down_proj"),
        rename("att_proj", "qkv_proj"),
        rename("attn_out", "o_proj"),
        rename("q_norm", "q_norm"),
        rename("k_norm", "k_norm"),
        rename("ff_proj", "up_gate_proj"),
        rename("ff_out", "down_proj"),
        rename("attn_norm", "input_layernorm"),
        rename("ff_norm", "post_attention_layernorm"),
        rename("^model.vision_backbone.", "vision_backbone."),
        rename("^model.transformer.blocks.", "model.layers."),
        rename("^model.transformer.ln_f.", "model.norm."),
    ],
    "Molmo2VisionBackbone": [
        fused(".merged_linear", ".gate_proj", ".up_proj"),
        fused(".merged_kv", ".k_proj", ".v_proj"),
        fused(".merged_qkv", ".wq", ".wk", ".wv"),
    ],
    "MolmoForCausalLM": [
        rename("image_projector.w1.", "image_projector.gate_proj."),
        rename("image_projector.w3.", "image_projector.up_proj."),
        rename("image_projector.w2.", "image_projector.down_proj."),
        rename("att_proj", "self_attn.qkv_proj"),
        rename("attn_out", "self_attn.o_proj"),
        rename("q_norm", "self_attn.q_norm"),
        rename("k_norm", "self_attn.k_norm"),
        rename("ff_proj", "mlp.gate_up_proj"),
        rename("ff_out", "mlp.down_proj"),
        rename("attn_norm", "input_layernorm"),
        rename("ff_norm", "post_attention_layernorm"),
        rename("^model.vision_backbone.", "vision_backbone."),
        rename("^model.transformer.blocks.", "model.layers."),
        rename("^model.transformer.ln_f.", "model.norm."),
        rename("^model.transformer.mlp.down_proj.", "lm_head."),
    ],
    "MolmoVisionBackbone": [
        fused(".merged_linear", ".gate_proj", ".up_proj"),
    ],
    "MossAudioModel": [
        rename("^lm_head.", "language_model.lm_head."),
        rename("^language_model.embed_tokens.", "language_model.model.embed_tokens."),
        rename("^language_model.layers.", "language_model.model.layers."),
        rename("^language_model.norm.", "language_model.model.norm."),
        drop("^audio_encoder.embed_positions"),
    ],
    "MossTranscribeDiarizeForConditionalGeneration": [
        rename("^language_model.layers.", "language_model.model.layers."),
        rename("^language_model.embed_tokens.", "language_model.model.embed_tokens."),
        rename("^language_model.norm.", "language_model.model.norm."),
        rename("^model.language_model.model.", "language_model.model."),
        rename("^model.language_model.lm_head.", "language_model.lm_head."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.whisper_encoder.", "whisper_encoder."),
        rename("^model.vq_adaptor.", "vq_adaptor."),
        drop("^model."),
    ],
    "MossTranscribeDiarizeWhisperEncoder": [
        rename(".fc1.", ".mlp.fc1."),
        rename(".fc2.", ".mlp.fc2."),
    ],
    "MuseGlimmerForCausalLM": [
        rename(
            "model.vision_tower.patch_embedder.position_embedding_table.weight",
            "model.vision_tower.positional_embedding_vlm",
        ),
        rename("model.vision_tower.layers.", "model.vision_tower.transformer."),
        rename(".norm1.", ".ln_1."),
        rename(".norm2.", ".ln_2."),
        rename(".attn.proj.", ".attn.o_proj."),
        rename(".mlp.fc1.", ".mlp.c_fc."),
        rename(".mlp.fc2.", ".mlp.c_proj."),
        rename(".self_attn.gate_proj", ".self_attn.output_gate_proj"),
        drop("^model.rotary_emb."),
        rename("^model.language_model.", "model."),
        rename("^language_model.", "model."),
        rename(
            "^model.vision_tower.patch_embedder.patch_embedding.",
            "model.vision_tower.conv1_linear.",
        ),
        rename("^model.vision_tower.", "vision_encoder."),
        rename("^vision_tower.", "vision_encoder."),
        rename("^model.vision_encoder.", "vision_encoder."),
        rename("^model.vision_adapter.fc1.", "model.vision_adapter.c_fc."),
        rename("^model.vision_adapter.fc2.", "model.vision_adapter.c_proj."),
        rename("^model.vision_adapter.", "vision_adapter."),
        rename("^model.vision_projection.", "vision_projection."),
        rename("^model.perception_emb_norm.", "perception_emb_norm."),
    ],
    "MuseSparkDSparkForCausalLM": [
        fused(
            ".feed_forward.gate_up_proj.",
            ".feed_forward.gate_proj.",
            ".feed_forward.up_proj.",
        ),
        fused(
            ".self_attn.qkv_proj.",
            ".self_attn.q_proj.",
            ".self_attn.k_proj.",
            ".self_attn.v_proj.",
            ".self_attn.o_gate_proj.",
        ),
    ],
    "Nemotron3_5TranscribeForConditionalGeneration": [
        rename(".mixer._fp32_params.", ".mixer."),
        rename("^llm.model.norm.", "language_model.model.norm_f."),
        rename("^llm.model.", "language_model.model."),
        rename("^llm.lm_head.", "language_model.lm_head."),
        rename("^backbone.", "language_model.model."),
        rename("^lm_head", "language_model.lm_head"),
        rename("^perception.", "audio_tower."),
    ],
    "NemotronHForCausalLM": [
        rename("A_log", "A"),
        rename("embeddings", "embed_tokens"),
        rename("^backbone", "model"),
        drop("^mtp"),
    ],
    # TODO: old loader: stacked_unparsed
    "NemotronHMTP": [
        rename("^language_model.", ""),
    ],
    # TODO: old loader: tensor_ops
    "NemotronH_Nano_VL_V2": [
        rename("^language_model.backbone", "language_model.model"),
    ],
    "NemotronLabsDiffusionForBlockDiffusion": [
        rename("^encoder.", "model."),
    ],
    "NomicBertModel": [
        rename("emb_ln", "embeddings.LayerNorm"),
        rename("attn.Wqkv", "attn.qkv_proj"),
        rename("norm1", "attn_ln"),
        rename("mlp.fc1.", "mlp.up_proj."),
        rename("mlp.fc11", "mlp.up_proj"),
        rename("mlp.fc12", "mlp.gate_proj"),
        rename("mlp.fc2", "mlp.down_proj"),
        rename("norm2", "mlp_ln"),
        rename("experts.mlp.", ""),
        rename("experts.", ""),
        rename("router.layer", "router"),
        # experts stacked on the rows; w2 stored `[e * i, h]`, the param `[e, h, i]`
        WeightConverter(
            ".mlp.w2$", ".mlp.w2", operations=[ExpertRows(), Transpose(1, 2)]
        ),
    ],
    "OPTForCausalLM": [
        rename("^decoder.", "model.decoder."),
    ],
    "OlmoHybridModel": [
        fused(
            ".linear_attn.in_proj_qkvg",
            ".linear_attn.q_proj",
            ".linear_attn.k_proj",
            ".linear_attn.v_proj",
            ".linear_attn.g_proj",
        ),
        fused(".conv1d", ".q_conv1d", ".k_conv1d", ".v_conv1d"),
    ],
    "OpenCUAForConditionalGeneration": [
        rename("^model.language_model.", "language_model.model."),
        rename("^model.visual.", "visual."),
        rename("^vision_tower.", "visual."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
    ],
    # TODO: old loader: stacked_unparsed
    "OpenPanguMTP": [
        fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa"),
    ],
    "PaddleOCRVLForConditionalGeneration": [
        rename("^model.", "language_model.model."),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    "Param2MoEModel": [
        rename("word_embeddings.", "embed_tokens."),
        rename(".attention.query_key_value.", ".self_attn.qkv_proj."),
        rename(".attention.dense.", ".self_attn.o_proj."),
        rename(".attention.query_layernorm.", ".self_attn.q_layernorm."),
        rename(".attention.key_layernorm.", ".self_attn.k_layernorm."),
        rename(".attention.", ".self_attn."),
        rename(".mlp.gate.expert_bias", ".mlp.gate.e_score_correction_bias"),
    ],
    "ParseAuxMTPModel": [
        rename("^decoder.embed_tokens.", "model.embed_tokens."),
        rename("^decoder.extra_heads.0.", "model.extra_head."),
        rename("^decoder.extra_proj.0.", "model.extra_proj."),
    ],
    "Phi3VForCausalLM": [
        rename("^model.vision_embed_tokens.wte", "embed_tokens"),
        rename("^model.vision_embed_tokens.", "vision_embed_tokens."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
    ],
    "Phi4ForCausalLMV": [
        drop("^model.vision_tower.vision_tower.vision_model.head."),
        rename("^model.vision_tower.vision_tower.", "vision_tower."),
        rename("^model.mm_projector.0.", "multi_modal_projector.linear_1."),
        rename("^model.mm_projector.2.", "multi_modal_projector.linear_2."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
    ],
    "Phi4MMForCausalLM": [
        rename("base_layer.", ""),
        drop("lora"),
        rename(
            "^model.embed_tokens_extend.audio_embed.audio_projection.vision.",
            "embed_tokens_extend.audio_projection_for_vision.",
        ),
        rename(
            "^model.embed_tokens_extend.audio_embed.audio_projection.speech.",
            "embed_tokens_extend.audio_projection.",
        ),
        rename("^model.embed_tokens_extend.audio_embed.", "embed_tokens_extend."),
        rename("^model.embed_tokens_extend.image_embed.", "vision_encoder."),
    ],
    # TODO: old loader: tensor_ops
    "PixtralForConditionalGeneration": [
        rename(
            r"^vision_encoder.transformer.layers.(\d+).attention.wq(?=.|$)",
            r"vision_encoder.transformer.layers.\1.attention.q_proj",
        ),
        rename(
            r"^vision_encoder.transformer.layers.(\d+).attention.wk(?=.|$)",
            r"vision_encoder.transformer.layers.\1.attention.k_proj",
        ),
        rename(
            r"^vision_encoder.transformer.layers.(\d+).attention.wv(?=.|$)",
            r"vision_encoder.transformer.layers.\1.attention.v_proj",
        ),
        rename(
            r"^vision_encoder.transformer.layers.(\d+).attention.wo(?=.|$)",
            r"vision_encoder.transformer.layers.\1.attention.o_proj",
        ),
        rename(
            r"^vision_encoder.transformer.layers.(\d+).feed_forward.w1(?=.|$)",
            r"vision_encoder.transformer.layers.\1.feed_forward.gate_proj",
        ),
        rename(
            r"^vision_encoder.transformer.layers.(\d+).feed_forward.w2(?=.|$)",
            r"vision_encoder.transformer.layers.\1.feed_forward.down_proj",
        ),
        rename(
            r"^vision_encoder.transformer.layers.(\d+).feed_forward.w3(?=.|$)",
            r"vision_encoder.transformer.layers.\1.feed_forward.up_proj",
        ),
        rename(".linear_1.", ".w_in."),
        rename(".linear_2.", ".w_out."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.vision_tower.", "vision_encoder."),
        rename("^model.multi_modal_projector.", "vision_language_adapter."),
        fused(".qkv_proj", ".wq", ".wk", ".wv"),
        fused(".gate_up_proj", ".w1", ".w3"),
    ],
    "Qwen2VisionTransformer": [
        fused(".qkv", ".q_proj", ".k_proj", ".v_proj"),
    ],
    "Qwen2_5OmniThinkerForConditionalGeneration": [
        rename("^thinker.lm_head.", "language_model.lm_head."),
        rename("^thinker.model.", "language_model.model."),
        rename("^thinker.", ""),
        drop("^talker."),
        drop("^token2wav."),
    ],
    "Qwen2_5_VisionTransformer": [
        fused(".mlp.gate_up_proj.", ".mlp.gate_proj.", ".mlp.up_proj."),
        fused(".attn.qkv.", ".attn.q.", ".attn.k.", ".attn.v."),
    ],
    "Qwen3ASRForConditionalGeneration": [
        rename("^thinker.lm_head.", "language_model.lm_head."),
        rename("^thinker.model.", "language_model.model."),
        rename("^thinker.", ""),
        rename("^model.audio_tower.", "audio_tower."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.multi_modal_projector.linear_1.", "audio_tower.proj1."),
        rename("^model.multi_modal_projector.linear_2.", "audio_tower.proj2."),
        drop("^talker."),
        drop("^code2wav."),
    ],
    "Qwen3ASRForcedAlignerForTokenClassification": [
        rename("^thinker.lm_head.", "classifier."),
        rename("^thinker.model.", "language_model.model."),
        rename("^thinker.", ""),
        drop("^talker."),
        drop("^code2wav."),
    ],
    "Qwen3Eagle3Model": [
        rename("midlayer.", "layers.0."),
    ],
    "Qwen3NextForCausalLM": [
        drop("^mtp."),
    ],
    "Qwen3OmniMoeAudioEncoder": [
        fused(
            ".self_attn.qkv_proj.",
            ".self_attn.q_proj.",
            ".self_attn.k_proj.",
            ".self_attn.v_proj.",
        ),
    ],
    "Qwen3Omni_VisionTransformer": [
        fused(".attn.qkv.", ".attn.q.", ".attn.k.", ".attn.v."),
    ],
    "Qwen3_5ForCausalLMBase": [
        rename("^model.language_model.", "model."),
        drop("^mtp."),
    ],
    "Qwen3_VisionTransformer": [
        fused("attn.qkv.", "attn.q.", "attn.k.", "attn.v."),
    ],
    "Qwen4ExpForCausalLM": [
        rename("^model.language_model.", "model."),
    ],
    "Qwen4ExpModel": [
        fused(".in_proj_qkvz", (".in_proj_qkv", 3), ".in_proj_z"),
        fused(".in_proj_ba", ".in_proj_b", ".in_proj_a"),
        fused(
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            "hyper_connection.input_mix_weight_down.weight",
            "hyper_connection.block_inject_weight.weight",
        ),
        fused("ple.kv_proj", "ple.key_proj", "ple.value_proj"),
        fused(".in_proj_ba", ".in_proj_b", ".in_proj_a"),
    ],
    "RadioModel": [
        rename(
            "radio_model.model.patch_generator.video_embedder",
            "embeddings.video_patch_projection",
        ),
        rename(
            "radio_model.model.patch_generator.embedder",
            "embeddings.patch_projection",
        ),
        rename(
            "radio_model.model.patch_generator.pos_embed",
            "embeddings.position_embedding",
        ),
        rename(
            "radio_model.model.patch_generator.cls_token.token",
            "embeddings.cls_register_token",
        ),
        rename("radio_model.model.blocks", "encoder.layer"),
        rename(".attn.", ".attention."),
        rename(".ls1", ".layer_scale1.lambda1"),
        rename(".ls2", ".layer_scale2.lambda1"),
        rename("attention.output.dense", "attention.proj"),
        rename("attention.o_proj", "attention.proj"),
        drop("input_conditioner"),
        drop("summary_idxs"),
        fused(
            "attention.qkv",
            "attention.attention.query",
            "attention.attention.key",
            "attention.attention.value",
        ),
        fused(
            "attention.qkv", "attention.q_proj", "attention.k_proj", "attention.v_proj"
        ),
    ],
    "RicursiveForCausalLM": [
        fused(".qkg_proj", ".q_proj", ".k_proj", ".g_proj"),
    ],
    "SiglipEmbeddingModel": [
        drop(".position_ids"),
    ],
    "SiglipVisionModel": [
        drop("head.attention"),
        drop("head.layernorm"),
        drop("head.mlp"),
        drop("head.probe"),
        drop("packing_position_embedding"),
    ],
    # Snowflake's GTE inherits it
    "GteNewModel": [
        rename("new.", ""),
        rename("layer", "layers"),
        rename("attention.qkv_proj", "attn.qkv_proj"),
        rename("attention.o_proj", "attn.out_proj"),
        drop("^classifier"),
        # stored [up, gate], the param is [gate, up]
        WeightConverter(
            ".mlp.up_gate_proj", ".mlp.gate_up_proj", operations=[SwapHalves()]
        ),
    ],
    "Step3p5ForCausalLM": [
        rename(".share_expert.", ".moe.share_expert."),
    ],
    "Step5ForCausalLM": [
        drop("^vit_large_projector.weight$"),
        rename("(?<!.moe).share_expert.", ".moe.share_expert."),
    ],
    "UltravoxModel": [
        rename("^audio_tower.model.encoder.", "audio_tower."),
        drop("^audio_tower.model."),
        drop("^audio_tower.proj_out."),
    ],
    "WhisperForConditionalGeneration": [
        drop("^proj_out."),
    ],
    "Zamba2ForCausalLM": [
        rename("A_log", "A"),
        rename("0.weight", "A.weight"),
        rename("1.weight", "B.weight"),
    ],
    "_TmlForCausalLMBase": [
        rename(".w13_dn", ".gate_up_proj"),
        rename(".w2_md", ".down_proj"),
        rename("^model.llm.layers.", "model.layers."),
        rename("^model.llm.embed_norm", "model.embed_norm"),
        rename("^model.llm.embed", "model.embed_tokens"),
        rename("^model.llm.norm", "model.norm"),
        rename("^model.llm.unembed", "lm_head"),
        rename("^language_model.layers.", "model.layers."),
        rename("^language_model.lm_head.", "lm_head."),
        rename(".w13_weight.scale$", ".routed_experts.w13_weight_scale"),
        rename(".w13_weight.scale2$", ".routed_experts.w13_weight_scale_2"),
        rename(".w2_weight.scale$", ".routed_experts.w2_weight_scale"),
        rename(".w2_weight.scale2$", ".routed_experts.w2_weight_scale_2"),
        rename(
            ".w13_weight.input_global_scale$",
            ".routed_experts.w13_input_global_scale",
        ),
        rename(
            ".w13_weight.weight_global_scale$",
            ".routed_experts.w13_weight_global_scale",
        ),
        rename(".w13_weight.weight_packed$", ".routed_experts.w13_weight_packed"),
        rename(".w13_weight.weight_scale$", ".routed_experts.w13_weight_scale"),
        rename(
            ".w2_weight.input_global_scale$",
            ".routed_experts.w2_input_global_scale",
        ),
        rename(
            ".w2_weight.weight_global_scale$",
            ".routed_experts.w2_weight_global_scale",
        ),
        rename(".w2_weight.weight_packed$", ".routed_experts.w2_weight_packed"),
        rename(".w2_weight.weight_scale$", ".routed_experts.w2_weight_scale"),
        fused(
            ".attn.qkvr.",
            ".attn.wq_du.",
            ".attn.wk_dv.",
            ".attn.wv_dv.",
            ".attn.wr_du.",
        ),
    ],
}

_CONVERSION_MAPPING["ColBERTModernBertModel"] = _CONVERSION_MAPPING["ColBERTLfm2Model"]
# TODO: old loader: tensor_ops
_CONVERSION_MAPPING["BailingMoeV3ForCausalLM"] = _CONVERSION_MAPPING[
    "BailingMoeV3MTPModel"
] + [
    fused(".qkvb_proj", ".q_proj", ".k_proj", ".v_proj", ".b_proj"),
]
_CONVERSION_MAPPING["CohereAsrForConditionalGeneration"] = (
    [
        rename("^transf_decoder._decoder", "decoder"),
        rename("^transf_decoder._embedding", "decoder.embedding"),
        rename("second_sub_layer.query_net", "second_sub_layer.q_proj"),
        rename("^log_softmax.mlp.layer0.", "proj_out."),
        rename("^(?!proj_out.)", "model."),
    ]
    + _CONVERSION_MAPPING["MossTranscribeDiarizeWhisperEncoder"]
    + [
        drop("model.conv.batch_norm.num_batches_tracked"),
        drop("^model.preprocessor.featurizer.fb"),
        drop("^model.preprocessor.featurizer.window"),
    ]
)
_CONVERSION_MAPPING["D3ForSequenceClassification"] = _CONVERSION_MAPPING[
    "ColQwen3_5Model"
]
# TODO: old loader: tensor_ops
_CONVERSION_MAPPING["DeepSeekV4MTP"] = _CONVERSION_MAPPING[
    "DSparkDeepseekV4ForCausalLM"
]
_CONVERSION_MAPPING["DeepseekOCRForCausalLM"] = _CONVERSION_MAPPING[
    "DeepseekOCR2ForCausalLM"
]
# TODO: old loader: tensor_ops
_CONVERSION_MAPPING["DeepseekV4Model"] = [
    fused("compressor.fused_wkv_wgate", "compressor.wkv", "compressor.wgate"),
] + _CONVERSION_MAPPING["DSparkDeepseekV4ForCausalLM"]
_CONVERSION_MAPPING["Exaone4_5_ForConditionalGeneration"] = _CONVERSION_MAPPING[
    "LlavaOnevision2ForConditionalGeneration"
] + [
    drop("^mtp."),
]
_CONVERSION_MAPPING["GLM4VModel"] = _CONVERSION_MAPPING["GLM4VForCausalLM"]
_CONVERSION_MAPPING["GraniteMoeSharedModel"] = _CONVERSION_MAPPING["GraniteMoeModel"]
_CONVERSION_MAPPING["InternS1ProForConditionalGeneration"] = _CONVERSION_MAPPING[
    "CohereCompassForConditionalGeneration"
]
_CONVERSION_MAPPING["KimiAudioForConditionalGeneration"] = _CONVERSION_MAPPING[
    "MossTranscribeDiarizeWhisperEncoder"
] + [
    rename("^model.encoder.", "audio_tower."),
    rename(
        "^model.vq_adaptor.layers.0.",
        "multi_modal_projector.vq_adaptor_layers_0.",
    ),
    rename(
        "^model.vq_adaptor.layers.3.",
        "multi_modal_projector.vq_adaptor_layers_3.",
    ),
    rename(
        "^model.vq_adaptor.layers.4.",
        "multi_modal_projector.vq_adaptor_layers_4.",
    ),
    rename("^model.layers.", "language_model.model.layers."),
    rename("^model.embed_tokens.", "language_model.model.embed_tokens."),
    rename("^model.norm.", "language_model.model.norm."),
    rename("^lm_head.", "language_model.lm_head."),
    drop("^model."),
    drop("^mimo_layers."),
    drop("^mimo_output."),
    drop("^mimo_norm."),
]
_CONVERSION_MAPPING["KimiK25ForConditionalGeneration"] = _CONVERSION_MAPPING[
    "KimiK3ForConditionalGeneration"
] + [
    rename(".self_attn.attn.q_scale$", ".self_attn.mla_attn.mla_attn.q_scale"),
    rename(".self_attn.attn.k_scale$", ".self_attn.mla_attn.mla_attn.k_scale"),
    rename(".self_attn.attn.v_scale$", ".self_attn.mla_attn.mla_attn.v_scale"),
]
_CONVERSION_MAPPING["Lfm2Model"] = _CONVERSION_MAPPING["Lfm2ForCausalLM"]
_CONVERSION_MAPPING["Lfm2VLForConditionalGeneration"] = _CONVERSION_MAPPING[
    "InternS1ForConditionalGeneration"
]
_CONVERSION_MAPPING["Lfm3ForCausalLM"] = _CONVERSION_MAPPING["Lfm2ForCausalLM"]
_CONVERSION_MAPPING["Lfm3Model"] = _CONVERSION_MAPPING["Lfm2ForCausalLM"]
_CONVERSION_MAPPING["Lfm3OmniForConditionalGeneration"] = _CONVERSION_MAPPING[
    "Gemma3ForConditionalGeneration"
] + [
    rename("^lfm.model.", "language_model.model."),
    rename("^lfm.lm_head.", "language_model.lm_head."),
    rename("^encoder.lfm.lfm.", "audio_encoder.lfm."),
    rename("^encoder.downsample.", "audio_encoder.downsample."),
    rename("^mlp.model.0.", "audio_projector.layer_norm."),
    rename("^mlp.model.1.", "audio_projector.linear_1."),
    rename("^mlp.model.3.", "audio_projector.linear_2."),
    rename("^model.audio_encoder.", "audio_encoder."),
    rename("^model.audio_projector.", "audio_projector."),
    drop("^codebook_embedding."),
    drop("^depth_linear."),
    drop("^depth_embeddings."),
    drop("^depthformer."),
    drop("^mask_embedding"),
    drop("^model.codebook_embedding."),
    drop("^model.depth_decoder."),
]
_CONVERSION_MAPPING["Lfm3VlForConditionalGeneration"] = _CONVERSION_MAPPING[
    "Gemma3ForConditionalGeneration"
]
_CONVERSION_MAPPING["LlamaNemotronVLForSequenceClassification"] = [
    rename("^model.", ""),
] + _CONVERSION_MAPPING["LlamaNemotronVLForEmbedding"]
_CONVERSION_MAPPING["LlavaForConditionalGeneration"] = _CONVERSION_MAPPING[
    "Gemma3ForConditionalGeneration"
]
_CONVERSION_MAPPING["LlavaNextForConditionalGeneration"] = _CONVERSION_MAPPING[
    "BeeForConditionalGeneration"
]
_CONVERSION_MAPPING["LlavaNextVideoForConditionalGeneration"] = _CONVERSION_MAPPING[
    "BeeForConditionalGeneration"
]
_CONVERSION_MAPPING["LlavaOnevisionForConditionalGeneration"] = _CONVERSION_MAPPING[
    "BeeForConditionalGeneration"
]
_CONVERSION_MAPPING["MiMoV2OmniForCausalLM"] = (
    [
        rename("^speech_embeddings.", "audio_encoder.speech_embeddings."),
    ]
    + _CONVERSION_MAPPING["LlavaOnevision2ForConditionalGeneration"]
    + [
        drop("^audio_tokenizer."),
    ]
)
_CONVERSION_MAPPING["MingV3ASRForConditionalGeneration"] = [
    rename("^model.audio.", "audio_encoder."),
    rename("^linear_proj_audio.", "audio_projector.proj."),
    drop("^model.visual."),
    drop("^linear_proj."),
] + _CONVERSION_MAPPING["PaddleOCRVLForConditionalGeneration"]
_CONVERSION_MAPPING["MiniCPMV4_0"] = _CONVERSION_MAPPING["MiniCPMV2_6"]
_CONVERSION_MAPPING["MiniCPMV4_5"] = _CONVERSION_MAPPING["MiniCPMV2_6"]
_CONVERSION_MAPPING["Mistral3ForConditionalGeneration"] = _CONVERSION_MAPPING[
    "Gemma3ForConditionalGeneration"
] + [
    rename("^model.", "language_model.model."),
    rename(".activation_scale$", ".input_scale"),
    rename(".weight_scale_inv$", ".weight_scale"),
]
_CONVERSION_MAPPING["MixtralModel"] = _CONVERSION_MAPPING["MixtralForCausalLM"]
_CONVERSION_MAPPING["OpenAIPrivacyFilterForTokenClassification"] = _CONVERSION_MAPPING[
    "GptOssForCausalLM"
] + [
    rename(".gate_up_proj_blocks$", ".routed_experts.w13_weight"),
    rename(".down_proj_blocks$", ".routed_experts.w2_weight"),
    rename(".gate_up_proj_scales$", ".routed_experts.w13_weight_scale"),
    rename(".down_proj_scales$", ".routed_experts.w2_weight_scale"),
    rename(".gate_up_proj$", ".routed_experts.w13_weight"),
    rename(".down_proj$", ".routed_experts.w2_weight"),
    rename(".gate_up_proj_bias$", ".routed_experts.w13_bias"),
    rename(".down_proj_bias$", ".routed_experts.w2_bias"),
    rename(".gate_up_proj.weight$", ".routed_experts.w13_weight"),
    rename(".gate_up_proj.weight_scale$", ".routed_experts.w13_weight_scale"),
    rename(".gate_up_proj.bias$", ".routed_experts.w13_bias"),
    rename(".gate_up_proj.input_scale$", ".routed_experts.w13_input_scale"),
    rename(".down_proj.weight$", ".routed_experts.w2_weight"),
    rename(".down_proj.weight_scale$", ".routed_experts.w2_weight_scale"),
    rename(".down_proj.bias$", ".routed_experts.w2_bias"),
    rename(".down_proj.input_scale$", ".routed_experts.w2_input_scale"),
]
_CONVERSION_MAPPING["PaliGemmaForConditionalGeneration"] = _CONVERSION_MAPPING[
    "Gemma3ForConditionalGeneration"
]
_CONVERSION_MAPPING["Qwen2VLForConditionalGeneration"] = _CONVERSION_MAPPING[
    "LlavaOnevision2ForConditionalGeneration"
]
_CONVERSION_MAPPING["Qwen2_5_VLForConditionalGeneration"] = _CONVERSION_MAPPING[
    "LlavaOnevision2ForConditionalGeneration"
]
_CONVERSION_MAPPING["Qwen3OmniMoeThinkerForConditionalGeneration"] = [
    rename("^thinker.lm_head.", "language_model.lm_head."),
    rename("^thinker.model.", "language_model.model."),
    rename("^thinker.", ""),
    drop("^talker."),
    drop("^code2wav."),
] + _CONVERSION_MAPPING["PaddleOCRVLForConditionalGeneration"]
_CONVERSION_MAPPING["Qwen3VLForConditionalGeneration"] = _CONVERSION_MAPPING[
    "CohereCompassForConditionalGeneration"
]
_CONVERSION_MAPPING["Qwen3_5ForConditionalGeneration"] = _CONVERSION_MAPPING[
    "CohereCompassForConditionalGeneration"
] + [
    drop("^mtp."),
]
_CONVERSION_MAPPING["Qwen3_5Model"] = _CONVERSION_MAPPING["InternS2MobiusModel"]
_CONVERSION_MAPPING["Qwen3_5MultiTokenPredictor"] = _CONVERSION_MAPPING[
    "InternS2MobiusModel"
]
_CONVERSION_MAPPING["RForConditionalGeneration"] = _CONVERSION_MAPPING[
    "BeeForConditionalGeneration"
]
_CONVERSION_MAPPING["Step3VLForConditionalGeneration"] = _CONVERSION_MAPPING[
    "PaddleOCRVLForConditionalGeneration"
]
_CONVERSION_MAPPING["Step3p7ForConditionalGeneration"] = (
    [
        rename(".attn.in_proj_weight", ".attn.qkv_proj.weight"),
        rename(".attn.in_proj_bias", ".attn.qkv_proj.bias"),
        rename(".mlp.c_fc", ".mlp.fc1"),
        rename(".mlp.c_proj", ".mlp.fc2"),
        rename("^model.vision_model.", "vision_model."),
        rename("^model.vit_large_projector.", "vit_large_projector."),
        rename("^model.vit_large_projector", "vit_large_projector"),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.language_model", "language_model.model"),
    ]
    + _CONVERSION_MAPPING["PaddleOCRVLForConditionalGeneration"]
    + [
        rename("^lm_head", "language_model.lm_head"),
    ]
)
_CONVERSION_MAPPING["StepVLForConditionalGeneration"] = [
    rename(".attn.in_proj_weight", ".attn.qkv_proj.weight"),
    rename(".attn.in_proj_bias", ".attn.qkv_proj.bias"),
    rename(".mlp.c_fc", ".mlp.fc1"),
    rename(".mlp.c_proj", ".mlp.fc2"),
] + _CONVERSION_MAPPING["PaddleOCRVLForConditionalGeneration"]
_CONVERSION_MAPPING["WhisperEncoder"] = _CONVERSION_MAPPING[
    "MossTranscribeDiarizeWhisperEncoder"
] + [
    WHISPER_QKV_BIAS,
]
_CONVERSION_MAPPING["InternS2PreviewForConditionalGeneration"] = _CONVERSION_MAPPING[
    "Qwen3_5ForConditionalGeneration"
] + [
    drop("^model.time_series."),
    drop("^time_series."),
]
_CONVERSION_MAPPING["WhisperDecoder"] = _CONVERSION_MAPPING["WhisperEncoder"] + [
    WeightConverter(
        [".encoder_attn.k_proj.bias", ".encoder_attn.v_proj.bias"],
        ".encoder_attn.kv_proj.bias",
        operations=[ZeroMissingSource(), OutputConcatenate()],
    ),
    fused(".encoder_attn.kv_proj", ".encoder_attn.k_proj", ".encoder_attn.v_proj"),
]


def _shard_order(shard) -> tuple:
    """Shards in a fused param's order: indices, then q, k, v, then the others"""
    if isinstance(shard, int):
        return (0, shard)
    return (1, {"q": 0, "k": 1, "v": 2}.get(shard, 3), str(shard))


def from_mapper(mapper: "WeightsMapper") -> list[WeightTransform]:
    """A vLLM `WeightsMapper` built by a model's helper (e.g. DeepSeek V4's, per expert
    dtype), as Transformers transforms: its rules in their order (renamings, regex,
    substrings, prefixes, suffixes), its stacked shards as fused converters.
    TODO: rewrite those helpers to return transforms directly."""
    rules: list[WeightTransform] = list(mapper.orig_to_new_renaming)
    for pattern, new in mapper.orig_to_new_regex.items():
        rules.append(
            drop(pattern.pattern) if new is None else rename(pattern.pattern, new)
        )
    for kind, wrap in (
        ("orig_to_new_substr", lambda k: re.escape(k)),
        ("orig_to_new_prefix", lambda k: "^" + re.escape(k)),
        ("orig_to_new_suffix", lambda k: re.escape(k) + "$"),
    ):
        for key, new in getattr(mapper, kind).items():
            rules.append(drop(wrap(key)) if new is None else rename(wrap(key), new))
    shards: dict[str, list[tuple]] = {}
    for source, (target, shard) in mapper.orig_to_new_stacked.items():
        for one in shard if isinstance(shard, list) else [shard]:
            shards.setdefault(target, []).append((one, source))
    for target, members in shards.items():
        members.sort(key=lambda member: _shard_order(member[0]))
        rules.append(fused(target, *(source for _, source in members)))
    return rules


def _layer_experts(model: nn.Module) -> list[WeightTransform]:
    """Per-expert checkpoints of the model's `FusedMoE` layers, by the names each layer
    gives its checkpoint's experts (`ckpt_gate_proj_name`, ...): Transformers' expert
    merge, for the models Transformers does not map"""
    names = {
        (m.ckpt_gate_proj_name, m.ckpt_up_proj_name, m.ckpt_down_proj_name)
        for m in model.modules()
        if hasattr(m, "ckpt_gate_proj_name")
    }
    out: list[WeightTransform] = []
    for gate, up, down in sorted(names, key=str):
        sources = [f"experts.*.{gate}.weight"] + (
            [f"experts.*.{up}.weight"] if up else []
        )
        out.append(
            WeightConverter(
                sources,
                TRANSFORMERS_EXPERTS["experts.gate_up_proj"],
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            )
        )
        out.append(
            WeightConverter(
                f"experts.*.{down}.weight",
                TRANSFORMERS_EXPERTS["experts.down_proj"],
                operations=[MergeModulelist(dim=0)],
            )
        )
    return out


def _entry_name(cls: type) -> str | None:
    """The class's entry, or the one it inherits (as it inherited its mapper). vLLM's
    classes only: a Transformers module inside a vLLM model (Lfm3 audio's `Lfm2Model`)
    keeps its checkpoint names"""
    return next(
        (
            c.__name__
            for c in cls.__mro__
            if c.__name__ in _CONVERSION_MAPPING and c.__module__.startswith("vllm.")
        ),
        None,
    )


def get_conversion_mapping(model: nn.Module) -> list[WeightTransform]:
    """The entries of the model's classes, each scoped to where its instances live
    (e.g. a vision tower's to `visual`), outermost first: its renamings see the
    checkpoint's names, its converters the model's"""
    instances: dict[str, list[str]] = {}
    first: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if (cls := _entry_name(type(module))) is not None:
            instances.setdefault(cls, []).append(name)
            first.setdefault(cls, module)
    model_types = {
        module.config.model_type
        for module in model.modules()
        if getattr(getattr(module, "config", None), "model_type", None)
    }
    out = to_vllm([e for t in sorted(model_types) for e in transformers_experts(t)])
    seen: dict[tuple, list[str]] = {}
    out += _layer_experts(model)
    for cls, names in sorted(instances.items(), key=lambda kv: min(map(len, kv[1]))):
        parts = os.path.commonprefix([name.split(".") for name in names])
        scope = ".".join(p for p in parts if not p.isdigit())
        entry = _CONVERSION_MAPPING[cls]  # a list, or a function of the module (config)
        for transform in entry(first[cls]) if callable(entry) else entry:
            # an inner class repeating its outer class's transform applies it once
            key = (type(transform), *transform._original_source_patterns)
            outer = seen.setdefault(key, [])
            if any(not o or f"{scope}.".startswith(f"{o}.") for o in outer):
                continue
            outer.append(scope)
            transform = deepcopy(transform)
            if scope:
                transform.scope_prefix, transform.base_model_prefix = scope, ""
            out.append(transform)
    # Last, the projections almost every model fuses: a model that does not keeps its
    # names, the converter's target not being one of its params
    return [*out, QKV, GATE_UP]


def rename_mapper(model) -> "WeightsMapper | None":
    """The renamings of a model (or a model class), for what names modules rather
    than loading them: LoRA adapters, quantization configs' layer lists, EPLB. Dropped
    names stay as they are."""
    from vllm.model_executor.models.utils import WeightsMapper

    if isinstance(model, nn.Module):
        transforms = get_conversion_mapping(model)
    else:
        entry = _CONVERSION_MAPPING.get(_entry_name(model), [])
        transforms = [] if callable(entry) else entry
    renamings = [
        t
        for t in transforms
        if not isinstance(t, WeightConverter)
        and "_dropped_" not in t._original_target_patterns
    ]
    return WeightsMapper(orig_to_new_renaming=renamings) if renamings else None
