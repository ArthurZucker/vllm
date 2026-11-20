# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How each model's checkpoint becomes its params: Transformers `WeightTransform`s.

One list per module class. All renamings apply in order, then the first matching
converter. A pattern without a param suffix (`.q_proj`) takes all that module's tensors
(weight, bias, scales). Sources are TP-sharded on read (`ops_loader.VllmShard`).
"""

import os
import re
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
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
    Affine,
    Deinterleave,
    EqualSources,
    ExpertRows,
    ExpertScalars,
    Gather,
    GroupedQKV,
    HeadPermute,
    MergeLora,
    Narrow,
    OutputConcatenate,
    ParamRows,
    Reciprocal,
    Refuse,
    RepeatLastSource,
    RepeatToParam,
    SwapHalves,
    ViewAsParam,
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


def _drops(patterns: str) -> list[WeightRenaming]:
    """`drop` for each space-separated pattern"""
    return [drop(p) for p in patterns.split()]


def _move(old: str, new: str, *names: str) -> list[WeightRenaming]:
    """Each of `names` from under the `old` prefix to under `new`"""
    return [rename(f"^{old}{name}", f"{new}{name}") for name in names]


class Fused(WeightConverter):
    """Concatenates its sources; `shards[i]`: how many param shards source `i` fills"""

    __slots__ = ("shards",)


def fused(target: str, *sources: str | tuple[str, int]) -> WeightConverter:
    """`sources` (module names, or `(name, shards)`) fused into `target`, in order"""
    patterns = [re.escape(s if isinstance(s, str) else s[0]) for s in sources]
    # Transformers anchors a bare name (`q_proj`) at the key's end: a module boundary
    patterns = [rf"(?<![^.]){p}" if re.fullmatch(r"\w+", p) else p for p in patterns]
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
        rf"(?<!\w){re.escape(hf)}{re.escape(suffix)}$",  # not `shared_experts.`
        (vllm.removesuffix("_weight") + "_bias") if suffix == "_bias" else vllm + new,
    )
    for hf, vllm in TRANSFORMERS_EXPERTS.items()
    for suffix, new in EXPERT_SUFFIXES.items()
]


def to_vllm(conversions: list[WeightTransform]) -> list[WeightTransform]:
    """Transformers' expert conversions retargeted to vLLM's fused experts"""
    out: list[WeightTransform] = [*FUSED_EXPERTS]
    for conversion in conversions:
        targets = list(conversion._original_target_patterns)
        for hf, vllm in TRANSFORMERS_EXPERTS.items():
            # whole names only: a shared expert's `shared_experts.gate_up_proj` stays
            targets = [re.sub(rf"(?<!\w){re.escape(hf)}", vllm, t) for t in targets]
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
# Draft models' vocabulary maps (EAGLE-3, DFlash, DSpark): `t2d` is training only
DRAFT_VOCAB = [rename("^d2t$", "draft_id_to_target_id"), drop("^t2d$")]
# A draft checkpoint saved without its body's `model.` prefix (EAGLE, DFlash, DSpark)
DRAFT_MODEL_PREFIX = rename(r"^(?!lm_head\.|draft_id_to_target_id$|model\.)", "model.")
QKV = fused(".qkv_proj", ".q_proj", ".k_proj", ".v_proj")
GATE_UP = fused(".gate_up_proj", ".gate_proj", ".up_proj")
MLA_QKV_A = fused(".fused_qkv_a_proj", ".q_a_proj", ".kv_a_proj_with_mqa")


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


def _beam(module: nn.Module) -> list[WeightTransform]:
    """Beam's attention gate, fused with q, k and v when `merged_qkv_gate`"""
    merged = any(getattr(m, "merged_qkv_gate", False) for m in module.modules())
    qkv = (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")
    return [
        rename(".self_attn.gate_proj", ".self_attn.output_gate_proj"),
        fused(".self_attn.qkv_gate_proj", *qkv, ".self_attn.output_gate_proj")
        if merged
        else fused(".self_attn.qkv_proj", *qkv),
        # not the experts' (`.mlp.experts.N.gate_proj`)
        fused(".mlp.gate_up_proj", ".mlp.gate_proj", ".mlp.up_proj"),
    ]


def _dots3_note(module: nn.Module) -> list[WeightTransform]:
    """The towers the model did not build (no config, no modality) are dropped"""
    return [
        rename("^model.", "language_model.model."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^mtp.", "language_model.mtp."),
        rename("^vision_encoder.", "visual."),
        rename("^audio_encoder.", "audio_tower."),
        *(
            drop(f"^{tower}.")
            for tower in ("visual", "audio_tower")
            if getattr(module, tower) is None
        ),
    ]


def _k3_dspark(module: nn.Module) -> list[WeightTransform]:
    """K3 DSpark's draft, saved without `model.`; the confidence head only if built"""
    confidence = getattr(module.config, "enable_confidence_head", False)
    layers = module.model.config.num_hidden_layers
    return [
        *([] if confidence else [drop("confidence_head")]),
        drop("embed_tokens"),
        drop("lm_head"),
        rename("^", "model."),
        GATE_UP,
        MLA_QKV_A,
        # every draft layer's `kv_a_proj_with_mqa`, duplicated on the stream
        fused(
            "context_kv_proj.", *(f"context_kv_proj_src.{i}." for i in range(layers))
        ),
    ]


# Muse Spark Omni's checkpoint: the text stack's layers, NaViT vision, waveform audio
MUSE_SPARK_OMNI = [
    rename("^model.language_model.", "model."),
    *(
        rename(rf"^(model.layers.\d+).{old}", rf"\1.{new}")
        for old, new in (
            ("input_layernorm.", "self_attn.pre_qkv_proj_norm."),
            ("post_attention_residual_gate.", "add_attn."),
            ("post_feedforward_residual_gate.", "add_ffn."),
            ("pre_feedforward_layernorm.", "pre_ffn_norm."),
            ("post_feedforward_layernorm.", "post_ffn_norm."),
            ("mlp.experts.post_expert_norm.", "feed_forward.post_expert_norm."),
            ("mlp.gate.e_score_correction_bias$", "feed_forward.expert_bias"),
            ("mlp.gate.", "feed_forward.router."),
            ("mlp.pre_expert_proj.", "feed_forward.pooler."),
            ("mlp.post_expert_proj.", "feed_forward.unpooler."),
            ("mlp.", "feed_forward."),
        )
    ),
    *(
        rename(f".experts.{hf}{suffix}$", f".experts.{vllm}{suffix or '_weight'}")
        for hf, vllm in (("gate_up_proj", "w13"), ("down_proj", "w2"))
        for suffix in ("_input_scale", "_weight_scale_2", "_weight_scale", "")
    ),
    rename("^model.vision_adapter.fc1.", "vision_model.adapter.c_fc."),
    rename("^model.vision_adapter.fc2.", "vision_model.adapter.c_proj."),
    rename(
        "^model.vision_tower.patch_embedder.patch_embedding.",
        "vision_model.conv1_linear.",
    ),
    rename(
        "^model.vision_tower.patch_embedder.position_embedding_table.weight$",
        "vision_model.positional_embedding_vlm",
    ),
    rename("^model.vision_tower.layers.", "vision_model.encoder.layers."),
    rename("^model.vision_tower.", "vision_model."),
    rename("^model.vision_projection.", "vision_projection."),
    *(
        rename(rf"^(vision_model.encoder.layers.\d+).{old}", rf"\1.{new}")
        for old, new in (
            ("attn.proj.", "attn.out_proj."),
            ("norm1.", "ln_1."),
            ("norm2.", "ln_2."),
            ("mlp.fc1.", "mlp.c_fc."),
            ("mlp.fc2.", "mlp.c_proj."),
        )
    ),
    *(
        rename(
            rf"^model.audio_tower.layers.(\d+).conv_blocks.{block}.",
            rf"audio_encoder.stages.\1.{block + 1}.",
        )
        for block in range(2)
    ),
    *(
        rename(rf"^(audio_encoder.stages.\d+.\d+).{old}", rf"\1.{new}")
        for old, new in (
            ("conv.", "conv.conv."),
            ("mlp.fc1.", "linear1."),
            ("mlp.fc2.", "linear2."),
        )
    ),
    rename(
        r"^model.audio_tower.layers.(\d+).patch_embedder.norm.",
        r"audio_encoder.stages.\1.0.norm.",
    ),
    rename(
        r"^model.audio_tower.layers.(\d+).patch_embedder.proj.",
        r"audio_encoder.stages.\1.0.linear.",
    ),
    rename("^model.audio_tower.post_proj_norm.", "audio_encoder.dual_norm."),
    rename("^model.audio_tower.proj.", "audio_encoder.o_proj."),
    rename("^model.audio_tower.norm.", "audio_encoder.norm."),
    fused(
        ".self_attn.qkv_proj.",
        ".self_attn.q_proj.",
        ".self_attn.k_proj.",
        ".self_attn.v_proj.",
        ".self_attn.gate_proj.",
    ),
    fused(".attn.qkv_proj.", ".attn.q_proj.", ".attn.k_proj.", ".attn.v_proj."),
]


def _deepseek_v4(
    module: nn.Module, vl: bool = False, image: bool = True
) -> list[WeightTransform]:
    """DeepSeek V4 / V4.1's reference checkpoint, under `language_model.` if `vl`"""
    path = type(module).__module__
    v41, amd, cpu = ".deepseek_v41." in path, ".amd." in path, ".cpu." in path
    # V4 on AMD / CPU: Quark exports (`.input_scale`), shared experts fused into the
    # routed ones (keeping their names), no LoRA-wrapped experts
    quark = (amd or cpu) and not v41
    fp4 = getattr(module.config, "expert_dtype", "fp4") == "fp4"
    expert_scale = "weight_scale" if fp4 else "weight_scale_inv"
    linear_scale = getattr(module, "linear_scale_name", "weight_scale_inv")
    root = "language_model." if vl else ""
    fused_shared = quark and bool(
        module.model.is_fused_shared_expert_enabled
        if cpu
        else module.model.fused_shared_expert_layers
    )
    image_tensors = ("vision.", "aligner.", "image_")
    out: list[WeightTransform] = []
    if not quark:
        lora_experts = r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
        out.append(rename(lora_experts, rf"\1.{expert_scale}"))
    if fp4:
        out.append(rename(r"(\.experts\.\d+\.w[123])\.scale$", r"\1.weight_scale"))
    if v41:
        # the `embed.weight` suffix renames the engram's fp8 table, not its scale;
        # Quark (on AMD) spells it `weight_scale`
        for scale in ("scale", "weight_scale") if amd else ("scale",):
            out.append(
                rename(rf"(engram\.embed)\.{scale}$", r"\1_tokens.weight_scale_inv")
            )
    out.append(rename(r"\.scale$", f".{linear_scale}"))
    if cpu and fp4:
        out.append(rename(r"(?<!\.w[123])\.weight_scale$", ".weight_scale_inv"))
    if not fused_shared:
        out.append(rename(".shared_experts.w2", ".shared_experts.down_proj"))
    # MTP weights are loaded by the draft model
    out.append(drop("mtp."))
    if v41 and not vl:  # V4.1's checkpoint declares the VL architecture
        out += [drop(name) for name in image_tensors]
    # `hc_head_fn`, ...: a pattern ending in a word character only matches whole words
    hc_head = ["hc_head_"] if vl or not v41 else []
    for prefix in ("layers.", "embed.", "norm.", *hc_head, "mtp."):
        out.append(rename("^" + prefix, f"{root}model.{prefix}"))
    if not image:
        out += [drop("^" + name) for name in image_tensors]
    # as Transformers saves it (`save_pretrained`): `model.hc_head.hc_*`, `attn.norm`
    out += [
        rename(r"^model\.hc_head\.hc_(base|fn|scale)$", rf"{root}model.hc_head_\1"),
        rename(r"\.attn\.norm\.", ".attn.kv_norm."),
    ]
    out += [
        rename("head.weight$", f"{root}lm_head.weight"),
        rename("embed.weight$", "embed_tokens.weight"),
        rename(".ffn.gate.bias$", ".ffn.gate.e_score_correction_bias"),
    ]
    if quark:
        out.append(rename(".input_scale$", ".input_scale_2"))
    return out


# DeepSeek V4's fused projections; not the routed experts' `w1` / `w3` (their own)
DEEPSEEK_V4_LAYER = [
    fused("attn.fused_wqa_wkv", "attn.wq_a", "attn.wkv"),
    fused(".shared_experts.gate_up_proj", ".shared_experts.w1", ".shared_experts.w3"),
]


def _deepseek_v4_draft_layers(module: nn.Module) -> list[WeightTransform]:
    """A DeepSeek V4 draft's layers in its target's checkpoint: `.scale` per dtype"""
    fp4 = getattr(module.config, "expert_dtype", "fp4") == "fp4"
    expert_scale = "weight_scale" if fp4 else "weight_scale_inv"
    linear_scale = getattr(module, "linear_scale_name", "weight_scale_inv")
    names = [n for n, _ in getattr(module, "named_parameters", list)()]
    inv_only = any(n.endswith(".weight_scale_inv") for n in names) and not any(
        n.endswith(".weight_scale") and ".experts." not in n for n in names
    )
    return [
        rename(r"(\.experts\.\d+\.w[123])\.scale$", rf"\1.{expert_scale}"),
        rename(r"\.scale$", f".{linear_scale}"),
        *(
            [
                rename(
                    r"^(?!.*\.experts\.\d+\.)(.*)\.weight_scale$",
                    r"\1.weight_scale_inv",
                )
            ]
            if inv_only
            else []
        ),
        rename(".shared_experts.w2.", ".shared_experts.down_proj."),
        rename(".ffn.gate.bias$", ".ffn.gate.e_score_correction_bias"),
        *DEEPSEEK_V4_LAYER,
    ]


def _deepseek_v4_dspark(module: nn.Module) -> list[WeightTransform]:
    """DeepSeek V4 / V4.1 DSpark's draft: the target checkpoint's `mtp.N.` as layer N,
    the head stack at model level, the rest dropped"""
    model = module.model
    heads = ["main_proj", "main_norm", "norm", "markov_head", "confidence_head"]
    if hasattr(model, "hc_head_fn"):  # V4's, at model level (V4.1's per layer)
        heads.append(r"hc_head_\w+")
    out = [
        drop(r"^(?!mtp\.|context_wkv_proj_src\.)"),
        *(
            [drop(r"^mtp\.\d+\.confidence_head\.")]
            if model.confidence_head is None
            else []
        ),
        rename(r"^mtp\.\d+\.markov_head\.embed\.", "model.markov_head.markov_w1."),
        rename(r"^mtp\.\d+\.markov_head\.head\.", "model.markov_head.markov_w2."),
        rename(rf"^mtp\.\d+\.((?:{'|'.join(heads)})(?:\.|$))", r"model.\1"),
        rename(r"^mtp\.(\d+)\.", r"model.layers.\1."),
        *_deepseek_v4_draft_layers(module),
    ]
    if hasattr(model, "context_wkv_proj"):
        layers = len(model.layers)
        sources = (f"context_wkv_proj_src.{i}." for i in range(layers))
        out.append(fused("model.context_wkv_proj.", *sources))
    return out


def _deepseek_v4_mtp(module: nn.Module) -> list[WeightTransform]:
    """DeepSeek V4's MTP draft: `mtp.N.` as layer `num_hidden_layers + N`"""
    config = module.config
    start = config.num_hidden_layers
    layer = r"^(model\.layers\.\d+)\."
    own = r"enorm|hnorm|e_proj|h_proj|shared_head|hc_head_\w+"
    return [
        *(
            rename(rf"^mtp\.{i}\.", f"model.layers.{start + i}.")
            for i in range(config.num_nextn_predict_layers)
        ),
        drop(r"^(?!model\.layers\.)"),
        # shared by the MTP layers: the first one's
        rename(rf"^model\.layers\.{start}\.emb\.tok_emb\.", "model.embed_tokens."),
        drop(r"\.emb\.tok_emb\."),
        rename(layer + r"head\.weight$", r"\1.shared_head.head.weight"),
        rename(layer + r"norm\.weight$", r"\1.shared_head.norm.weight"),
        rename(layer + rf"(?!(?:{own})(?:\.|$))", r"\1.mtp_block."),
        *_deepseek_v4_draft_layers(module),
    ]


# Tml / Inkling layers: the MLPs, the routed experts' quantized tensors and the
# attention's fused projection
TML_LAYERS = [
    rename(".w13_dn", ".gate_up_proj"),
    rename(".w2_md", ".down_proj"),
    *(
        rename(
            f".{w}_weight.{src}$",
            f".routed_experts.{w}_"
            + {"scale": "weight_scale", "scale2": "weight_scale_2"}.get(src, src),
        )
        for w in ("w13", "w2")
        for src in (
            "scale",
            "scale2",
            "input_global_scale",
            "weight_global_scale",
            "weight_packed",
            "weight_scale",
        )
    ),
    fused(
        ".attn.qkvr.", ".attn.wq_du.", ".attn.wk_dv.", ".attn.wv_dv.", ".attn.wr_du."
    ),
]
TML = [
    rename("^model.llm.layers.", "model.layers."),
    rename("^model.llm.embed_norm", "model.embed_norm"),
    rename("^model.llm.embed", "model.embed_tokens"),
    rename("^model.llm.norm", "model.norm"),
    rename("^model.llm.unembed", "lm_head"),
    rename("^language_model.layers.", "model.layers."),
    rename("^language_model.lm_head.", "lm_head."),
    *TML_LAYERS,
]


# MiniMax M3's attention: in the param's shard order
# (`MinimaxM3QKVParallelLinearWithIndexer`)
MINIMAX_M3_QKV = fused(
    ".qkv_proj", ".q_proj", ".k_proj", ".v_proj", ".index_q_proj", ".index_k_proj"
)


def _minimax_m3_mtp(module: nn.Module) -> list[WeightTransform]:
    """MiniMax M3's MTP draft: its layers and shared embeddings, the rest dropped"""
    names = [n for n, _ in module.named_parameters()]
    scale_inv = any(n.endswith("weight_scale_inv") for n in names)
    return [
        rename("^language_model.", ""),
        drop(
            r"^(?!model\.mtp\.layers\.|model\.embed_tokens\.weight$|lm_head\.weight$)"
        ),
        rename("^model.mtp.layers.", "model.layers."),
        *([] if scale_inv else [rename(r"\.weight_scale_inv$", ".weight_scale")]),
        MINIMAX_M3_QKV,
    ]


def _gemma4(module: nn.Module) -> list[WeightTransform]:
    """`attention_k_eq_v` full-attention layers ship `k_proj` only: K loads as K and V"""
    config = module.config
    shared = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
    k_eq_v = getattr(config, "attention_k_eq_v", False)
    return [
        WeightConverter(
            [f"layers.{i}.self_attn.q_proj", f"layers.{i}.self_attn.k_proj"],
            f"layers.{i}.self_attn.qkv_proj",
            operations=[RepeatLastSource(), OutputConcatenate()],
        )
        for i, layer_type in enumerate(config.layer_types[:shared])
        if k_eq_v and layer_type == "full_attention"
    ]


def _mimo_mtp(module: nn.Module) -> list[WeightTransform]:
    """MTP layers stored 0-indexed without the `mtp_block` wrapper"""
    config = module.config
    start, layers = config.num_hidden_layers, config.num_nextn_predict_layers
    blocks = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
    return [
        *(rename(rf"\.{block}\.", f".mtp_block.{block}.") for block in blocks),
        *(
            rename(rf"^model\.mtp_layers\.{i}\.", f"model.mtp_layers.{i + start}.")
            for i in range(layers)
        ),
    ]


def _ernie_mtp(module: nn.Module) -> list[WeightTransform]:
    """MTP weights stored as a flat `mtp_*.0.` block, after the decoder layers"""
    layer = module.config.num_hidden_layers
    parts = ("mtp_emb_norm", "mtp_hidden_norm", "mtp_linear_proj", "mtp_block")
    return [
        rename(rf"model\.{part}\.0\.", f"model.layers.{layer}.{part}.")
        for part in parts
    ]


def _k2_horizon(module: nn.Module) -> list[WeightTransform]:
    """An attention layer's q/k rows and norms folded to NeoX rope order; MoVA: q, k"""
    if not module.fold_rope_weights:
        return []
    from vllm.model_executor.models.k2_horizon import _rope_weight_perm

    head_dim = module.head_dim
    permute = HeadPermute(_rope_weight_perm(head_dim, module.rope_head_dim), head_dim)
    mova = hasattr(module, "qk_proj")
    qk = (
        fused("qk_proj", "q_proj", "k_proj")
        if mova
        else fused("qkv_proj", "q_proj", "k_proj", "v_proj")
    )
    qk.operations = [HeadPermute(permute.index, head_dim, (0, 1)), *qk.operations]
    norms = [
        WeightConverter(f"{norm}.weight", f"{norm}.weight", operations=[permute])
        for norm in ("q_norm", "k_norm")
        if getattr(module, "query_key_norm", False)
    ]
    return [qk, *norms]


def _kimi_kda(module: nn.Module) -> list[WeightTransform]:
    """A KDA layer's projections into `in_proj_qkvgfab`, in its shard order"""
    full_rank = getattr(module, "use_full_rank_gate", False)
    parts = (
        ("q", "k", "v", "g", "f_a", "b") if full_rank else ("q", "k", "v", "b", "f_a")
    )
    return [
        fused("in_proj_qkvgfab", *(f"{p}_proj" for p in parts)),
        fused("conv1d", "q_conv1d", "k_conv1d", "v_conv1d"),
    ]


def _minicpm_moe(module: nn.Module) -> list[WeightTransform]:
    """MiniCPM's experts (`experts.N.w{1,3,2}`) merged into its `ws` / `w2s`"""
    return [
        WeightConverter(
            ["experts.*.w1.weight", "experts.*.w3.weight"],
            "ws",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            "experts.*.w2.weight", "w2s", operations=[MergeModulelist(dim=0)]
        ),
    ]


def _muse_spark_moe(module: nn.Module) -> list[WeightTransform]:
    """The router output multiplier folded into the router weight as it loads"""
    scale = module.router_output_multiplier
    if scale is None:
        return []
    return [
        WeightConverter("router.weight", "router.weight", operations=[Affine(scale)])
    ]


def _leader_written(module: nn.Module) -> list[WeightTransform]:
    """A DP-shared host table (`leader_written`): dropped but on the group's leader"""
    return [drop(rf"^{name}$") for name in getattr(module, "leader_written", ())]


def _engram_table(module: nn.Module) -> list[WeightTransform]:
    """An engram table's rows this rank holds (`row_start`), as their bytes"""
    return [
        *_leader_written(module),
        *(
            WeightConverter(name, name, operations=[ViewAsParam(), ParamRows()])
            for name in ("weight", "weight_scale_inv")
        ),
    ]


def _flint_experts(module: nn.Module) -> list[WeightTransform]:
    """Flint's NVFP4 experts stored one by one, merged; gate and up share a scale_2"""
    kinds = ("weight", "weight_scale", "weight_scale_2")
    return [
        *(
            WeightConverter(
                [f"*.gate_proj.{kind}", f"*.up_proj.{kind}"],
                f"w13_{kind}",
                operations=[
                    MergeModulelist(dim=0),
                    EqualSources() if kind == "weight_scale_2" else Concatenate(dim=1),
                ],
            )
            for kind in kinds
        ),
        *(
            WeightConverter(
                f"*.down_proj.{kind}", f"w2_{kind}", operations=[MergeModulelist(dim=0)]
            )
            for kind in kinds
        ),
    ]


# Step 3.5 / Step 5's per-expert scalars stacked on disk (`moe.gate_proj.input_scale`,
# `[E]` or one for all), NVFP4's `weight_scale_2` alike
STACKED_EXPERT_SCALARS = [
    WeightConverter(
        [rf"\.moe\.{p}_proj\.{kind}$" for p in projs],
        f".moe.experts.routed_experts.{param}_{kind}",
        operations=[ExpertScalars()],
    )
    for kind in ("input_scale", "weight_scale_2")
    for param, projs in (("w13", ("gate", "up")), ("w2", ("down",)))
]


def _stacked_experts(
    module: nn.Module, start: int, count: int, total: int, refuse: str | None = None
) -> list[WeightTransform]:
    """A mega-MoE layer's stacked experts: `count` of them from `start`, or `refuse`d"""
    ops = [Refuse(refuse)] if refuse else [Narrow(0, start, count, full=total)]
    return [
        WeightConverter(rf"^(?:routed_experts\.)?{name}$", name, operations=ops)
        for name, _ in module.named_parameters(recurse=False)
    ]


def _per_expert(module: nn.Module) -> list[WeightTransform]:
    """A mega-MoE layer's experts stored one by one, merged as their bytes"""
    return [
        WeightConverter(
            ["*.w1.weight", "*.w3.weight"],
            "w13_weight",
            operations=[ViewAsParam(), MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            "*.w2.weight",
            "w2_weight",
            operations=[ViewAsParam(), MergeModulelist(dim=0)],
        ),
    ]


# DeepGEMM MegaMoE experts' checkpoint tensors: (suffix, param suffix, ops after the
# merge into this rank's slots)
_MEGA_MOE_TENSORS = (
    ("weight", "weight", []),
    ("weight_scale", "weight_scale", [RepeatToParam()]),
    ("weight_scale_inv", "weight_scale_inv", []),
    ("weight_scale_2", "weight_scale_2", []),
    ("weight_global_scale", "weight_scale_2", [Reciprocal()]),
    ("input_scale", "input_scale", []),
    ("input_global_scale", "input_scale", [Reciprocal()]),
)


def _mega_moe_experts(module: nn.Module) -> list[WeightTransform]:
    """DeepGEMM MegaMoE experts as their bytes, into this rank's slots: one by one
    (merged), stacked (gathered), and a fused shared expert into `shared_l{1,2}_*`.
    Projections are renamed first, so a model's own `.w1` converters do not take them"""
    params = dict(module.named_parameters(recurse=False))
    slots = module.local_logical_experts
    present = sorted(set(slots))
    gather = (
        []
        if slots == present
        else [Gather(torch.tensor([present.index(s) for s in slots]), dim=0)]
    )
    stacked = Gather(torch.tensor(slots), dim=0)
    stacked.needs_full = True  # all experts read, then this rank's
    out: list[WeightTransform] = [
        *(
            rename(rf"^(\d+)\.(?:{names})\.", rf"\1.{proj}.")
            for proj, names in (
                ("gate", "w1|gate_proj"),
                ("up", "w3|up_proj"),
                ("down", "w2|down_proj"),
            )
        ),
        drop(r"^\d+\.(?:gate|up)\.input_(?:global_)?scale$"),
        rename(r"^shared_experts\.(?:w1|gate_proj)\.", "shared_gate."),
        rename(r"^shared_experts\.(?:w3|up_proj)\.", "shared_up."),
        rename(r"^shared_experts\.gate_up_proj\.", "shared_gate_up."),
        rename(r"^shared_experts\.(?:w2|down_proj)\.", "shared_down."),
        rename(r"\.weight_packed$", ".weight"),
        drop(r"^gate_up_proj\.input_(?:global_)?scale$"),
    ]
    for suffix, param_suffix, ops in _MEGA_MOE_TENSORS:
        for param, projs, fused_name in (
            ("w13", ("gate", "up"), "gate_up_proj"),
            ("w2", ("down",), "down_proj"),
        ):
            if (target := f"{param}_{param_suffix}") not in params:
                continue
            # Transformers v5's experts fused on disk (`[E, ...]`, gate rows then
            # up, one global scale per expert): this rank's slots gathered
            out.append(
                WeightConverter(
                    rf"^{fused_name}\.{suffix}$",
                    target,
                    operations=[ViewAsParam(), stacked, *ops],
                )
            )
            one = param_suffix == "weight_scale_2"  # one L1 scale per expert
            gate_up = EqualSources() if one else Concatenate(dim=1)
            out.append(
                WeightConverter(
                    [f"*.{p}.{suffix}" for p in projs],
                    target,
                    operations=[
                        ViewAsParam(),
                        MergeModulelist(dim=0),
                        *([gate_up] if len(projs) > 1 else []),
                        *gather,
                        *ops,
                    ],
                )
            )
    out += [
        WeightConverter(
            rf"^(?:routed_experts\.)?{name}$", name, operations=[ViewAsParam(), stacked]
        )
        for name in params
        if not name.startswith("shared_")
    ]
    if getattr(module, "has_fused_shared_experts", False):
        for suffix in ("weight", "weight_scale", "weight_scale_inv"):
            for param, sources, merge in (
                ("shared_l1", ["^shared_gate", "^shared_up"], [Concatenate(dim=0)]),
                ("shared_l1", ["^shared_gate_up"], []),
                ("shared_l2", ["^shared_down"], []),
            ):
                # `weight_scale_inv` into `weight_scale` unless staged as is
                target = f"{param}_{suffix}"
                if target not in params:
                    target = f"{param}_weight_scale"
                if target not in params:
                    continue
                repeat = [RepeatToParam()] if target.endswith("_scale") else []
                out.append(
                    WeightConverter(
                        [f"{s}.{suffix}$" for s in sources],
                        target,
                        operations=[ViewAsParam(), *merge, *repeat],
                    )
                )
    return out


def _mega_moe_shared_expert(module: nn.Module) -> list[WeightTransform]:
    """A shared expert fused into DeepGEMM MegaMoE experts: its tensors theirs"""
    if not getattr(getattr(module, "experts", None), "has_fused_shared_experts", 0):
        return []
    return [rename("^shared_experts.", "experts.shared_experts.")]


def _fused_shared_expert(module: nn.Module) -> list[WeightTransform]:
    """An AITER-fused shared expert as the routed experts' extra one; several are
    chunked by the model's stream (`split_fused_shared_experts`)"""
    if not getattr(module, "is_fused_shared_expert_enabled", False):
        return []
    if (getattr(module, "n_shared_experts", 1) or 1) > 1:
        return []
    new = f"experts.{module.n_routed_experts}."
    return [rename(rf"^{name}\.", new) for name in ("shared_expert", "shared_experts")]


def _deepseek_v4_moe(module: nn.Module) -> list[WeightTransform]:
    """A DeepSeek V4 MoE block's shared expert: AITER-fused, or `w2` as `down_proj`"""
    if getattr(module, "is_fused_shared_expert_enabled", False):
        return _fused_shared_expert(module)
    return [
        rename(r"^shared_experts\.w2\.", "shared_experts.down_proj."),
        *_mega_moe_shared_expert(module),
    ]


def _step5_attention(module: nn.Module) -> list[WeightTransform]:
    """Step 5's attention: the indexer's tensors (AMD), or its fused projections"""
    out: list[WeightTransform] = []
    if hasattr(getattr(module, "indexer", None), "sparse_indexer_q"):
        out.append(rename("^sparse_indexer_", "indexer.sparse_indexer_"))
    if getattr(module, "qkv_indexer_proj", None) is not None:
        indexer = ("sparse_indexer_q", "sparse_indexer_k", "sparse_indexer_z")
        out.append(
            fused("qkv_indexer_proj", "q_proj", "k_proj", "v_proj", *indexer, "g_proj")
        )
    elif getattr(module, "qkvg_proj", None) is not None:
        out.append(fused("qkvg_proj", "q_proj", "k_proj", "v_proj", "g_proj"))
    return out


def _mistral(module: nn.Module) -> list[WeightTransform]:
    """Mistral's consolidated checkpoint; q, k rows from interleaved rope pairs"""
    config = module.config
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    qkv = fused(
        ".self_attn.qkv_proj", ".attention.wq", ".attention.wk", ".attention.wv"
    )
    pairs = torch.arange(head_dim).view(-1, 2).t().reshape(-1)
    qkv.operations = [HeadPermute(pairs, head_dim, (0, 1)), *qkv.operations]
    return [
        *(
            rename(rf"\.attention\.{name}_fake_quantizer\.qscale_act$", new)
            for name, new in (
                ("kv", ".self_attn.kv_scale"),
                ("q", ".self_attn.attn.q_scale"),
                ("k", ".self_attn.k_scale"),
                ("v", ".self_attn.v_scale"),
            )
        ),
        *MISTRAL_FORMAT,
        qkv,
    ]


def _glm_ocr_mtp(module: nn.Module) -> list[WeightTransform]:
    """GLM-OCR's MTP layers (`_spec_layers`), their head the target's `lm_head`"""
    start = module.config.num_hidden_layers
    return [
        rename("^model.language_model.layers.", "model.layers."),
        rename("^lm_head.weight$", f"model.layers.{start}.shared_head.head.weight"),
        *_spec_layers(module),
    ]


def _hy_v3_mtp(module: nn.Module) -> list[WeightTransform]:
    """HY V3's MTP layers: `qkv_proj` stored per kv head, `gate_and_up_proj` [up, gate]"""
    config = module.config
    start, spec = config.num_hidden_layers, _mtp_layers(module)
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", heads)
    return [
        drop(rf"^model\.layers\.(?:{spec})\.(?:embed_tokens|shared_head)\."),
        rename("^lm_head.weight$", f"model.layers.{start}.shared_head.head.weight"),
        rename("gate_proj_bias", "gate_proj.bias"),
        rename("up_proj_bias", "up_proj.bias"),
        rename(".mlp.gate.wg.", ".mlp.gate."),
        rename(".mlp.router.gate.", ".mlp.gate."),
        *_spec_layers(module, own="enorm|hnorm|eh_proj|final_layernorm|shared_head"),
        WeightConverter(
            r"\.self_attn\.qkv_proj\.weight$",
            ".self_attn.qkv_proj.weight",
            operations=[GroupedQKV(heads, kv_heads)],
        ),
        WeightConverter(
            r"\.gate_and_up_proj\.weight$",
            ".gate_up_proj.weight",
            operations=[SwapHalves()],
        ),
    ]


def _longcat_mtp(module: nn.Module) -> list[WeightTransform]:
    """LongCat-Flash's MTP layer (`model.mtp.`) as layer `mtp_start_layer_idx`"""
    layer = f"model.layers.{module.model.mtp_start_layer_idx}."
    return [
        drop(r"^(?!model\.mtp\.)"),  # the target's
        rename(r"^model\.mtp\.embed_tokens\.", "model.embed_tokens."),
        rename(r"^model\.mtp\.layers\.0\.eh_proj\.", layer + "eh_proj."),
        rename(r"^model\.mtp\.layers\.0\.(enorm|hnorm)\.m\.", layer + r"\1."),
        rename(r"^model\.mtp\.norm\.", layer + "final_layernorm."),
        rename(
            r"^model\.mtp\.layers\.0\.(?:transformer_layer\.)?", layer + "mtp_block."
        ),
        MLA_QKV_A,
    ]


def _llama4_attention(module: nn.Module) -> list[WeightTransform]:
    """Llama 4's q, k rows from Meta's interleaved rope pairs into NeoX halves"""
    head_dim = module.head_dim
    halves = torch.arange(head_dim).view(-1, 2).t().reshape(-1)
    qkv = fused("qkv_proj", "q_proj", "k_proj", "v_proj")
    qkv.operations = [HeadPermute(halves, head_dim, (0, 1)), *qkv.operations]
    return [qkv]


def _falcon_attention(module: nn.Module) -> list[WeightTransform]:
    """Falcon's `query_key_value`, stored per kv head unless multi-query"""
    if module.total_num_kv_heads == 1:
        return []
    grouped = GroupedQKV(module.total_num_heads, module.total_num_kv_heads)
    return [
        WeightConverter(
            f"^query_key_value.{t}$", f"query_key_value.{t}", operations=[grouped]
        )
        for t in ("weight", "bias")
    ]


def _kimi_linear(module: nn.Module) -> list[WeightTransform]:
    """Kimi Linear / K3: the DSA indexer's names; `weight_packed` unless experts pack"""
    out = []
    if getattr(module.config, "dsa_config", None) is not None:
        out += [
            rename(".self_attn.indexer.wq_b.", ".self_attn.dsa_indexer.wq_index."),
            rename(
                ".self_attn.indexer.k_norm.", ".self_attn.dsa_indexer.k_index_norm."
            ),
            rename(".self_attn.indexer.", ".self_attn.dsa_indexer."),
        ]
    mega = any(getattr(m, "use_mega_moe", False) for m in module.modules())
    packed = any(n.endswith("w13_weight_packed") for n, _ in module.named_parameters())
    if not mega and not packed:
        out.append(rename(r"\.weight_packed$", ".weight"))
    return out


def _kimi_mla(module: nn.Module) -> list[WeightTransform]:
    """A K3 MLA / DSA layer's q_a and kv_a fused, with the output gate where built"""
    if getattr(module, "fused_qkv_a_g_proj", None) is not None:
        return [fused("fused_qkv_a_g_proj", "q_a_proj", "kv_a_proj_with_mqa", "g_proj")]
    if (fused_a := getattr(module, "fused_qkv_a_proj", None)) is None:
        return []
    sources = ["q_a_proj", "kv_a_proj_with_mqa"]
    if len(getattr(fused_a, "output_sizes", ())) == 3:
        sources.append("g_proj")
    return [fused("fused_qkv_a_proj", *sources)]


def _mimo_v2_attention(module: nn.Module) -> list[WeightTransform]:
    """MiMo V2.5's FP8 `qkv_proj`, fused and pre-sharded on disk with its scales"""
    if getattr(module.qkv_proj, "weight_scale_inv", None) is None:
        return []
    from vllm.model_executor.models.mimo_v2 import FusedFp8QKV

    return [
        WeightConverter(
            [r"^qkv_proj\.weight$", r"^qkv_proj\.weight_scale_inv$"],
            ["qkv_proj.weight", "qkv_proj.weight_scale_inv"],
            operations=[FusedFp8QKV()],
        )
    ]


def _qwen4_exp_qsa(module: nn.Module) -> list[WeightTransform]:
    """A Qwen4-Exp QSA layer's KV scales on itself; the indexer's qk in `qkv_proj`"""
    out = [
        rename(
            rf"^(?:{kv}_proj\.(?:{kv}|output)_scale|(?:attn\.)?_?{kv}_scale)$",
            f"_{kv}_scale",
        )
        for kv in "kv"
    ]
    if getattr(module, "fuse_indexer_projection", False):
        out.append(
            fused("qkv_proj", "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")
        )
    return out


def _interns1_pro(module: nn.Module) -> list[WeightTransform]:
    """InternS1-Pro: the VL names, its FoPE coefficients onto the rotary embedding"""
    return [
        *_CONVERSION_MAPPING["CohereCompassForConditionalGeneration"],
        *(
            rename(f"^{re.escape(old)}$", new)
            for old, new in module.get_frope_params_map().items()
        ),
    ]


def _voxtral(module: nn.Module) -> list[WeightTransform]:
    """Voxtral's consolidated checkpoint: its Whisper encoder in Whisper's names, its
    audio adapter, the rest its language model's (`_mistral`)"""
    causal = getattr(module.config.audio_config, "is_causal", False)
    layer = r"^whisper_encoder\.transformer\.layers\.(\d+)\."
    new = r"whisper_encoder.whisper_encoder.layers.\1."
    mlp = (
        (("w1", "mlp.gate_proj"), ("w2", "mlp.down_proj"), ("w3", "mlp.up_proj"))
        if causal
        else (("w1", "fc1"), ("w2", "fc2"))  # `WhisperEncoder`'s entry: `mlp.`
    )
    parts = (
        ("attention.wq", "self_attn.q_proj"),
        ("attention.wk", "self_attn.k_proj"),
        ("attention.wv", "self_attn.v_proj"),
        ("attention.wo", "self_attn.out_proj"),
        ("attention_norm", "self_attn_layer_norm"),
        ("ffn_norm", "final_layer_norm"),
        *((f"feed_forward.{w}", name) for w, name in mlp),
    )
    encoder = "whisper_encoder.whisper_encoder."
    return [
        rename(r"^mm_streams_embeddings\.embedding_module\.", ""),
        rename(r"^mm_whisper_embeddings\.", ""),
        rename(r"^whisper_encoder\.conv_layers\.0\.(?:conv\.)?", encoder + "conv1."),
        rename(r"^whisper_encoder\.conv_layers\.1\.(?:conv\.)?", encoder + "conv2."),
        *(rename(layer + re.escape(old) + r"\.", new + f"{n}.") for old, n in parts),
        rename(r"^whisper_encoder\.transformer\.norm\.", encoder + "layer_norm."),
        rename(r"^audio_language_projection\.0\.", "audio_language_adapter.w_in."),
        rename(r"^audio_language_projection\.2\.", "audio_language_adapter.w_out."),
        rename(
            r"^(?!whisper_encoder\.|audio_language_adapter\.|language_model\.)",
            "language_model.",
        ),
        WHISPER_QKV_BIAS,  # the encoder's k_proj has no bias
    ]


# NeMo's routed experts, one `[experts, in, out]` tensor a projection (Nemotron 3.5
# Transcribe's BF16 checkpoint); a non-gated MoE, its up projection in `w13`
NEMO_EXPERTS = [
    WeightConverter(
        rf"\.experts\.{nemo}$",
        f".experts.routed_experts.{w}_weight",
        operations=[Transpose(1, 2)],
    )
    for nemo, w in (("gate_and_up_projs", "w13"), ("down_projs", "w2"))
]


def _stacked_moe(*kinds: str) -> list[WeightTransform]:
    """Step's experts stacked on disk (`moe.gate_proj.<kind>`, `[E, I, H]`)"""
    experts = ".moe.experts.routed_experts"
    return [
        *(rename(rf"\.moe\.down_proj\.{k}$", f"{experts}.w2_{k}") for k in kinds),
        *(
            WeightConverter(
                [rf"\.moe\.gate_proj\.{k}$", rf"\.moe\.up_proj\.{k}$"],
                f"{experts}.w13_{k}",
                operations=[Concatenate(dim=1)],
            )
            for k in kinds
        ),
    ]


def _thinker(head: str = "language_model.lm_head.") -> list[WeightTransform]:
    """Qwen Omni / ASR checkpoints' `thinker.`"""
    return [
        rename("^thinker.lm_head.", head),
        rename("^thinker.model.", "language_model.model."),
        rename("^thinker.", ""),
    ]


QWEN4_EXP_FUSED = [
    fused(
        "hyper_connection.input_mix_weight_down_block_inject.weight",
        "hyper_connection.input_mix_weight_down.weight",
        "hyper_connection.block_inject_weight.weight",
    ),
    fused("ple.kv_proj", "ple.key_proj", "ple.value_proj"),
]
# Lfm3 audio's modules, then its audio generation's (depth decoder, codebooks) dropped
LFM3_AUDIO = [
    *_move("lfm.", "language_model.", "model.", "lm_head."),
    rename("^encoder.lfm.lfm.", "audio_encoder.lfm."),
    rename("^encoder.downsample.", "audio_encoder.downsample."),
    rename("^mlp.model.0.", "audio_projector.layer_norm."),
    rename("^mlp.model.1.", "audio_projector.linear_1."),
    rename("^mlp.model.3.", "audio_projector.linear_2."),
]
LFM3_AUDIO_DROPS = _drops(
    "^codebook_embedding. ^depth_linear. ^depth_embeddings. ^depthformer. "
    "^mask_embedding ^model.codebook_embedding. ^model.depth_decoder."
)
# A class's entry: its transforms, or a function of one of its modules returning them
VISION_QKV = [fused(".attn.qkv.", ".attn.q.", ".attn.k.", ".attn.v.")]
_CONVERSION_MAPPING: dict[str, "list[WeightTransform] | Callable"] = {
    "K2HorizonAttention": _k2_horizon,
    "KimiGatedDeltaNetAttention": _kimi_kda,
    "KimiK3DeltaAttention": _kimi_kda,
    "MiniCPMMoE": _minicpm_moe,
    # a MoE layer's experts (`fc1` / `fc2`, `[E, ...]`), not a dense layer's MLP
    "Moondream3TextMoE": lambda m: [
        rename(f"^fc{i}.weight$", f"fc{i}_weight") for i in (1, 2)
    ],
    "MuseSparkMoE": _muse_spark_moe,
    "ParallelEngramEmbedding": _engram_table,
    "FlintNvfp4Experts": _flint_experts,
    "Qwen4ExpPLESharedHostEmbedding": _leader_written,
    "MuseSparkMegaMoEExperts": lambda m: _stacked_experts(
        m, m.experts_start_idx, m.num_local_experts, m.num_experts
    ),
    "Step5MegaMoEExperts": lambda m: _stacked_experts(
        m,
        m.expert_start,
        m.num_local_experts,
        m.num_experts,
        refuse="Reloading transformed MegaMoE weights is unsupported"
        if m.ready
        else None,
    ),
    "DeepseekV4AiterMegaMoEExperts": _per_expert,
    "DeepseekV4MegaMoEExperts": _per_expert,
    "DeepGemmMegaMoEExperts": _mega_moe_experts,
    "DeepseekV4MoE": _deepseek_v4_moe,
    "Glm5NextMoE": lambda m: _mega_moe_shared_expert(m) + _fused_shared_expert(m),
    "AliceAISparseMoeBlock": _fused_shared_expert,
    # q, k, v and the beta in one projection, unless the beta is excluded from FP8
    "BailingMoeV3KimiDeltaAttention": lambda m: (
        []
        if m.qkvb_proj is None
        else [fused("qkvb_proj", *(f"{p}_proj" for p in "qkvb"))]
    ),
    "Glm4MoE": _fused_shared_expert,
    "Qwen3NextSparseMoeBlock": _fused_shared_expert,
    "K2HorizonMoVAAttention": _k2_horizon,
    "MistralLarge4ForCausalLM": MISTRAL_LARGE4,
    # the DSpark draft's tensors (`mtp_dspark.`) of the target's checkpoint
    "MistralLarge4DSparkForCausalLM": [
        drop(r"^(?!mtp_dspark\.)"),
        rename("^mtp_dspark.backbone.layers.", "model.layers."),
        *(
            rename(f"^mtp_dspark.backbone.{name}", f"model.{name}")
            for name in ("target_projection.", "target_norm.", "final_norm.")
        ),
        rename("^mtp_dspark.backbone.mask_embedding$", "model.mask_embedding"),
        rename(
            "^mtp_dspark.autoregressive_head.embedding.", "model.markov_head.markov_w1."
        ),
        rename(
            "^mtp_dspark.autoregressive_head.output.", "model.markov_head.markov_w2."
        ),
        rename("^mtp_dspark.autoregressive_head.joint_proj.", "model.joint_proj."),
        *MISTRAL_LARGE4,
    ],
    "MistralForCausalLM": _mistral,
    "Step5Attention": _step5_attention,
    "Gemma4Model": _gemma4,
    "GPT2LMHeadModel": GPT2,
    "GPT2ForSequenceClassification": GPT2,
    "BloomForCausalLM": BLOOM,
    "ColBERTModel": [COLBERT_LINEAR, rename("^bert.", "model.")],
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
    "BeamModel": _beam,
    "DFlashQwen3Model": [
        # whole module names, also bare or wildcarded (quant configs' exclusions)
        rename(r"(?<![A-Za-z0-9_])midlayer(?=\.|\*|$)", "layers.0"),
        rename(r"(?<![A-Za-z0-9_])encoder.output_norm_enc(?=\.|\*|$)", "hidden_norm"),
        rename(r"(?<![A-Za-z0-9_])encoder.fc(?=\.|\*|$)", "fc"),
    ],
    "Dots3NoteForCausalLM": _dots3_note,
    # the MTP layer's embedding, stored under `mtp.`
    "Dots3NoteMTP": lambda m: [
        rename(
            "^model.mtp.embed_tokens.",
            f"model.layers.{m.config.num_hidden_layers}.embed_tokens.",
        ),
        *_CONVERSION_MAPPING["DeepseekV32MTP"](m),
    ],
    "K3DSparkForCausalLM": _k3_dspark,
    "MiniMaxM3SparseForCausalLM": [
        drop("mtp."),
        # ModelOpt MXFP8 layers name the checkpoint's block scales this way
        rename("weight_scale_inv", "weight_scale"),
        # in the param's shard order (`MinimaxM3QKVParallelLinearWithIndexer`)
        MINIMAX_M3_QKV,
        GATE_UP,
    ],
    "MuseSparkForConditionalGeneration": MUSE_SPARK_OMNI,
    "MiMoMTP": _mimo_mtp,
    "ErnieMTP": _ernie_mtp,
    # a VL wrapper's language model repeats its transforms, applied once
    "DeepseekV41ForCausalLM": lambda m: _deepseek_v4(m.language_model, vl=True),
    "DeepseekV41LLMForCausalLM": _deepseek_v4,
    "DeepseekV4ForCausalLM": _deepseek_v4,
    "DeepseekV4ForConditionalGeneration": lambda m: _deepseek_v4(
        m.language_model, vl=True, image=m.vision is not None
    ),
    "AIMv2Model": [fused(".fc13", ".fc1", ".fc3")],
    "AXK1Model": [MLA_QKV_A],
    "AfmoeForCausalLM": [rename(".router.gate.weight$", ".gate.weight")],
    "AliceAIForCausalLM": [drop("^mtp.")],
    "AliceAIModel": [rename(".linear_attn.a_log_bias", ".linear_attn.A_log")],
    "Alpamayo2SuperForConditionalGeneration": [
        rename("^vlm.model.visual.", "visual."),
        rename("^vlm.lm_head.", "language_model.lm_head."),
        rename("^vlm.model.language_model.", "language_model.model."),
        drop("^expert."),
    ],
    "BagelForConditionalGeneration": [drop("^vit_pos_embed.pos_embed")],
    "BailingMoeV25Model": [
        rename(".attention.", ".self_attn."),
        rename(".mlp.gate.e_score_correction_bias$", ".mlp.gate.expert_bias"),
        rename(
            r"\.mlp\.experts\.(e_score_correction|expert)_bias$",
            ".mlp.gate.expert_bias",
        ),
        MLA_QKV_A,
    ],
    # a full-attention layer's output projection, `dense` as the linear ones'
    "BailingMoeV25MLAAttention": [rename("^dense.", "o_proj.")],
    "BailingMoeV3MTPModel": [
        rename(".attention.", ".self_attn."),
        rename("^attention.", "self_attn."),
        MLA_QKV_A,
    ],
    "BailingMoeV3VLForConditionalGeneration": [
        rename(".attention.", ".self_attn."),
        rename("^attention.", "self_attn."),
        *_move("model.", "", "visual", "linear_proj"),
        rename("^linear_proj.0", "linear_proj.linear_fc1"),
        rename("^linear_proj.2", "linear_proj.linear_fc2"),
        rename("^lm_head", "language_model.lm_head"),
        rename("^model.", "language_model.model."),
    ],
    "BeamForCausalLM": [rename(".self_attn.gate_proj", ".self_attn.output_gate_proj")],
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
            "cls.predictions.transform.LayerNorm.gamma", "mlm_head.layer_norm.weight"
        ),
        rename("cls.predictions.transform.LayerNorm.beta", "mlm_head.layer_norm.bias"),
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
    "BertWithRope": [rename("^model.", "")],
    "CLIPEmbeddingModel": [drop(".position_ids")],
    "CanaryForConditionalGeneration": [
        rename(r"(decoder.layers.\d+.\w+_attn).o_proj.", r"\1.out_projection."),
        rename(r"(decoder.layers.\d+.mlp).fc1.", r"\1.dense_in."),
        rename(r"(decoder.layers.\d+.mlp).fc2.", r"\1.dense_out."),
        fused(".encoder_attn.kv_proj", ".encoder_attn.k_proj", ".encoder_attn.v_proj"),
    ],
    "ChatGLMModel": [rename(".word_embeddings", "")],
    "Cohere2MoeForCausalLM": [drop("^lm_head.")],
    "Cohere2VisionForConditionalGeneration": [
        *_move("model.", "", "vision_tower.", "multi_modal_projector."),
        rename("^model.language_model.", "language_model.model."),
    ],
    "CohereASRModel": [
        fused(*(f".second_sub_layer.{p}" for p in ("kv_proj", "key_net", "value_net"))),
        fused(
            *(
                f".first_sub_layer.{p}"
                for p in ("qkv_proj", "query_net", "key_net", "value_net")
            )
        ),
    ],
    "CohereCompassForConditionalGeneration": [
        rename("^model.visual.", "visual."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.language_model.", "language_model.model."),
    ],
    "CohereForCausalLM": [drop("_quantizer."), drop("^lm_head")],
    "Cohere_VisionTransformer": [fused(".qkv.", ".q.", ".k.", ".v.")],
    "ColModernVBertForRetrieval": [
        rename("^model.text_model.layers.", "text_layers."),
        rename("^model.text_model.embeddings.", "text_embeddings."),
        rename("^model.text_model.final_norm.", "text_final_norm."),
        rename("^model.connector.modality_projection.", "connector.proj."),
        rename("^model.custom_text_proj.", "custom_text_proj."),
        rename("^model.vision_model.vision_model.", "vision_model.vision_model."),
        rename("^model.", ""),
    ],
    "ColPaliModel": [
        rename("^embedding_proj_layer.", "custom_text_proj."),
        *_move(
            "vlm.", "", "vision_tower.", "language_model.", "multi_modal_projector."
        ),
        *_move(
            "model.", "", "vision_tower.", "language_model.", "multi_modal_projector."
        ),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    "ColQwen3Model": [
        # the projection: ColPali's / TomoroAI's name, under any prefix
        rename(
            r"^(?:.*\.)?(?:custom_text_proj|embedding_proj_layer)\.",
            "custom_text_proj.",
        ),
        rename("^vlm.model.visual.", "visual."),
        rename("^vlm.lm_head.", "language_model.lm_head."),
        rename("^vlm.model.language_model.", "language_model.model."),
        rename("^model.visual.", "visual."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.language_model.model.", "language_model.model."),
        rename("^model.language_model.", "language_model.model."),
    ],
    "ColQwen3_5Model": [
        rename(
            r"^(?:.*\.)?(?:custom_text_proj|embedding_proj_layer)\.",
            "custom_text_proj.",
        ),
        rename("^language_model.", "language_model.model."),
        drop("^mtp."),
    ],
    "ColUltraVBertForRetrieval": [
        rename("^vision_model.(?!vision_model.)", "vision_model.vision_model.")
    ],
    "Cosmos3EdgeForConditionalGeneration": _cosmos3_edge(),
    "Cosmos3ForConditionalGeneration": [
        rename("^(layers|embed_tokens|norm).", r"language_model.model.\1."),
        rename(
            "^(blocks.|merger.|patch_embed.|pos_embed.|deepstack_merger_list.)",
            r"visual.\1",
        ),
        *_drops(
            "^audio_modality_embed(?:..*)?$ ^action_modality_embed(?:..*)?$ _moe_gen "
            ".add_q_proj. .add_k_proj. .add_v_proj. .to_add_out. .norm_added_q. "
            ".norm_added_k."
        ),
        *(rename(f".to_{p}.", f".{p[0]}_proj.") for p in ("q", "k", "v", "out")),
        *(rename(f".norm_{p}.", f".{p}_norm.") for p in "qk"),
        *_drops(
            ".input_quantizer. .weight_quantizer. .output_quantizer. ^proj_in. "
            "^proj_out. ^time_embedder. ^audio_proj_in. ^audio_proj_out. "
            "^action_proj_in. ^action_proj_out."
        ),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    "DSparkDeepseekV4ForCausalLM": _deepseek_v4_dspark,
    "DeepSeekV4MTP": _deepseek_v4_mtp,
    "DeepseekOCR2ForCausalLM": [
        *_move("model.", "language_model.model.", "embed_tokens.", "layers.", "norm."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", ""),
    ],
    "DeepseekV2Eagle3Model": [rename("midlayer.", "layers.0."), MLA_QKV_A],
    "DeepseekV2Model": [MLA_QKV_A],
    "DeepseekV32Model": [MLA_QKV_A],
    "DeepseekVLV2ForCausalLM": [rename("^language.", "language_model.")],
    "DiffusionGemmaForConditionalGeneration": [
        rename("^model.decoder.self_conditioning.", "self_conditioning."),
        rename("^model.decoder.", "model."),
        rename("^model.encoder.language_model.", "model."),
        *_move("model.encoder.", "", "vision_tower.", "embed_vision."),
    ],
    "DotsOCRForCausalLM": [
        rename(".attn.qkv_proj.", ".attn.qkv."),
        rename(".attn.out_proj.", ".attn.proj."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
        fused(".fc13.", ".fc1.", ".fc3."),
    ],
    # EAGLE's draft saves its body without `model.`, as its body's entry expects
    "EagleDeepseekV3ForCausalLM": [rename(r"^(?!lm_head\.|model\.)", "model.")],
    "EagleMistralLarge3ForCausalLM": [
        # the draft's input projection
        rename("^eagle_linear.weight$", "model.fc.weight"),
        rename("^eagle_linear.qscale_act$", "model.fc.input_scale"),
        rename("^eagle_linear.qscale_weight$", "model.fc.weight_scale"),
        *MISTRAL_FORMAT,
    ],
    # the draft's layers (`mtp.`), and the embedding and head it shares with the target
    "ExaoneMoeMTP": [
        drop(r"^(?!mtp\.)(?!.*(?:embed_tokens|lm_head))"),
        rename("^mtp.", "model."),
    ],
    "Ernie4_5_MoeModel": [
        drop("mtp"),
        rename(".moe_statics.e_score_correction_bias", ".gate.e_score_correction_bias"),
    ],
    # the text experts' router (`gate.weight`) and the vision ones' (`gate.weight_1`),
    # stored `[hidden, experts]`; the experts split by `split_text_vision_experts`
    "Ernie4_5_VLMoeMoE": [
        rename(
            ".mlp.moe_statics.e_score_correction_bias$", ".mlp.e_score_correction_bias"
        ),
        WeightConverter(
            r"\.mlp\.gate\.weight$",
            ".mlp.text_experts_gate.weight",
            operations=[Transpose()],
        ),
        WeightConverter(
            r"\.mlp\.gate\.weight_1$",
            ".mlp.vision_experts_gate.weight",
            operations=[Transpose()],
        ),
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
    "ExaoneForCausalLM": [fused(".gate_up_proj", ".c_fc_0", ".c_fc_1")],
    "ExaoneMoeForCausalLM": [drop("^mtp.")],
    "FalconH1ForCausalLM": [rename("A_log", "A"), rename("mamba", "mamba.mamba")],
    "FireRedASR2ForConditionalGeneration": [
        rename("llm.", "model.decoder."),
        rename("encoder.", "model.encoder.audio_encoder."),
        rename("encoder_projector.", "model.encoder_projector."),
        rename("net.0", "pre_layer_norm"),
        rename("net.1", "linear_expand"),
        rename("net.4", "linear_project"),
        drop("^model.encoder.audio_encoder.positional_encoding.pe"),
    ],
    "FlashModel": [MLA_QKV_A],
    "FunASREncoder": [
        fused(".self_attn.qkv", *(f".self_attn.{p}_proj" for p in "qkv"))
    ],
    "FunASRForConditionalGeneration": [
        *(rename(f"linear_{p}.", f"{p}_proj.") for p in ("q", "k", "v", "out")),
        rename("audio_adaptor.", "model.encoder.audio_adaptor."),
        rename("audio_encoder.", "model.encoder.audio_encoder."),
        rename("llm.model.", "model.decoder."),
        rename("llm.lm_head", "lm_head"),
    ],
    "FunAudioChatForConditionalGeneration": [drop("^audio_invert_tower.")],
    "GLM4VForCausalLM": [
        rename(".word_embeddings", ""),
        fused(
            *(f"linear_proj.{p}" for p in ("merged_proj", "gate_proj", "dense_h_to_4h"))
        ),
    ],
    "GPTJForCausalLM": [
        rename("(?<!.attn).attn.([qkv]_(?:scale|zero_point))$", r".attn.attn.\1"),
        drop("attn.bias"),
        drop("attn.masked_bias"),
    ],
    "Gemma3ForConditionalGeneration": [
        rename("^model.language_model.", "language_model.model."),
        *_move("model.", "", "vision_tower.", "multi_modal_projector."),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    "Gemma3nForCausalLM": [
        *_drops("embed_audio. embed_vision. audio_tower. vision_tower.")
    ],
    "Gemma3nForConditionalGeneration": [
        *_move("model.", "", "embed_audio.", "embed_vision."),
        rename("^model.language_model.", "language_model.model."),
        *_move("model.", "", "vision_tower.", "audio_tower.", "multi_modal_projector."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model", "language_model.model"),
    ],
    "Gemma3nTextModel": [
        # KV-shared layers only have `q_proj`: it keeps its name, K/V load nowhere
        QKV,
        *_move("", "self_decoder.", "embed_tokens.", "embed_tokens_per_layer."),
        rename(
            "^per_layer_model_projection.", "self_decoder.per_layer_model_projection."
        ),
        rename(
            "^per_layer_projection_norm.", "self_decoder.per_layer_projection_norm."
        ),
        rename("^altup_projections.", "self_decoder.altup_projections."),
    ],
    "Gemma4ForCausalLM": [
        *_drops("audio_tower. vision_tower. embed_audio. embed_vision."),
        rename("^model.language_model.", "model."),
    ],
    "Gemma4ForConditionalGeneration": [
        *_move(
            "model.", "", "vision_tower", "embed_vision", "audio_tower.", "embed_audio."
        ),
        rename("^model.language_model.", "language_model.model."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model", "language_model.model"),
    ],
    "Gemma4MTP": [*_move("", "model.", "pre_projection.", "post_projection.")],
    "Gemma4UnifiedForConditionalGeneration": [
        *_move("model.", "", "embed_audio.", "embed_vision."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.vision_embedder.", "vision_embedder."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model", "language_model.model"),
    ],
    "Glm4MoeLiteModel": [MLA_QKV_A],
    "Glm4vForConditionalGeneration": [
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.visual.", "visual."),
    ],
    "Glm4vVisionTransformer": VISION_QKV,
    "Glm5NextModel": [
        MLA_QKV_A,
        fused(
            ".in_proj_qkvbfg_a",
            *(f".{p}_proj" for p in ("q", "k", "v", "b", "f_a", "g_a")),
        ),
        fused(".conv1d", ".q_conv1d", ".k_conv1d", ".v_conv1d"),
    ],
    "Glm5NextVisionTransformer": VISION_QKV,
    "GlmAsrEncoder": [WHISPER_QKV_BIAS],
    "GlmAsrForConditionalGeneration": [drop("^audio_tower.embed_positions")],
    "GptOssForCausalLM": [
        rename(".self_attn.", ".attn."),
        rename(".embed_tokens.weight$", ".embedding.weight"),
        # Quark's experts (`gate_up_proj.weight`, ...), fused or one by one; experts
        # streamed one by one in vLLM's names (`experts.N.w13_weight`)
        rename(r"\.experts\.gate_up_proj\.", ".experts.routed_experts.w13_"),
        rename(r"\.experts\.down_proj\.", ".experts.routed_experts.w2_"),
        rename(r"\.experts\.(\d+)\.(gate_up_proj|w13)[._]", r".experts.\1.w13."),
        rename(r"\.experts\.(\d+)\.(down_proj|w2)[._]", r".experts.\1.w2."),
        *(
            WeightConverter(
                rf"experts.*.{w}\.{kind}",
                f"experts.routed_experts.{w}_{kind}",
                operations=[MergeModulelist(dim=0)],
            )
            for w in ("w13", "w2")
            for kind in "weight weight_scale weight_packed bias input_scale".split()
        ),
    ],
    "Granite4VisionForConditionalGeneration": [
        rename("^model.language_model.", "language_model.model."),
        *_move("model.", "", "layerwise_projectors.", "spatial_projectors."),
        *_move("model.", "", "image_newline", "vision_tower."),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    "GraniteMoeHybridModel": [
        rename(".A_log$", ".A"),
        rename(
            ".block_sparse_moe.input_linear.",
            ".block_sparse_moe.experts.routed_experts.w13_",
        ),
        rename(
            ".block_sparse_moe.output_linear.",
            ".block_sparse_moe.experts.routed_experts.w2_",
        ),
        rename(
            ".block_sparse_moe.router.layer.weight$", ".block_sparse_moe.gate.weight"
        ),
    ],
    # straight to vLLM's experts: `FUSED_EXPERTS` (Transformers' names) ran already
    "GraniteMoeModel": [
        rename("moe.input_linear.weight$", "moe.experts.routed_experts.w13_weight"),
        rename("moe.output_linear.weight$", "moe.experts.routed_experts.w2_weight"),
        rename(".router.layer.weight$", ".gate.weight"),
        rename(".router.weight$", ".gate.weight"),
    ],
    "GraniteSpeechForConditionalGeneration": [
        rename("^language_model.model.model.", "language_model.model.")
    ],
    "HCXVisionV2ForCausalLM": [
        drop("discrete_vision_model"),
        rename("^model.", ""),
        rename("^vision_model.", "visual."),
    ],
    "HYV4ForCausalLM": [MLA_QKV_A],
    "HrmTextForCausalLM": [rename(".attn.", ".self_attn.")],
    "InklingForConditionalGeneration": [
        *TML,
        rename("^model.audio.", "audio."),
        rename("^model.visual.", "visual.vision_encoder."),
    ],
    "InklingMTP": [
        drop(r"^(?!.*\.mtp\.|model\.llm\.embed_norm\.weight$)"),  # the target's
        rename("^model.llm.layers.", "model.layers."),
        # the target's embedding norm, applied before the depth layers
        rename("^model.llm.embed_norm.weight$", "model.backbone_embed_norm.weight"),
        drop("^model.llm.embed.weight$"),  # embedding and head: the target's
        rename("^model.llm.norm", "model.norm"),
        drop("^model.llm.unembed."),
        rename("^language_model.layers.", "model.layers."),
        rename("^language_model.lm_head.", "lm_head."),
        rename("^model.mtp.", "model."),
        *TML_LAYERS,
    ],
    "InternLM2ForCausalLM": [fused(".gate_up_proj", ".w1", ".w3")],
    "InternS1ForConditionalGeneration": [
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.language_model.", "language_model.model."),
        *_move("model.", "", "vision_tower.", "multi_modal_projector."),
    ],
    "InternS2MobiusModel": [
        fused(".in_proj_qkvz", (".in_proj_qkv", 3), ".in_proj_z"),
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
    "JambaForCausalLM": [rename(".self_attn.", "."), rename(".A_log", ".A")],
    "JinaEmbeddingsV5DecoderModel": [rename("^", "model.")],
    "JinaEmbeddingsV5EncoderModel": [rename("^", "model.")],
    "JinaForRanking": [drop("^lm_head.")],
    "KimiK3ForConditionalGeneration": [
        rename("^language_model.layers.", "language_model.model.layers."),
        rename("^mm_projector.proj.0", "mm_projector.linear_1"),
        rename("^mm_projector.proj.2", "mm_projector.linear_2"),
    ],
    "Lfm2ForCausalLM": [rename(".conv.", ".short_conv."), fused(".w13", ".w1", ".w3")],
    "Lfm2MoeForCausalLM": [rename(".conv.", ".short_conv.")],
    "Lfm2MoeModel": [
        rename(".conv.", ".short_conv."),
        rename("expert_bias", "gate.e_score_correction_bias"),
        fused(".feed_forward.w13.", ".feed_forward.w1.", ".feed_forward.w3."),
    ],
    "Lfm3AudioForConditionalGeneration": [
        *LFM3_AUDIO,
        rename("^model.language_model.", "language_model.model."),
        *_move("model.", "", "audio_encoder.", "audio_projector."),
        rename("^lm_head.", "language_model.lm_head."),
        *LFM3_AUDIO_DROPS,
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
    "LlamaNemotronVLChatModel": [drop("norm_mean"), drop("norm_std")],
    "LlamaNemotronVLForEmbedding": [
        rename("^language_model.layers.", "language_model.model.layers."),
        rename("^language_model.embed_tokens.", "language_model.model.embed_tokens."),
        rename("^language_model.norm.", "language_model.model.norm."),
        rename("^vision_model.encoder.", "vision_model.vision_model.encoder."),
        rename("^vision_model.embeddings.", "vision_model.vision_model.embeddings."),
        rename(
            "^vision_model.post_layernorm.", "vision_model.vision_model.post_layernorm."
        ),
    ],
    "LlavaOnevisionForConditionalGeneration": [
        rename("^model.language_model.", "language_model.model."),
        *_move(
            "model.", "", "vision_tower.", "multi_modal_projector.", "image_newline"
        ),
        rename("^lm_head.", "language_model.lm_head."),
    ],
    "LlavaOnevision2ForConditionalGeneration": [
        rename("^model.language_model.", "language_model.model."),
        rename("^model.visual.", "visual."),
        rename("^lm_head.", "language_model.lm_head."),
        rename("^model.", "language_model.model."),
    ],
    "LongCatFlashMTP": _longcat_mtp,
    # each embedder's projection, transposed, stacked into `oe_projection`
    "NgramEmbedding": [
        WeightConverter(
            "post_projs.*.weight",
            "oe_projection",
            operations=[MergeModulelist(dim=0), Transpose(1, 2)],
        )
    ],
    "LongcatFlashNgramForCausalLM": [drop("^model.mtp.")],
    "MBartDecoderNoPos": [
        fused(".encoder_attn.kv_proj", ".encoder_attn.k_proj", ".encoder_attn.v_proj")
    ],
    "Mamba2ForCausalLM": [rename(".A_log", ".A")],
    "MiMoForCausalLM": [drop("^model.mtp_layers.")],
    # its checkpoint is its Whisper `encoder`'s, whose entry then renames `fc1`
    "MingAudioEncoder": [
        rename(r"^(?!encoder\.)", "encoder."),
        rename("blocks.", "layers."),
        *(
            rename(f".attn.{n}.", f".self_attn.{n[0]}_proj.")
            for n in ("query", "key", "value")
        ),
        rename(".attn.out.", ".self_attn.out_proj."),
        rename(".attn_ln.", ".self_attn_layer_norm."),
        rename(".mlp_ln.", ".final_layer_norm."),
        rename(".mlp.0.", ".fc1."),
        rename(".mlp.2.", ".fc2."),
        rename("ln_post.", "layer_norm."),
        rename("positional_embedding$", "embed_positions.weight"),
    ],
    "MiniCPMOBaseModel": [drop("^tts")],
    "MiniCPMV2_6": [drop("^apm."), drop("^audio"), drop("^tts")],
    "MiniCPMV4_6ForConditionalGeneration": [
        rename("^model.vision_tower.vit_merger.", "vit_merger."),
        rename("^model.vision_tower.", "vpm."),
        *_move("model.", "", "vpm.", "vit_merger.", "merger."),
        rename("^model.language_model.", "language_model.model."),
        rename("^lm_head.", "language_model.lm_head."),
        drop("^mtp."),
    ],
    "MiniMaxM3MTP": _minimax_m3_mtp,
    "MiniMaxM3SparseForConditionalGeneration": [
        rename(".mlp.fc1.", ".fc1."),
        rename(".mlp.fc2.", ".fc2."),
        rename("^multi_modal_projector.", "vision_tower.multi_modal_projector."),
        rename("^patch_merge_mlp.", "vision_tower.patch_merge_mlp."),
    ],
    "MiniMaxVLVisionModel": [fused(".qkv_proj.", ".q_proj.", ".k_proj.", ".v_proj.")],
    "MistralLarge3ForCausalLM": [*MISTRAL_FORMAT],
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
    "ModernBertForTokenClassification": [drop("^drop")],
    "ModernBertModel": [
        rename("^model.layers.", "encoder_layer.layers."),
        rename("^layers.", "encoder_layer.layers."),
        rename("^model.", ""),
    ],
    "Moondream3ForCausalLM": [
        rename("^model.", ""),
        rename("vision.proj_mlp.", "vision_proj."),
        rename("^text.wte$", "text.wte.weight"),
        rename("text.lm_head.", "lm_head."),
        rename(".attn.qkv.", ".attn.qkv_proj."),
        rename(".attn.proj.", ".attn.out_proj."),
        *(rename(f".attn.tau.{t}", f".attn.tau_{t}") for t in ("alpha", "wq", "wv")),
        rename(".mlp.router.", ".mlp.gate."),
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
    "MolmoVisionBackbone": [fused(".merged_linear", ".gate_proj", ".up_proj")],
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
        *_move("model.", "", "whisper_encoder.", "vq_adaptor."),
        # Transformers' names
        rename("^model.audio_tower.", "whisper_encoder."),
        rename("^model.multi_modal_projector.linear_1.", "vq_adaptor.layers.0."),
        rename("^model.multi_modal_projector.linear_2.", "vq_adaptor.layers.2."),
        rename("^model.multi_modal_projector.norm.", "vq_adaptor.layers.3."),
        drop("^model."),
    ],
    "MuseGlimmerForCausalLM": [
        rename(
            "model.vision_tower.patch_embedder.position_embedding_table.weight",
            "model.vision_tower.positional_embedding_vlm",
        ),
        rename("model.vision_tower.layers.", "model.vision_tower.transformer."),
        *(rename(f".norm{i}.", f".ln_{i}.") for i in (1, 2)),
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
        *_move("model.", "", "vision_adapter.", "vision_projection."),
        rename("^model.perception_emb_norm.", "perception_emb_norm."),
    ],
    "MuseSparkDSparkForCausalLM": [
        drop(r"^(?:embed_tokens\.|mask_embedding$)"),  # the target's; the vocab row's
        DRAFT_MODEL_PREFIX,
        fused(*(f".feed_forward.{p}_proj." for p in ("gate_up", "gate", "up"))),
        fused(*(f".self_attn.{p}_proj." for p in ("qkv", "q", "k", "v", "o_gate"))),
    ],
    "Nemotron3_5TranscribeForConditionalGeneration": [
        rename(".mixer._fp32_params.", ".mixer."),
        rename("^llm.model.norm.", "language_model.model.norm_f."),
        rename("^llm.model.", "language_model.model."),
        rename("^llm.lm_head.", "language_model.lm_head."),
        rename("^backbone.", "language_model.model."),
        rename("^lm_head", "language_model.lm_head"),
        rename("^perception.", "audio_tower."),
        *NEMO_EXPERTS,
    ],
    "NemotronHForCausalLM": [
        rename("A_log", "A"),
        rename("embeddings", "embed_tokens"),
        rename("^backbone", "model"),
        drop("^mtp"),
    ],
    "NemotronHMTP": [
        rename("^language_model.", ""),
        drop(r"^(?!mtp\.)"),  # the target's
        drop("rotary_emb.inv_freq"),
        rename("^mtp.layers.", "model.layers."),
    ],
    "NemotronH_Nano_VL_V2": [
        rename("^language_model.backbone", "language_model.model"),
        rename("^vision_projector.vision_final_layernorm.", "vision_final_layernorm."),
        rename("^sound_projection.", "sound_encoder.projection."),
    ],
    "NemotronLabsDiffusionForBlockDiffusion": [rename("^encoder.", "model.")],
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
    # experts stacked on the rows, `w2` stored `[e * i, h]` as Nomic's
    "DbrxForCausalLM": [
        WeightConverter(
            [".experts.mlp.w1$", ".experts.mlp.v1$"],
            ".experts.routed_experts.w13_weight",
            operations=[ExpertRows("ffn_config.moe_num_experts"), Concatenate(dim=1)],
        ),
        WeightConverter(
            ".experts.mlp.w2$",
            ".experts.routed_experts.w2_weight",
            operations=[ExpertRows("ffn_config.moe_num_experts"), Transpose(1, 2)],
        ),
    ],
    "OPTForCausalLM": [rename("^decoder.", "model.decoder.")],
    "OlmoHybridModel": [
        fused(
            ".linear_attn.in_proj_qkvg",
            *(f".linear_attn.{p}_proj" for p in ("q", "k", "v", "g")),
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
        rename("^model.embed_tokens.", "embed_tokens."),
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
    "PixtralForConditionalGeneration": [
        # Mistral's consolidated names: the vision tower's and the adapter's
        rename("^vision_tower.", "vision_encoder."),
        rename("^multi_modal_projector.", "vision_language_adapter."),
        *(
            rename(
                rf"^vision_encoder.transformer.layers.(\d+).{old}(?=.|$)",
                rf"vision_encoder.transformer.layers.\1.{new}",
            )
            for old, new in (
                ("attention.wq", "attention.q_proj"),
                ("attention.wk", "attention.k_proj"),
                ("attention.wv", "attention.v_proj"),
                ("attention.wo", "attention.o_proj"),
                ("feed_forward.w1", "feed_forward.gate_proj"),
                ("feed_forward.w2", "feed_forward.down_proj"),
                ("feed_forward.w3", "feed_forward.up_proj"),
            )
        ),
        rename(".linear_1.", ".w_in."),
        rename(".linear_2.", ".w_out."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.vision_tower.", "vision_encoder."),
        rename("^model.multi_modal_projector.", "vision_language_adapter."),
        # the rest is the language model's, in its own format (`MistralForCausalLM`)
        rename(
            r"^(?!vision_encoder\.|vision_language_adapter\.|patch_merger\.|"
            r"pre_mm_projector_norm\.|language_model\.)",
            "language_model.",
        ),
    ],
    "Qwen2VisionTransformer": [fused(".qkv", ".q_proj", ".k_proj", ".v_proj")],
    "Qwen2_5OmniThinkerForConditionalGeneration": [
        *_thinker(),
        *_drops("^talker. ^token2wav."),
    ],
    "Qwen2_5_VisionTransformer": [
        fused(".mlp.gate_up_proj.", ".mlp.gate_proj.", ".mlp.up_proj."),
        fused(".attn.qkv.", ".attn.q.", ".attn.k.", ".attn.v."),
    ],
    "Qwen3ASRForConditionalGeneration": [
        *_thinker(),
        rename("^model.audio_tower.", "audio_tower."),
        rename("^model.language_model.", "language_model.model."),
        rename("^model.multi_modal_projector.linear_1.", "audio_tower.proj1."),
        rename("^model.multi_modal_projector.linear_2.", "audio_tower.proj2."),
        *_drops("^talker. ^code2wav."),
    ],
    "Qwen3ASRForcedAlignerForTokenClassification": [
        *_thinker("classifier."),
        *_drops("^talker. ^code2wav."),
    ],
    "Qwen3Eagle3Model": [rename("midlayer.", "layers.0.")],
    "Qwen3NextForCausalLM": [drop("^mtp.")],
    "Qwen3OmniMoeAudioEncoder": [
        fused(*(f".self_attn.{p}_proj." for p in ("qkv", "q", "k", "v")))
    ],
    "Qwen3Omni_VisionTransformer": VISION_QKV,
    "Qwen3_5ForCausalLMBase": [
        rename("^model.language_model.", "model."),
        drop("^mtp."),
    ],
    "Qwen3_VisionTransformer": [fused("attn.qkv.", "attn.q.", "attn.k.", "attn.v.")],
    "Qwen4ExpForCausalLM": [rename("^model.language_model.", "model.")],
    # a QSA layer owns its KV cache: its scales are its `_k_scale` / `_v_scale`
    "Qwen4ExpModel": lambda m: [
        *(
            rename(
                rf"^layers\.{i}\.self_attn\.(?:{kv}_proj\.(?:{kv}|output)_scale"
                rf"|attn\._?{kv}_scale|_?{kv}_scale)$",
                f"layers.{i}.self_attn._{kv}_scale",
            )
            for i in sorted(m._qsa_layer_ids)
            for kv in "kv"
        ),
        fused(".in_proj_qkvz", (".in_proj_qkv", 3), ".in_proj_z"),
        fused(".in_proj_ba", ".in_proj_b", ".in_proj_a"),
        *QWEN4_EXP_FUSED,
    ],
    "RadioModel": [
        *(
            rename(f"radio_model.model.patch_generator.{old}", f"embeddings.{new}")
            for old, new in (
                ("video_embedder", "video_patch_projection"),
                ("embedder", "patch_projection"),
                ("pos_embed", "position_embedding"),
                ("cls_token.token", "cls_register_token"),
            )
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
            *(f"attention.attention.{p}" for p in ("query", "key", "value")),
        ),
        fused("attention.qkv", *(f"attention.{p}_proj" for p in "qkv")),
    ],
    "RicursiveForCausalLM": [fused(".qkg_proj", ".q_proj", ".k_proj", ".g_proj")],
    "SiglipEmbeddingModel": [drop(".position_ids")],
    "SiglipVisionModel": _drops(
        "head.attention head.layernorm head.mlp head.probe packing_position_embedding"
    ),
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
    "Step3TextModel": _stacked_moe("weight"),
    "Step3p5Model": [*_stacked_moe("weight", "weight_scale"), *STACKED_EXPERT_SCALARS],
    "Step3p5ForCausalLM": [rename(".share_expert.", ".moe.share_expert.")],
    "Step5Model": [
        *_stacked_moe("weight", "weight_scale_inv"),
        *STACKED_EXPERT_SCALARS,
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
    "WhisperEncoder": [
        rename(".fc1.", ".mlp.fc1."),
        rename(".fc2.", ".mlp.fc2."),
        WHISPER_QKV_BIAS,  # its k_proj has no bias
    ],
    "WhisperForConditionalGeneration": [drop("^proj_out.")],
    "Zamba2ForCausalLM": [
        rename("A_log", "A"),
        rename("0.weight", "A.weight"),
        rename("1.weight", "B.weight"),
    ],
    "_TmlForCausalLMBase": TML,
}

_CONVERSION_MAPPING["BailingMoeV3ForCausalLM"] = [
    *_CONVERSION_MAPPING["BailingMoeV3MTPModel"],
    fused(".qkvb_proj", ".q_proj", ".k_proj", ".v_proj", ".b_proj"),
]
# the experts' correction bias: the router's param (`LagunaMoE`)
_CONVERSION_MAPPING["LagunaForCausalLM"] = [
    rename(
        r"\.mlp\.experts\.e_score_correction_bias$",
        ".mlp.experts.router.e_score_correction_bias",
    )
]
_CONVERSION_MAPPING["CohereAsrForConditionalGeneration"] = [
    # NeMo's names, with or without the `model.` body (Transformers saves it)
    rename(r"^(?:model\.)?transf_decoder\._decoder", "decoder"),
    rename(r"^(?:model\.)?transf_decoder\._embedding", "decoder.embedding"),
    rename("second_sub_layer.query_net", "second_sub_layer.q_proj"),
    rename("^log_softmax.mlp.layer0.", "proj_out."),
    rename("^(?!proj_out.)", "model."),
    *_CONVERSION_MAPPING["WhisperEncoder"],
    drop("model.conv.batch_norm.num_batches_tracked"),
    *_drops("^model.preprocessor.featurizer.fb ^model.preprocessor.featurizer.window"),
]
_CONVERSION_MAPPING["DeepseekV4Model"] = [
    fused("compressor.fused_wkv_wgate", "compressor.wkv", "compressor.wgate"),
    *DEEPSEEK_V4_LAYER,
]
_CONVERSION_MAPPING["Exaone4_5_ForConditionalGeneration"] = [
    *_CONVERSION_MAPPING["LlavaOnevision2ForConditionalGeneration"],
    drop("^mtp."),
]
_CONVERSION_MAPPING["InternS1ProForConditionalGeneration"] = _interns1_pro
_CONVERSION_MAPPING["KimiAudioForConditionalGeneration"] = [
    *_CONVERSION_MAPPING["WhisperEncoder"],
    rename("^model.encoder.", "audio_tower."),
    rename("^model.vq_adaptor.layers.0.", "multi_modal_projector.vq_adaptor_layers_0."),
    rename("^model.vq_adaptor.layers.3.", "multi_modal_projector.vq_adaptor_layers_3."),
    rename("^model.vq_adaptor.layers.4.", "multi_modal_projector.vq_adaptor_layers_4."),
    *_move("model.", "language_model.model.", "layers.", "embed_tokens.", "norm."),
    rename("^lm_head.", "language_model.lm_head."),
    *_drops("^model. ^mimo_layers. ^mimo_output. ^mimo_norm."),
]
_CONVERSION_MAPPING["KimiK25ForConditionalGeneration"] = [
    *_CONVERSION_MAPPING["KimiK3ForConditionalGeneration"],
    rename(".self_attn.attn.q_scale$", ".self_attn.mla_attn.mla_attn.q_scale"),
    rename(".self_attn.attn.k_scale$", ".self_attn.mla_attn.mla_attn.k_scale"),
    rename(".self_attn.attn.v_scale$", ".self_attn.mla_attn.mla_attn.v_scale"),
]
_CONVERSION_MAPPING["Lfm3OmniForConditionalGeneration"] = [
    *_CONVERSION_MAPPING["Gemma3ForConditionalGeneration"],
    *LFM3_AUDIO,
    *_move("model.", "", "audio_encoder.", "audio_projector."),
    *LFM3_AUDIO_DROPS,
]
_CONVERSION_MAPPING["LlamaNemotronVLForSequenceClassification"] = [
    rename("^model.", ""),
    *_CONVERSION_MAPPING["LlamaNemotronVLForEmbedding"],
]
# its text layers saved as `encoder.layers.` (Transformers reverses its vision
# tower's unanchored rename on save, and undoes it on load)
_CONVERSION_MAPPING["QianfanOCRForConditionalGeneration"] = [
    rename(r"^language_model\.model\.encoder\.layers\.", "language_model.model.layers.")
]
# its vision tower saved bare (`vision_tower.encoder.`), as vLLM's CLIP's `vision_model`
_CONVERSION_MAPPING["LlavaNextVideoForConditionalGeneration"] = [
    *_CONVERSION_MAPPING["LlavaOnevisionForConditionalGeneration"],
    rename(r"^vision_tower\.(?!vision_model\.)", "vision_tower.vision_model."),
]
_CONVERSION_MAPPING["MiMoV2OmniForCausalLM"] = [
    rename("^speech_embeddings.", "audio_encoder.speech_embeddings."),
    *_CONVERSION_MAPPING["LlavaOnevision2ForConditionalGeneration"],
    drop("^audio_tokenizer."),
]
_CONVERSION_MAPPING["MingV3ASRForConditionalGeneration"] = [
    rename("^model.audio.", "audio_encoder."),
    rename("^linear_proj_audio.", "audio_projector.proj."),
    *_drops("^model.visual. ^linear_proj."),
    *_CONVERSION_MAPPING["PaddleOCRVLForConditionalGeneration"],
]
_CONVERSION_MAPPING["Mistral3ForConditionalGeneration"] = [
    *_CONVERSION_MAPPING["Gemma3ForConditionalGeneration"],
    rename("^model.", "language_model.model."),
    rename(".activation_scale$", ".input_scale"),
    rename(".weight_scale_inv$", ".weight_scale"),
]
_CONVERSION_MAPPING["OpenAIPrivacyFilterForTokenClassification"] = [
    *_CONVERSION_MAPPING["GptOssForCausalLM"],
    *(
        rename(f".{proj}{suffix}$", f".routed_experts.{w}_{param}")
        for suffix, param in (
            ("_blocks", "weight"),
            ("_scales", "weight_scale"),
            ("", "weight"),
            ("_bias", "bias"),
        )
        for proj, w in (("gate_up_proj", "w13"), ("down_proj", "w2"))
    ),
    *(
        rename(f".{proj}.{kind}$", f".routed_experts.{w}_{kind}")
        for proj, w in (("gate_up_proj", "w13"), ("down_proj", "w2"))
        for kind in ("weight", "weight_scale", "bias", "input_scale")
    ),
]
_CONVERSION_MAPPING["Qwen3OmniMoeThinkerForConditionalGeneration"] = [
    *_thinker(),
    *_drops("^talker. ^code2wav."),
    *_CONVERSION_MAPPING["PaddleOCRVLForConditionalGeneration"],
]
_CONVERSION_MAPPING["Qwen3_5ForConditionalGeneration"] = [
    *_CONVERSION_MAPPING["CohereCompassForConditionalGeneration"],
    drop("^mtp."),
]
STEP_VIT = [
    rename(".attn.in_proj_weight", ".attn.qkv_proj.weight"),
    rename(".attn.in_proj_bias", ".attn.qkv_proj.bias"),
    rename(".mlp.c_fc", ".mlp.fc1"),
    rename(".mlp.c_proj", ".mlp.fc2"),
]
_CONVERSION_MAPPING["Step3p7ForConditionalGeneration"] = [
    *STEP_VIT,
    *_move(
        "model.", "", "vision_model.", "vit_large_projector.", "vit_large_projector"
    ),
    rename("^model.language_model.", "language_model.model."),
    rename("^model.language_model", "language_model.model"),
    *_CONVERSION_MAPPING["PaddleOCRVLForConditionalGeneration"],
    rename("^lm_head", "language_model.lm_head"),
]
# extends the base model's checkpoint (as its class does)
_CONVERSION_MAPPING["Step5ForConditionalGeneration"] = [
    *_CONVERSION_MAPPING["Step3p7ForConditionalGeneration"],
    rename("(?<!.moe).share_expert.", ".moe.share_expert."),
]
# the draft's tensors of the target's checkpoint (`mtp.`), its embedding and head
_CONVERSION_MAPPING["Qwen4ExpMTP"] = [
    rename("^model.language_model.", ""),
    rename("^language_model.", ""),
    rename("^embed_tokens.", "model.embed_tokens."),
    rename("^model.mtp.", "mtp."),
    rename(r"^(?:mtp\.|model\.)?shared_head\.head\.", "lm_head."),
    rename("^model.lm_head.", "lm_head."),
    drop(r"^(?!mtp\.|model\.embed_tokens\.|lm_head\.)"),
    rename("^mtp.", "model."),
]
_CONVERSION_MAPPING["Qwen4ExpMultiTokenPredictor"] = [
    *_CONVERSION_MAPPING["InternS2MobiusModel"],
    *QWEN4_EXP_FUSED,
]
_CONVERSION_MAPPING["StepVLForConditionalGeneration"] = [
    *STEP_VIT,
    *_CONVERSION_MAPPING["PaddleOCRVLForConditionalGeneration"],
]
_CONVERSION_MAPPING["InternS2PreviewForConditionalGeneration"] = [
    *_CONVERSION_MAPPING["Qwen3_5ForConditionalGeneration"],
    *_drops("^model.time_series. ^time_series."),
]


def _mtp_layers(module: nn.Module) -> str:
    """The MTP layers' indices, after the decoder's (`num_hidden_layers + i`)"""
    config = module.config
    start = config.num_hidden_layers
    return "|".join(str(start + i) for i in range(config.num_nextn_predict_layers))


def _spec_layers(
    module: nn.Module, own: str = "enorm|hnorm|eh_proj|shared_head"
) -> list[WeightTransform]:
    """DeepSeek V3-style MTP layers: all but `own` under `mtp_block.`, one embedding"""
    start, spec = module.config.num_hidden_layers, _mtp_layers(module)
    return [
        rename(rf"^model\.layers\.{start}\.embed_tokens\.", "model.embed_tokens."),
        drop(rf"^model\.layers\.(?:{spec})\.embed_tokens\."),
        rename(
            rf"^(model\.layers\.(?:{spec}))\.(?!(?:{own})(?:\.|$))", r"\1.mtp_block."
        ),
    ]


def _mla_qkv_a(module: nn.Module) -> list[WeightTransform]:
    """MLA's latent projections fused, when built (a `q_lora_rank`)"""
    if getattr(module.config, "q_lora_rank", None) is None:
        return []
    return [MLA_QKV_A]


def _step_mtp(module: nn.Module) -> list[WeightTransform]:
    """Step's MTP layers, without `transformer.`; the head `shared_head.output`"""
    return [
        rename(r"^layers\.", "model.layers."),
        drop("tok_embeddings"),
        rename(rf"^(model\.layers\.(?:{_mtp_layers(module)}))\.transformer\.", r"\1."),
        rename(r"\.shared_head\.output\.", ".shared_head.head."),
        *_spec_layers(module),
    ]


def _hy_v4_mtp(module: nn.Module) -> list[WeightTransform]:
    """HY V4's `model.mtp_layers.<i>` as layer `num_hidden_layers + i`"""
    start, spec = module.config.num_hidden_layers, _mtp_layers(module)
    own = "enorm|hnorm|eh_proj|final_layernorm|shared_head"
    out = [
        *(
            rename(rf"^model\.mtp_layers\.{i}\.", f"model.layers.{start + i}.")
            for i in range(module.config.num_nextn_predict_layers)
        ),
        drop(rf"^model\.layers\.(?:{spec})\.(?:embed_tokens|shared_head)\."),
        rename(r"^lm_head\.weight$", f"model.layers.{start}.shared_head.head.weight"),
        rename(
            rf"^(model\.layers\.(?:{spec}))\.(?!(?:{own})(?:\.|$))", r"\1.mtp_block."
        ),
        rename(r"\.gate\.e_score_correction_bias$", ".expert_bias"),
        *_mla_qkv_a(module),
    ]
    if getattr(module.quant_config, "weight_block_size", None) is not None:
        out.append(rename(r"(\.mlp\.experts\..*)\.scale$", r"\1.weight_scale_inv"))
    return out


def _kimi_k3_mtp(module: nn.Module) -> list[WeightTransform]:
    """K3's MTP layers in the multimodal checkpoint; MLA's projections fused if built"""
    params = [name for name, _ in module.named_parameters()]
    out = [rename(r"^language_model\.", "")]
    mega = any(getattr(m, "use_mega_moe", False) for m in module.modules())
    if not mega and not any(n.endswith("w13_weight_packed") for n in params):
        out.append(rename(r"\.weight_packed$", ".weight"))
    out += _spec_layers(module)
    gate = getattr(module.config, "mla_use_output_gate", False)
    fusions = {}
    for name in params:
        for target in (".fused_qkv_a_g_proj.", ".fused_qkv_a_proj."):
            if target in name:
                prefix = name.split(target)[0]
                parts = [".q_a_proj", ".kv_a_proj_with_mqa"]
                parts += [".g_proj"] if gate else []
                fusions[prefix] = fused(
                    prefix + target[:-1], *(prefix + p for p in parts)
                )
    return [*out, *fusions.values()]


def _jina_roberta(module: nn.Module) -> list[WeightTransform]:
    """jina-embeddings-v3, its LoRA weights merged with the text-matching adapter"""
    config = module.config
    out = [
        rename("emb_ln", "embeddings.LayerNorm"),
        rename("mixer.Wqkv", "attn.qkv_proj"),
        rename("mixer.out_proj", "attn.out_proj"),
        rename("norm1", "attn_ln"),
        rename("mlp.fc1.", "mlp.up_proj."),
        rename("mlp.fc2", "mlp.down_proj"),
        rename("norm2", "mlp_ln"),
    ]
    if getattr(config, "lora_rank", None):
        lora = [
            r"\.parametrizations\.weight\.original$",
            r"\.parametrizations\.weight\.0\.lora_A$",
            r"\.parametrizations\.weight\.0\.lora_B$",
        ]
        scaling = config.lora_alpha / config.lora_rank
        out.append(WeightConverter(lora, ".weight", operations=[MergeLora(scaling)]))
    return out


# MTP drafts: the target's checkpoint, their layers renamed
_CONVERSION_MAPPING["DeepSeekMTP"] = lambda m: [*_spec_layers(m), *_mla_qkv_a(m)]
_CONVERSION_MAPPING["Glm4MoeMTP"] = lambda m: [
    rename(
        r"^lm_head\.weight$",
        f"model.layers.{m.config.num_hidden_layers}.shared_head.head.weight",
    ),
    *_spec_layers(m),
]
_CONVERSION_MAPPING["Glm5NextMTP"] = lambda m: [
    rename(r"^model\.language_model\.", "model."),
    *_spec_layers(m),
    *_mla_qkv_a(m),
]
_CONVERSION_MAPPING["HYV4MTP"] = _hy_v4_mtp
_CONVERSION_MAPPING["KimiK3MTP"] = _kimi_k3_mtp
_CONVERSION_MAPPING["Step3p5MTP"] = _step_mtp
_CONVERSION_MAPPING["Step5MTP"] = _step_mtp
_CONVERSION_MAPPING["Exaone4_5_MTP"] = [
    rename(r"^model\.language_model\.embed_tokens\.", "model.embed_tokens."),
    drop(r"^(?!mtp\.)(?!.*(?:embed_tokens|lm_head))"),
    rename("^mtp.", "model."),
]
_CONVERSION_MAPPING["Qwen3NextMTP"] = [rename("^mtp.", "model.")]
_CONVERSION_MAPPING["Qwen3_5MTP"] = [
    # the target's tensors, but for the embedding and head it shares
    drop(r"^(?!mtp\.)(?!.*(?:embed_tokens|lm_head))"),
    rename("^mtp.", "model."),
    rename(r"language_model\.(?=embed_tokens\.)", ""),
]
_CONVERSION_MAPPING["BailingMoeV25MTPModel"] = [
    rename(r"^model\.word_embeddings\.", "model.embed_tokens."),
    rename(".attention.dense.", ".self_attn.o_proj."),
    rename(".attention.", ".self_attn."),
    rename(r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.gate.expert_bias"),
    MLA_QKV_A,
]
_CONVERSION_MAPPING["BailingMoeV3MultiTokenPredictor"] = [
    rename(r"^word_embeddings\.", "embed_tokens."),
    rename(r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.gate.expert_bias"),
]
# EAGLE-3 drafts: the decoder layer `midlayer`, the body's names without `model.`
_CONVERSION_MAPPING["Eagle3LlamaForCausalLM"] = [
    rename(r"^(?:model\.)?midlayer\.", "model.layers.0."),
    drop("mask_hidden"),  # P-EAGLE's, parallel drafting is unsupported
]
_CONVERSION_MAPPING["DSpineQwen3Model"] = [
    *_CONVERSION_MAPPING["DFlashQwen3Model"],
    rename("transfer_embeddings", "transfer_codebook"),
]
_CONVERSION_MAPPING["Qwen3XPressForCausalLM"] = [
    rename(r"^xpress_head\.mlp\.(gate|up|down)_proj\.", r"xpress_head.mlp_\1."),
    drop("mask_embedding"),
    # the body's names without `model.`, for its entry (`DFlashQwen3Model`)
    rename(r"^(?!model\.|lm_head\.|draft_id_to_target_id$)", "model."),
]
_CONVERSION_MAPPING["BertSpladeSparseEmbeddingModel"] = [
    *(
        rename(rf"^(?:model\.|bert\.)?cls\.predictions\.{old}\.", f"mlm_head.{new}.")
        for old, new in (
            (r"transform\.dense", "dense"),
            (r"transform\.LayerNorm", "layer_norm"),
            ("decoder", "decoder"),
        )
    ),
    *_CONVERSION_MAPPING["BertEmbeddingModel"],
]
_CONVERSION_MAPPING["JinaRobertaModel"] = _jina_roberta
_CONVERSION_MAPPING["DeepseekV2MoE"] = _fused_shared_expert
_CONVERSION_MAPPING["HYV3Model"] = [rename(".router.gate.", ".gate.")]
_CONVERSION_MAPPING["HYV4Model"] = [
    rename(r"\.gate\.e_score_correction_bias$", ".expert_bias"),
    rename(".router.gate.", ".gate."),
    rename(r"\bhc_fn$", "hc_fn.weight"),
    rename(r"\bhc_head_fn$", "hc_head_fn.weight"),
]
# the loop gates (`gate_projections.<i>`): converters, so `gate_proj`'s do not take them
_CONVERSION_MAPPING["IQuestLoopCoderModel"] = [
    WeightConverter(
        rf"gate_projections\.(\d+)\.{tensor}$",
        rf"gate_projections.\1.gate_proj.{tensor}",
        operations=[OutputConcatenate()],
    )
    for tensor in ("weight", "bias")
]
# ModelOpt's checkpoint: the text model at the top, one scale for all experts
_CONVERSION_MAPPING["Llama4ForConditionalGeneration"] = [
    rename(r"^model\.", "language_model.model."),
    rename(r"^lm_head\.weight$", "language_model.lm_head.weight"),
    *(
        WeightConverter(
            rf"\.feed_forward\.experts\.{proj}_{kind}_scale$",
            f".feed_forward.experts.routed_experts.{w}_{kind}_scale",
            operations=[RepeatToParam()],
        )
        for proj, w in (("down_proj", "w2"), ("gate_up_proj", "w13"))
        for kind in ("input", "weight")
    ),
]
_CONVERSION_MAPPING["OpenPanguModel"] = [
    rename(r"(?<!gate\.)e_score_correction_bias", "gate.e_score_correction_bias")
]
_CONVERSION_MAPPING["OpenVLAForActionPrediction"] = [
    rename("^vision_backbone.featurizer.", "vision_backbone.dinov2_featurizer."),
    rename("^vision_backbone.fused_featurizer.", "vision_backbone.siglip_featurizer."),
    rename(r"\.(ls[12])\.scale_factor", r".\1.gamma"),  # timm's LayerScale
]
_CONVERSION_MAPPING["WhisperDecoder"] = [
    *_CONVERSION_MAPPING["WhisperEncoder"],
    WeightConverter(
        [".encoder_attn.k_proj.bias", ".encoder_attn.v_proj.bias"],
        ".encoder_attn.kv_proj.bias",
        operations=[ZeroMissingSource(), OutputConcatenate()],
    ),
    fused(".encoder_attn.kv_proj", ".encoder_attn.k_proj", ".encoder_attn.v_proj"),
]
_CONVERSION_MAPPING["AXK1MoE"] = _fused_shared_expert
# EAGLE / DSpark drafts saved without the `model.` prefix
for _draft in (
    "Eagle3DeepseekV2ForCausalLM",
    "EagleLlama4ForCausalLM",
    "LiLiCorrForCausalLM",  # its head's `lilicorr.`, saved with `model.` or not
    "DSpineQwen3ForCausalLM",
):
    _CONVERSION_MAPPING[_draft] = [DRAFT_MODEL_PREFIX]
# its attention ships `q_proj` alone (k / v are the target's): kept, not fused
_CONVERSION_MAPPING["Gemma4DSparkForCausalLM"] = [
    DRAFT_MODEL_PREFIX,
    fused(".self_attn.q_proj", ".self_attn.q_proj"),
]
_CONVERSION_MAPPING["GlmOcrMTP"] = _glm_ocr_mtp
_CONVERSION_MAPPING["HYV3MTP"] = _hy_v3_mtp
_CONVERSION_MAPPING["Llama4Attention"] = _llama4_attention
# Llama 4's experts `[E, in, out]` on disk (vLLM's transposed), ModelOpt FP8's scales too
_CONVERSION_MAPPING["Llama4MoE"] = [
    WeightConverter(
        rf"experts\.routed_experts\.{w}_{kind}$",
        f"experts.routed_experts.{w}_{kind}",
        operations=[Transpose(1, 2, check_dims=True)],
    )
    for w in ("w13", "w2")
    for kind in ("weight", "weight_scale")
]
_CONVERSION_MAPPING["FalconAttention"] = _falcon_attention
_CONVERSION_MAPPING["KimiLinearModel"] = _kimi_linear
for _mla in ("KimiMLAAttention", "MultiHeadLatentAttention", "KimiK3DSAAttention"):
    _CONVERSION_MAPPING[_mla] = _kimi_mla
# the indexer's `wk` and `weights_proj`, one GEMM where built so
_CONVERSION_MAPPING["KimiK3DSAIndexer"] = lambda m: (
    [fused("wkv_index", "wk", "weights_proj")] if hasattr(m, "wkv_index") else []
)
_CONVERSION_MAPPING["MiMoV2Attention"] = _mimo_v2_attention
_CONVERSION_MAPPING["Qwen4ExpQSAAttention"] = _qwen4_exp_qsa
_CONVERSION_MAPPING["VoxtralForConditionalGeneration"] = _voxtral
# the MTP head's NeMo tensors (`llm.mtp.`)
_CONVERSION_MAPPING["Nemotron3_5TranscribeMTP"] = [
    rename("^llm.mtp.", "mtp."),
    *_CONVERSION_MAPPING["NemotronHMTP"],
    *NEMO_EXPERTS,
]
# Classes whose checkpoints are another class's: the same entry
for _entry, _aliases in {
    "ColBERTLfm2Model": ["ColBERTModernBertModel"],
    "ColQwen3_5Model": ["D3ForSequenceClassification"],
    "DeepseekOCR2ForCausalLM": ["DeepseekOCRForCausalLM"],
    "GLM4VForCausalLM": ["GLM4VModel"],
    "GraniteMoeModel": ["GraniteMoeSharedModel"],
    "Lfm2ForCausalLM": ["Lfm2Model", "Lfm3ForCausalLM", "Lfm3Model"],
    "InternS1ForConditionalGeneration": ["Lfm2VLForConditionalGeneration"],
    "Gemma3ForConditionalGeneration": [
        "Lfm3VlForConditionalGeneration",
        "LlavaForConditionalGeneration",
        "PaliGemmaForConditionalGeneration",
    ],
    "LlavaOnevisionForConditionalGeneration": ["LlavaNextForConditionalGeneration"],
    "MiniCPMV2_6": ["MiniCPMV4_0", "MiniCPMV4_5"],
    "MixtralForCausalLM": ["MixtralModel"],
    "Mamba2ForCausalLM": ["MambaForCausalLM"],
    "Qwen2VisionTransformer": ["BlipVisionModel"],
    "LlavaOnevision2ForConditionalGeneration": [
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
    ],
    "CohereCompassForConditionalGeneration": ["Qwen3VLForConditionalGeneration"],
    "InternS2MobiusModel": ["Qwen3_5Model", "Qwen3_5MultiTokenPredictor"],
    "PaddleOCRVLForConditionalGeneration": ["Step3VLForConditionalGeneration"],
    "WhisperEncoder": ["UltravoxTransformerProjector"],
    "DeepSeekMTP": ["Glm4MoeLiteMTP", "DeepseekV32MTP", "OpenPanguMTP"],
    "Eagle3LlamaForCausalLM": ["Eagle3Qwen3ForCausalLM"],
}.items():
    for _alias in _aliases:
        _CONVERSION_MAPPING[_alias] = _CONVERSION_MAPPING[_entry]


def _shard_order(shard) -> tuple:
    """Shards in a fused param's order: indices, then q, k, v, then the others"""
    if isinstance(shard, int):
        return (0, shard)
    return (1, {"q": 0, "k": 1, "v": 2}.get(shard, 3), str(shard))


def from_mapper(mapper: "WeightsMapper") -> list[WeightTransform]:
    """A vLLM `WeightsMapper` as transforms: its rules in order, stacked shards fused.
    TODO: rewrite the helpers building them to return transforms directly."""
    rules: list[WeightTransform] = list(mapper.orig_to_new_renaming)
    for pattern, new in mapper.orig_to_new_regex.items():
        rules.append(
            drop(pattern.pattern) if new is None else rename(pattern.pattern, new)
        )
    for kind, wrap in (
        ("orig_to_new_substr", re.escape),
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
    """Per-expert checkpoints merged, by each `FusedMoE` layer's `ckpt_*_proj_name`"""
    names = {
        (m.ckpt_gate_proj_name, m.ckpt_up_proj_name, m.ckpt_down_proj_name)
        for m in model.modules()
        if hasattr(m, "ckpt_gate_proj_name")
    }
    out: list[WeightTransform] = []
    w13 = TRANSFORMERS_EXPERTS["experts.gate_up_proj"].removesuffix("weight")
    w2 = TRANSFORMERS_EXPERTS["experts.down_proj"].removesuffix("weight")
    # `weight` also takes its scales (`weight_scale_inv`); GPTQ / AWQ's tensors
    for gate, up, down in sorted(names, key=str):
        for kind in ("weight", "bias", "qweight", "qzeros", "scales"):
            gate_up = [f"experts.*.{p}.{kind}" for p in (gate, up) if p]
            merge = [MergeModulelist(dim=0), Concatenate(dim=1)]
            out.append(WeightConverter(gate_up, w13 + kind, operations=merge))
            out.append(
                WeightConverter(
                    f"experts.*.{down}.{kind}",
                    w2 + kind,
                    operations=[MergeModulelist(dim=0)],
                )
            )
    return out


def _entry_name(cls: type) -> str | None:
    """The class's entry, or the one it inherits; vLLM's classes only"""
    return next(
        (
            c.__name__
            for c in cls.__mro__
            if c.__name__ in _CONVERSION_MAPPING and c.__module__.startswith("vllm.")
        ),
        None,
    )


def get_conversion_mapping(model: nn.Module) -> list[WeightTransform]:
    """The entries of the model's classes, each scoped to its instances, outermost
    first"""
    instances: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        if (cls := _entry_name(type(module))) is not None:
            instances.setdefault(cls, []).append(name)
    model_types = {
        module.config.model_type
        for module in model.modules()
        if getattr(getattr(module, "config", None), "model_type", None)
    }
    out = to_vllm([e for t in sorted(model_types) for e in transformers_experts(t)])
    seen: dict[tuple, list[str]] = {}
    out += _layer_experts(model)
    for cls, names in sorted(instances.items(), key=lambda kv: min(map(len, kv[1]))):
        entry = _CONVERSION_MAPPING[cls]  # a list, or a function of the module
        if callable(entry):  # each instance its own, scoped to it (a layer's)
            scoped = [
                (name, t) for name in names for t in entry(model.get_submodule(name))
            ]
        else:
            parts = os.path.commonprefix([name.split(".") for name in names])
            scope = ".".join(p for p in parts if not p.isdigit())
            scoped = [(scope, t) for t in entry]
        for scope, transform in scoped:
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
    return [*DRAFT_VOCAB, *out, QKV, GATE_UP]


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
