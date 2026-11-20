# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling MTP (Multi-Token Prediction) draft model (NVIDIA).

Mirrors the reference `mtp_model.py` shipped with the checkpoint: each MTP
depth `i` owns `hidden_norm` / `embed_norm` RMSNorms, an `input_proj`
(`2H -> H`) and a full Inkling transformer block (dense MLP, with the
same short convolutions as the backbone; its attention is full or sliding-window
per depth, selected by `mtp_config.local_layer_ids`). When enabled, a shared
`chain_norm` is applied after every depth; its output is both the logits input
and the previous hidden state fed to the next depth.

The draft shares the target's token embedding table and LM head
(`load_eagle_model` wires those references) and applies the backbone
`embed_norm` on top: the depth layers were trained on the same normed
embeddings the backbone consumes (their own `embed_norm` weights are
near-identity trims, unlike the backbone's whitening `embed_norm`).
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsMultiModalEmbeddings
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.inkling.common.loading import inkling_expert_weights
from vllm.sequence import IntermediateTensors

from ..configs import InklingModelConfig
from .layernorm import InklingRMSNorm
from .model import (
    InklingDecoderLayer,
    InklingReplicatedEmbedding,
    _TmlForCausalLMBase,
)
from .ops.norm import embed_dual_rmsnorm_cat, embed_rmsnorm

# Checkpoint attention projections (wq_du/wk_dv/wv_dv/wr_du) -> fused qkvr.
# Mirrors the backbone's hf_to_vllm_mapper.orig_to_new_stacked; kept as a
# local (pname, wname, shard) list since the MTP loader remaps by hand.
_ATTENTION_PARAMS_MAPPING = [
    ("qkvr", "wq_du", 0),
    ("qkvr", "wk_dv", 1),
    ("qkvr", "wv_dv", 2),
    ("qkvr", "wr_du", 3),
]


def _select_mtp_depth_count(n_predict: int, num_spec: int | None) -> int:
    num_layers = min(n_predict, num_spec) if num_spec else n_predict
    if num_layers <= 0:
        raise ValueError(
            "Inkling MTP requires num_nextn_predict_layers and "
            "num_speculative_tokens to select at least one depth layer."
        )
    return num_layers


class InklingMTPDepthLayer(nn.Module):
    """One MTP depth: norm both inputs, fuse (2H->H), run a Inkling block."""

    def __init__(
        self,
        config: InklingModelConfig,
        prefix: str,
        is_local: bool,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.input_proj",
        )
        # A force-dense-MLP block; ``is_local`` selects sliding-window vs
        # full attention (the swa_* head config and sliding_window window) to
        # match this depth's checkpoint transformer_block weights.
        self.transformer_block = InklingDecoderLayer(
            config,
            layer_id=0,
            is_local=is_local,
            quant_config=quant_config,
            prefix=f"{prefix}.transformer_block",
            force_dense_mlp=True,
        )

    def forward(self, combined: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # ``combined`` is the fused-normed [rmsnorm(hidden) | embed_norm(emb)]
        # input, built by InklingMultiTokenPredictor.fused_input_cat in one launch.
        hidden = self.input_proj(combined)
        # The short conv self-fetches its paged SWA-cache metadata from the
        # forward context (via its conv_owner prefix); no conv_meta to thread.
        return self.transformer_block(positions, hidden)


class InklingMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config: InklingModelConfig = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        self.config = config
        # The checkpoint ships num_nextn_predict_layers depth blocks, but only
        # the first ``num_speculative_tokens`` are exercised (step i uses depth
        # i). Build only those to save memory — each depth is a full Inkling block
        # with its own (large) full-history sconv caches and KV cache.
        n_predict = config.num_nextn_predict_layers
        num_spec = vllm_config.speculative_config.num_speculative_tokens
        self.num_mtp_layers = _select_mtp_depth_count(n_predict, num_spec)
        self.chain_hidden_post_norm = config.chain_hidden_post_norm

        # Depth blocks whose attention is sliding-window (swa_* head config)
        # rather than full; keyed by MTP depth via the checkpoint's
        # mtp_config.local_layer_ids (promoted onto the draft config). Mirrors
        # InklingModel's local_ids split, but over MTP depths, not backbone
        # layers.
        local_ids = set(config.local_layer_ids)

        # Keyed by depth index (str) to mirror the checkpoint layout.
        self.layers = nn.ModuleDict(
            {
                str(idx): InklingMTPDepthLayer(
                    config,
                    f"{prefix}.layers.{idx}",
                    idx in local_ids,
                    quant_config=vllm_config.quant_config,
                )
                for idx in range(self.num_mtp_layers)
            }
        )
        self.chain_norm = (
            InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.chain_hidden_post_norm
            else None
        )
        # The target's raw token embedding (pre embed_norm), attached by
        # load_eagle_model. Never materialized here: building our own
        # replicated copy would transiently double the 2.3 GiB table.
        self.embed_tokens: InklingReplicatedEmbedding = None  # type: ignore[assignment]
        # The depth layers consume the *backbone-normed* embedding
        # (embed_norm(embed(ids))), not the raw one: mtp embed_norm weights
        # are near-identity (trained on already-normalized inputs), and
        # feeding raw embeddings drops MTP1 acceptance from ~0.85 to ~0.70.
        # Weight loaded from the target's embed_norm.weight; gated like the
        # target's InklingModel.embed_norm.
        self.backbone_embed_norm = (
            InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: object | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draft-prefill embedding: fused gather + backbone embed_norm, then
        the target's tower embeddings scattered in unnormed (the backbone
        convention — MM embeds are merged after embed_norm)."""
        norm = self.backbone_embed_norm
        embeds = embed_rmsnorm(
            input_ids,
            self.embed_tokens.weight,
            norm.weight if norm is not None else None,
            norm.variance_epsilon if norm is not None else 0.0,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:  # type: ignore[arg-type]
            return embeds
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def fused_input_cat(
        self,
        layer: InklingMTPDepthLayer,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """The depth layer's [rmsnorm(hidden) | embed_norm(embed)] input in one
        launch: embedding row gather + the backbone embed_norm + the depth
        embed_norm chain on one side, hidden_norm on the other, written
        straight into the cat buffer."""
        hidden_w = layer.hidden_norm.weight
        embed_w = layer.embed_norm.weight
        eps = layer.hidden_norm.variance_epsilon
        if inputs_embeds is not None:
            # Draft prefill with target-merged MM embeddings (already
            # backbone-normed via embed_input_ids); only the depth embed_norm
            # remains.
            return embed_dual_rmsnorm_cat(
                previous_hidden, hidden_w, embed_w, eps, embeds=inputs_embeds
            )
        return embed_dual_rmsnorm_cat(
            previous_hidden,
            hidden_w,
            embed_w,
            eps,
            input_ids=input_ids,
            embed_table=self.embed_tokens.weight,
            pre_norm_weight=(
                self.backbone_embed_norm.weight
                if self.backbone_embed_norm is not None
                else None
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # The draft's short conv is a paged SWA-cache layer (its conv_owner is
        # auto-enumerated as a draft attention layer); its per-token metadata is
        # built by the speculator's build_attn_metadata and read from the
        # forward context, so nothing extra is threaded here.
        depth = spec_step_idx % self.num_mtp_layers
        layer = self.layers[str(depth)]
        combined = self.fused_input_cat(
            layer, previous_hidden_states, input_ids, inputs_embeds
        )
        hidden = layer(combined, positions)
        if self.chain_norm is not None:
            hidden = self.chain_norm(hidden)
        return hidden


class InklingMTP(nn.Module, SupportsMultiModalEmbeddings):
    packed_modules_mapping = _TmlForCausalLMBase.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config: InklingModelConfig = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        self.config = config
        self.model = InklingMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.checkpoint_transforms = [partial(inkling_expert_weights, self)]
        # The target's (vocab-sharded) LM head, attached by load_eagle_model;
        # never materialized here (same reasoning as model.embed_tokens).
        self.lm_head: ParallelLMHead = None  # type: ignore[assignment]
        self.logits_processor = LogitsProcessor(
            config.padded_vocab_size,
            org_vocab_size=config.vocab_size,
            soft_cap=config.final_logit_softcapping,
        )
        self._logits_zero: torch.Tensor | None = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: object | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        # The MTP shares the base model's LM head, which is trained on
        # ``hidden / mup``-scaled inputs, so apply the same mup scaling here
        # for a matching logit scale — folded into the lm_head GEMM alpha
        # (fp32 epilogue) like the target's compute_logits. (Argmax-invariant
        # for greedy draft sampling, but it matters for the gumbel sampling
        # distribution at temperature > 0.)
        mup = self.config.logits_mup_width_multiplier
        if not mup:
            return self.logits_processor(self.lm_head, hidden_states)
        assert self.logits_processor.soft_cap is None
        assert self.logits_processor.scale == 1.0
        w = self.lm_head.weight
        if self._logits_zero is None:
            self._logits_zero = w.new_zeros(1)
        logits = torch.addmm(
            self._logits_zero,
            hidden_states,
            w.t(),
            beta=0.0,
            alpha=1.0 / mup,
        )
        logits = self.logits_processor._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.logits_processor.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Greedy draft tokens via rank-local argmax + tiny (value, index)
        reduction — no full-vocab logits all-gather. The muP divisor is a
        positive scalar, so the argmax is invariant and the scaling is
        skipped entirely."""
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)


EntryClass = [InklingMTP]
