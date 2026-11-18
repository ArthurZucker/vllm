# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ColBERT late interaction model for retrieval and reranking.

ColBERT uses per-token embeddings and late interaction (MaxSim) scoring
instead of single-vector representations or cross-encoder concatenation.

This module provides:

- `ColBERTMixin` — mixin that adds ColBERT late-interaction support
  to any embedding model.
- `ColBERTModel` — ColBERT with BERT backbone (original architecture).
- `ColBERTModernBertModel` — ColBERT with ModernBERT backbone.
- `ColBERTJinaRobertaModel` — ColBERT with Jina XLM-RoBERTa backbone.

Reference: https://arxiv.org/abs/2004.12832
"""

import re

import torch
from torch import nn

from vllm.config import PoolerConfig, VllmConfig
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed

from .bert import BertEmbeddingModel, BertModel
from .interfaces import HasInnerState, IsHybrid, SupportsLateInteraction
from .interfaces_base import default_pooling_type
from .lfm2 import Lfm2ForCausalLM, Lfm2Model


def _checkpoint_colbert_dim() -> int | None:
    """The projection's size, when only its checkpoint tensor has it"""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.model_executor.model_loader.conversion_mapping import COLBERT_LINEAR
    from vllm.transformers_utils.config import get_safetensors_params_metadata

    config = get_current_vllm_config_or_none()
    if config is None:
        return None
    model = config.model_config
    pattern = COLBERT_LINEAR._original_source_patterns[0]
    metadata = get_safetensors_params_metadata(model.model, revision=model.revision)
    shapes = [
        info["shape"] for name, info in metadata.items() if re.match(pattern, name)
    ]
    return shapes[0][0] if shapes else None


class ColBERTMixin(nn.Module, SupportsLateInteraction):
    """Mixin that adds ColBERT late interaction support to any embedding model.

    ColBERT (Contextualized Late Interaction over BERT) uses per-token
    embeddings with a linear projection layer.  This mixin provides:

    - the ColBERT linear projection, sized by the config or the checkpoint
    - a builder for the token-embedding pooler

    **Integration:**

    1. Inherit from both `ColBERTMixin` and `nn.Module`.
    2. In `__init__`: call `super().__init__()`, then
       `_init_colbert_components`, then create `self.model`
       (the backbone) and `self.pooler` via `_build_colbert_pooler`.
    The checkpoint's names are in `conversion_mapping.py` (`COLBERT_LINEAR`).
    """

    # Set during _init_colbert_components
    colbert_dim: int | None
    colbert_linear: nn.Linear | None
    hidden_size: int
    head_dtype: torch.dtype

    # ------------------------------------------------------------------ init

    def _init_colbert_components(
        self,
        hidden_size: int,
        colbert_dim: int | None,
        head_dtype: torch.dtype,
    ) -> None:
        """Initialise ColBERT projection layer.

        Args:
            hidden_size: Hidden dimension of the encoder backbone.
            colbert_dim: Output dimension for ColBERT embeddings.  If
                `None`, will be inferred from weights during loading (or
                auto-loaded from sentence-transformers Dense layers).
            head_dtype: Data type for the projection layer.

        """
        self.hidden_size = hidden_size
        self.colbert_dim = colbert_dim or _checkpoint_colbert_dim()
        self.head_dtype = head_dtype

        if self.colbert_dim is not None:
            self.colbert_linear = self._build_colbert_linear()
        else:
            self.colbert_linear = None

    def _build_colbert_linear(self) -> nn.Linear:
        """Build the ColBERT linear projection layer."""
        if self.colbert_dim is None:
            raise ValueError("colbert_dim must be set before building the linear layer")
        return nn.Linear(
            self.hidden_size,
            self.colbert_dim,
            bias=False,
            dtype=self.head_dtype,
        )

    # ---------------------------------------------------------------- pooler

    def _build_colbert_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        """Build pooler for ColBERT token embeddings.

        When `colbert_linear` is set, it is used as the projector.
        Otherwise `pooler_for_token_embed` falls back to auto-loading
        sentence-transformers Dense layers (`1_Dense/` etc.).
        """
        pooler = pooler_for_token_embed(
            pooler_config,
            projector=self.colbert_linear,
        )
        # a Sentence-Transformers projector (`1_Dense/`) is loaded as it is built
        projector = getattr(getattr(pooler, "head", None), "projector", None)
        if projector is not None and projector is not self.colbert_linear:
            for param in projector.parameters():
                param.loaded_at_init = True
        return pooler

    # --------------------------------------------------------- config helper

    @classmethod
    def get_colbert_dim_from_config(cls, hf_config) -> int | None:
        """Extract ColBERT dimension from a HuggingFace config.

        Checks `colbert_dim`, `dim` and `projection_dim` in that order.
        """
        return (
            getattr(hf_config, "colbert_dim", None)
            or getattr(hf_config, "dim", None)
            or getattr(hf_config, "projection_dim", None)
        )

    # -------------------------------------------------------- weight loading


# -----------------------------------------------------------------------
# Concrete model: ColBERT + BERT backbone  (original architecture)
# -----------------------------------------------------------------------


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColBERTModel(ColBERTMixin, BertEmbeddingModel):
    """ColBERT late interaction model with BERT backbone.

    Supports the `token_embed` task (per-token embeddings for late
    interaction).  MaxSim scoring is computed externally.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config

        # Must run before super().__init__ because _build_pooler reads these.
        colbert_dim = self.get_colbert_dim_from_config(config)
        self._init_colbert_components(
            hidden_size=config.hidden_size,
            colbert_dim=colbert_dim,
            head_dtype=vllm_config.model_config.head_dtype,
        )

        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config, prefix=prefix)

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        return self._build_colbert_pooler(pooler_config)


# -----------------------------------------------------------------------
# Concrete model: ColBERT + ModernBERT backbone
# -----------------------------------------------------------------------

from .modernbert import ModernBertModel  # noqa: E402


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColBERTModernBertModel(ColBERTMixin, nn.Module):
    """ColBERT late interaction model with ModernBERT backbone.

    For `lightonai/GTE-ModernColBERT-v1` and similar models.
    The projection is auto-loaded from sentence-transformers `1_Dense/`
    when not present in the main checkpoint.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        colbert_dim = self.get_colbert_dim_from_config(config)
        self._init_colbert_components(
            hidden_size=config.hidden_size,
            colbert_dim=colbert_dim,
            head_dtype=vllm_config.model_config.head_dtype,
        )

        self.model = ModernBertModel(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = self._build_colbert_pooler(pooler_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )


# -----------------------------------------------------------------------
# Concrete model: ColBERT + Jina XLM-RoBERTa backbone
# -----------------------------------------------------------------------

from .bert_with_rope import JinaRobertaModel  # noqa: E402


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColBERTJinaRobertaModel(ColBERTMixin, nn.Module):
    """ColBERT late interaction model with Jina XLM-RoBERTa backbone.

    For `jinaai/jina-colbert-v2` and similar models.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        colbert_dim = self.get_colbert_dim_from_config(config)
        self._init_colbert_components(
            hidden_size=config.hidden_size,
            colbert_dim=colbert_dim,
            head_dtype=vllm_config.model_config.head_dtype,
        )

        self.model = JinaRobertaModel(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = self._build_colbert_pooler(pooler_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )


# -----------------------------------------------------------------------
# Concrete model: ColBERT + LFM2 backbone
# -----------------------------------------------------------------------


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColBERTLfm2Model(ColBERTMixin, nn.Module, HasInnerState, IsHybrid):
    """ColBERT late interaction model with LFM2 backbone.

    For `LiquidAI/LFM2-ColBERT-350M` and similar models.

    The projection is auto-loaded from sentence-transformers `1_Dense/`
    when not present in the main checkpoint.
    """

    is_pooling_model = True
    # LFM2 is a hybrid model (attention + SSM layers); these flags ensure
    # HybridAttentionMambaModelConfig.verify_and_update_config runs so that
    # mamba_block_size and related cache settings are correctly initialised.
    is_hybrid = True
    has_inner_state = True

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return Lfm2ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return Lfm2ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Lfm2ForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        colbert_dim = self.get_colbert_dim_from_config(config)
        self._init_colbert_components(
            hidden_size=config.hidden_size,
            colbert_dim=colbert_dim,
            head_dtype=vllm_config.model_config.head_dtype,
        )

        self.model = Lfm2Model(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = self._build_colbert_pooler(pooler_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
