# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 MTP (multi-token prediction) draft model -- ROCm/AMD variant.

Byte-identical to `nvidia/mtp.py` except this file lives under `amd/` so its
`from .model import ...` resolves to the self-contained AMD model (native Gemma
RMSNorm, native MXFP8 MoE, Triton sparse attention). The MTP logic is
platform-agnostic. (Mirrors `vllm.models.deepseek_v4.amd.mtp`.)

TODO(future, separate diff): since this is byte-identical to `nvidia/mtp.py`,
both copies could be consolidated into a single `common/mtp.py` that dispatches
its model import (`..amd.model` vs `..nvidia.model`) via
`current_platform.is_rocm()` -- the same dispatch `minimax_m3/__init__.py`
uses. This was prototyped and VERIFIED working (`MiniMaxM3MTP` resolves through
`common.mtp` to the AMD decoder layer / RMSNorm on ROCm), but it deletes the
upstream `nvidia/mtp.py` and touches the NVIDIA load path, so it is deferred to
a dedicated refactor diff to keep this AMD-enablement change NVIDIA-untouched.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    in_single_pp_stage,
)
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    maybe_prefix,
)
from vllm.models.minimax_m3.amd.indexer_msa import (
    MiniMaxM3IndexerMSAImpl,
    msa_indexer_cp_enabled,
    select_msa_indexer_impl_cls,
)
from vllm.models.minimax_m3.amd.ops.sparse_pa import ASM_PAGE_SIZE
from vllm.models.minimax_m3.common.sparse_attention import (
    minimax_m3_use_aiter_sparse_pa,
)
from vllm.sequence import IntermediateTensors

from .model import (
    MiniMAXGemmaRMSNorm,
    MiniMaxM3DecoderLayer,
)


class _SparseIndexerContext(NamedTuple):
    topk_indices_buffer: torch.Tensor | None = None
    msa_indexer_impl_cls: type[MiniMaxM3IndexerMSAImpl] | None = None
    sparse_bt_buffer: torch.Tensor | None = None
    sparse_ctx_buffer: torch.Tensor | None = None
    indexer_cp: bool = False


def _build_sparse_indexer_context(
    config: PreTrainedConfig, vllm_config: VllmConfig
) -> _SparseIndexerContext:
    # Reserved top-k indices buffer shared by all the sparse-attention indexer
    # layers below; the indexer writes its per-head decode/prefill block
    # selection into it, the attend reads it back.
    tp_size = get_tensor_model_parallel_world_size()
    num_index_heads = max(1, config.index_n_heads // tp_size)
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    topk_indices_buffer = torch.empty(
        num_index_heads,
        max_tokens,
        config.index_topk_blocks,
        dtype=torch.int32,
    )

    num_kv_heads = max(1, config.num_key_value_heads // tp_size)
    indexer_kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("bf16")

    impl_cls = select_msa_indexer_impl_cls(
        topk_blocks=config.index_topk_blocks,
        sparse_block_size=config.index_block_size,
        index_head_dim=config.index_head_dim,
        indexer_kv_dtype=indexer_kv_dtype,
    )
    msa_indexer_impl_cls = None
    sparse_bt_buffer = None
    sparse_ctx_buffer = None
    if impl_cls is not None and minimax_m3_use_aiter_sparse_pa(
        num_kv_heads, emits_sparse_block_table=True
    ):
        # One row per (token, kv head): the table addresses the attend's cache,
        # not the index cache.
        rows = max_tokens * num_kv_heads
        msa_indexer_impl_cls = impl_cls
        sparse_bt_buffer = torch.empty(
            rows,
            config.index_topk_blocks * (config.index_block_size // ASM_PAGE_SIZE),
            dtype=torch.int32,
        )
        sparse_ctx_buffer = torch.empty(rows, dtype=torch.int32)

    indexer_cp = msa_indexer_cp_enabled(
        vllm_config, msa_indexer_selected=msa_indexer_impl_cls is not None
    )
    return _SparseIndexerContext(
        topk_indices_buffer,
        msa_indexer_impl_cls,
        sparse_bt_buffer,
        sparse_ctx_buffer,
        indexer_cp,
    )


class MiniMaxM3MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        indexer_ctx: _SparseIndexerContext,
    ) -> None:
        super().__init__()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.enorm = MiniMAXGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = MiniMAXGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.transformer_layer = MiniMaxM3DecoderLayer(
            config=config,
            prefix=prefix,
            cache_config=cache_config,
            quant_config=quant_config,
            force_sparse_attn=True,
            force_moe=True,
            topk_indices_buffer=indexer_ctx.topk_indices_buffer,
            msa_indexer_impl_cls=indexer_ctx.msa_indexer_impl_cls,
            sparse_bt_buffer=indexer_ctx.sparse_bt_buffer,
            sparse_ctx_buffer=indexer_ctx.sparse_ctx_buffer,
            indexer_cp=indexer_ctx.indexer_cp,
        )
        self.final_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # Mask out inputs at position 0, as not needed by MTP.
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)

        # Combine the normalized token embeddings with the normalized
        # previous hidden states.
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states, _ = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        # Apply transformer layer.
        hidden_states, residual = self.transformer_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )

        hidden_states += residual
        return hidden_states


class MiniMaxM3MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        assert vllm_config.speculative_config is not None
        # Use the draft (MTP) config, not the target model's. This is flat for a
        # standalone checkpoint, and the promoted text_config for a bundled one.
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.num_mtp_layers = config.num_mtp_modules
        indexer_ctx = _build_sparse_indexer_context(config, vllm_config)
        if indexer_ctx.msa_indexer_impl_cls is None:
            indexer_kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype(
                "bf16"
            )
            if indexer_kv_dtype != "bf16":
                from vllm.platforms.rocm import on_gfx942

                if not on_gfx942():
                    raise NotImplementedError(
                        f"MiniMax M3 MTP could not select the MSA indexer, leaving "
                        f"its draft layers on the bf16 indexer, so "
                        f"indexer_kv_dtype={indexer_kv_dtype!r} has nowhere to go. "
                        f"See the preceding 'MSA unavailable' log line for why. Drop "
                        f"indexer_kv_dtype from --attention-config, or speculate "
                        f"with an EAGLE3 drafter instead."
                    )
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): MiniMaxM3MultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}", indexer_ctx
                )
                for idx in range(self.num_mtp_layers)
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )


class MiniMaxM3MTP(nn.Module):
    # Shares the target's input embedding unless loading finds its own (only
    # tracked under V2 PP, where the target's may not be available).
    has_own_embed_tokens = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self._tracks_own_embed = in_single_pp_stage()
        if self._tracks_own_embed:
            # Preserve embeddings from loaders that bypass load_weights.
            self.has_own_embed_tokens = vllm_config.load_config.load_format != "dummy"

        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = MiniMaxM3MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        current_step_idx = spec_step_idx % self.model.num_mtp_layers
        mtp_layer = self.model.layers[str(current_step_idx)]
        return self.logits_processor(
            self.lm_head, mtp_layer.final_layernorm(hidden_states)
        )

    def _map_checkpoint_name(self, name: str) -> str | None:
        """Map a full checkpoint key to this MTP module's parameter name.

        The MTP module only owns the *.mtp.layers.* weights plus the token
        embedding and LM head, which the checkpoint shares with the main model.
        Everything else belongs to other modules and is ignored here by returning
        None.
        """
        # In the bundled checkpoint, the MTP weights are prefixed with
        # "language_model". The standalone MTP checkpoint has no such prefix.
        # Strip it if present.
        name = name.removeprefix("language_model.")

        if name == "model.embed_tokens.weight":
            return "model.embed_tokens.weight"
        if name == "lm_head.weight":
            return "lm_head.weight"
        if "model.mtp.layers" in name:
            if "weight_scale_inv" in name:
                # The checkpoint stores block scales as "weight_scale_inv".
                # The ModelOpt MXFP8 layers expose them as "weight_scale".
                name = name.replace("weight_scale_inv", "weight_scale")
            # Strip "mtp" from prefix.
            return name.replace(".mtp.", ".")
        return None

