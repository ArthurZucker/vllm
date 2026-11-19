# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import in_single_pp_stage
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.reload import weight_update_active
from vllm.model_executor.models.utils import (
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .model import (
    MiniMAXGemmaRMSNorm,
    MiniMaxM3DecoderLayer,
)

EMBED = "model.embed_tokens.weight"


class MiniMaxM3MultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
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
            vllm_config=vllm_config,
            prefix=prefix,
            force_sparse_attn=True,
            force_moe=True,
            is_mtp_block=True,
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
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): MiniMaxM3MultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
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
            self.checkpoint_transforms = [self._track_own_embed]

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

    def _track_own_embed(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """The stream as is; whether it ships the input embedding decides
        `has_own_embed_tokens` (a partial weight update keeps the last answer)"""
        own = False
        for name, weight in weights:
            own = own or name.removeprefix("language_model.") == EMBED
            yield name, weight
        if own:
            self.has_own_embed_tokens = True
        elif not weight_update_active():
            # A complete checkpoint without embeddings borrows the target's.
            self.has_own_embed_tokens = False

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
