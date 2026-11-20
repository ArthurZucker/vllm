# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_mtp.py

import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .deepseek_v2 import DeepseekV2DecoderLayer
from .longcat_flash import scale_mla_lora_norms_on_load
from .utils import maybe_prefix


class LongCatMultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="eh_proj",
        )
        self.mtp_block = DeepseekV2DecoderLayer(vllm_config, prefix)
        scale_mla_lora_norms_on_load(self.mtp_block.self_attn, config)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states, _ = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class LongCatMultiTokenPredictor(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        # TODO: Use `num_hidden_layers` once Transformers 5.18.0 is the minimum
        self.mtp_start_layer_idx = config.num_layers * 2
        self.num_mtp_layers = 1
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): LongCatMultiTokenPredictorLayer(
                    config,
                    prefix=f"{prefix}.layers.{idx}",
                    vllm_config=vllm_config,
                    quant_config=quant_config,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

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
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )


class LongCatFlashMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # LongCat MTP has no MoE layers: clear n_routed_experts so the predictor
        # builds a dense MLP. object.__setattr__ bypasses the ngram remote
        # config's strict validation (it rejects setting the int field to None).
        object.__setattr__(vllm_config.model_config.hf_config, "n_routed_experts", None)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = (
            None
            if "mtp" in getattr(self.config, "disable_quant_module", [])
            else vllm_config.quant_config
        )

        self.model = LongCatMultiTokenPredictor(
            vllm_config=vllm_config,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
