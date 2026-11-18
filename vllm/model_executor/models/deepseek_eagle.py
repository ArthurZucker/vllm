# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV3ForCausalLM,
)

from .utils import maybe_prefix


@support_torch_compile
class DeepseekV2Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config
        self.vocab_size = self.config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                    config=self.config,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )

        self.fc = nn.Linear(
            self.config.model.hidden_size * 2,
            self.config.model.hidden_size,
            bias=False,
        )

        self.enorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.hnorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)

        inputs = torch.cat(
            [self.enorm(input_embeds), self.hnorm(hidden_states)], dim=-1
        )
        hidden_states = self.fc(inputs)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, hidden_states



class EagleDeepseekV3ForCausalLM(DeepseekV3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config
        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = DeepseekV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

        # Set MoE hyperparameters
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support multimodal inputs yet."
            )
        return self.model(input_ids, positions, hidden_states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

