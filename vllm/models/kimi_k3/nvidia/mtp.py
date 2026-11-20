# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Kimi-K3 Multi-Token-Prediction (MTP) draft model."""

import copy

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from ..common.mtp import fused_mtp_input
from .low_latency_gemm import enable_kimi_k3_low_latency_gemm
from .model import (
    KimiDecoderLayer,
)

logger = init_logger(__name__)


class SharedHead(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class KimiK3MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "eh_proj"),
            return_bias=False,
        )

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        # The MTP block starts without the base model's AttnRes state.
        block_config = copy.copy(config)
        block_config.attn_res_block_size = None
        # NOTE: the prefix must end in the numeric spec-layer index so that
        # KimiDecoderLayer can parse ``layer_idx`` and pick MLA (full attn).
        # Own aux stream for the MLA g_proj output-gate overlap (DeepseekV4
        # convention: the MTP block creates its own stream).
        aux_stream = torch.cuda.Stream()
        self.mtp_block = KimiDecoderLayer(
            block_config, vllm_config, prefix=prefix, aux_stream=aux_stream
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs_embeds is not None
        hidden_states = self.eh_proj(
            fused_mtp_input(
                positions,
                inputs_embeds,
                previous_hidden_states,
                self.enorm.weight,
                self.hnorm.weight,
                self.enorm.variance_epsilon,
            )
        )
        if self.mtp_block.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)

        hidden_states, _, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        if self.mtp_block.use_sequence_parallel:
            assert residual is not None
            hidden_states = hidden_states + residual
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]
            logits_hidden_states = self.shared_head.norm(hidden_states)
            return logits_hidden_states, hidden_states

        # Produce the normalized logits input and the pre-norm recurrent state
        # in one fused add-RMSNorm launch.
        logits_hidden_states, hidden_states = self.shared_head.norm(
            hidden_states, residual
        )
        return logits_hidden_states, hidden_states


class KimiK3MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: KimiLinearConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): KimiK3MultiTokenPredictorLayer(
                    config, vllm_config, f"{prefix}.layers.{idx}"
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
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(mtp_layer.shared_head.head, hidden_states)
        return logits


class KimiK3MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.model = KimiK3MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        enable_kimi_k3_low_latency_gemm(self, vllm_config.model_config.dtype)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return self.model.compute_logits(hidden_states, spec_step_idx)
