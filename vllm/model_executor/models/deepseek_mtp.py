# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV2MoE,
)
from .interfaces import SupportsPP
from .utils import (
    get_fused_shared_expert_layers,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        *,
        defer_lm_head: bool = False,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if defer_lm_head:
            self.head = None
        else:
            self.head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "head"),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.device = current_platform.device_type

        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = DeepseekV2DecoderLayer(
            vllm_config,
            prefix,
            config=self.config,
            topk_indices_buffer=topk_indices_buffer,
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
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
        if self.mtp_block.use_sequence_parallel_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[: positions.shape[0]]
        # Recycle the post-final-norm hidden into the next draft step.
        # compute_logits applies shared_head (== final norm) to the pre-norm
        # element, so logits and the recycle each get exactly one final-norm.
        # Matches SGLang's deepseek_nextn.
        return hidden_states, self.shared_head(hidden_states)


class DeepSeekMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
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

    def set_skip_topk(self, skip: bool):
        """Toggle skip_topk on all MTP layers with sparse attention.

        Called by the proposer to implement index_share_for_mtp_iteration:
        step 0 sets skip=False (compute own indices), steps 1+ set skip=True
        (reuse step 0's indices).
        """
        for layer in self.layers.values():
            mtp_block = getattr(layer, "mtp_block", None)
            if mtp_block is not None:
                self_attn = getattr(mtp_block, "self_attn", None)
                if self_attn is not None:
                    mla_attn = getattr(self_attn, "mla_attn", None)
                    if mla_attn is not None and hasattr(mla_attn, "skip_topk"):
                        mla_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor):
        """Gather the top-k index rows at `slot_ids` to the front of the buffer."""
        num_slots = slot_ids.numel()
        for layer in self.layers.values():
            mtp_block = getattr(layer, "mtp_block", None)
            if mtp_block is not None:
                self_attn = getattr(mtp_block, "self_attn", None)
                if self_attn is not None:
                    mla_attn = getattr(self_attn, "mla_attn", None)
                    if mla_attn is not None and hasattr(
                        mla_attn, "topk_indices_buffer"
                    ):
                        topk_indices_buffer = mla_attn.topk_indices_buffer
                        topk_indices_buffer[:num_slots] = topk_indices_buffer[slot_ids]

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
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits


@support_torch_compile
class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.vllm_config = vllm_config
        self.model = DeepSeekMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Set MoE hyperparameters
        self.set_moe_parameters()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def set_moe_parameters(self):
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            assert isinstance(layer, DeepSeekMultiTokenPredictorLayer)
            layer = layer.mtp_block
            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)
        self.fused_shared_expert_layers = get_fused_shared_expert_layers(
            self,
            DeepseekV2MoE,
            enable_eplb=self.vllm_config.parallel_config.enable_eplb,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)
