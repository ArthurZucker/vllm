# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project



import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MixtureOfExperts,
    DeepseekV2MoE,
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    get_fused_shared_expert_layers,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.models.deepseek_v32.common.kernels import fused_eh_norm
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .model import DeepseekV32DecoderLayer


class DeepseekV32MultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = DeepseekV32DecoderLayer(
            vllm_config,
            prefix,
            config=config,
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
        eh_input = fused_eh_norm(
            positions,
            inputs_embeds,
            previous_hidden_states,
            self.enorm.weight,
            self.hnorm.weight,
            self.enorm.variance_epsilon,
        )
        hidden_states = self.eh_proj(eh_input)
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.shared_head.norm(hidden_states)
        return hidden_states, hidden_states


class DeepseekV32MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepseekV32MultiTokenPredictorLayer(
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
        for layer in self.layers.values():
            self_attn = getattr(layer.mtp_block, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "skip_topk"):
                self_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor):
        """Gather the top-k index rows at `slot_ids` to the front of the buffer."""
        num_slots = slot_ids.numel()
        for layer in self.layers.values():
            self_attn = getattr(layer.mtp_block, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "topk_indices_buffer"):
                topk_indices_buffer = self_attn.topk_indices_buffer
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
        return self.logits_processor(mtp_layer.shared_head.head, hidden_states)


class DeepseekV32MTP(nn.Module, DeepseekV2MixtureOfExperts, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepseekV32MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )
        self.fused_shared_expert_layers = get_fused_shared_expert_layers(
            self,
            DeepseekV2MoE,
            enable_eplb=vllm_config.parallel_config.enable_eplb,
        )

    def set_moe_parameters(self):
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            mlp = layer.mtp_block.mlp
            if isinstance(mlp, DeepseekV2MoE):
                example_moe = mlp
                self.moe_mlp_layers.append(mlp)
                self.moe_layers.append(mlp.experts)
        self.extract_moe_parameters(example_moe)

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
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        return (
            get_spec_layer_idx_from_weight_name(self.config, name) is None
            and "embed_tokens" not in name
        )

