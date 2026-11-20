# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
# All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import torch
from torch import nn
from transformers import Llama4TextConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import (
    Attention,
    ChunkedLocalAttention,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import sequence_parallel_chunk

from .llama import LlamaForCausalLM, LlamaMLP, LlamaModel
from .utils import (
    PPMissingLayer,
    extract_layer_index,
    fast_topk,
)

logger = init_logger(__name__)


class Llama4MoE(nn.Module):
    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_scores, router_indices = fast_topk(gating_output, topk, dim=-1)
        # pseudo-standard is that the router scores are floats
        router_scores = torch.sigmoid(router_scores.float())
        return (router_scores, router_indices.to(torch.int32))

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()

        intermediate_size_moe = config.intermediate_size
        self.router = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            prefix=f"{prefix}.router",
        )

        self.shared_expert = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            hidden_act="silu",
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.shared_expert",
            reduce_results=False,
            disable_tp=self.is_sequence_parallel,
        )

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config if parallel_config else None
        self.enable_eplb = parallel_config.enable_eplb if parallel_config else False
        self.n_redundant_experts = (
            eplb_config.num_redundant_experts if eplb_config else 0
        )

        self.n_routed_experts: int = config.num_local_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_shared_experts: int = 1
        self.n_local_experts: int = config.num_local_experts
        self.n_physical_experts = self.n_local_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_expert,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            custom_routing_function=Llama4MoE.custom_routing_function,
            intermediate_size=intermediate_size_moe,
            apply_router_weight_on_input=True,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            is_sequence_parallel=self.is_sequence_parallel,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
        )

    def forward(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.router(hidden_states)

        experts_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        if self.is_sequence_parallel:
            experts_out = tensor_model_parallel_all_gather(experts_out, 0)
            experts_out = experts_out[:num_tokens]

        return experts_out


class Llama4Attention(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        self.no_rope_layers = config.no_rope_layers
        self.nope = self.no_rope_layers[self.layer_idx] == 0
        self.use_qk_norm = config.use_qk_norm and not self.nope
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_temperature_tuning = self.nope and config.attn_temperature_tuning

        self.floor_scale = getattr(config, "floor_scale", 8192.0)
        self.attn_scale = getattr(config, "attn_scale", 0.1)
        self.max_position_embeddings = max_position_embeddings
        self.n_rep = self.num_heads // self.num_kv_heads
        self.qk_norm = (
            RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,
                dtype=torch.float32,
            )
            if self.use_qk_norm
            else None
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        is_neox_style = True

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=is_neox_style,
            )
            if not self.nope
            else None
        )

        use_chunked_local_attn = not self.nope and config.attention_chunk_size
        attn_cls = ChunkedLocalAttention if use_chunked_local_attn else Attention
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **(
                {"attention_chunk_size": config.attention_chunk_size}
                if use_chunked_local_attn
                else {}
            ),
        )

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0

        return attn_scale.unsqueeze(-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        if self.qk_norm is not None:
            # Normalization is applied on the head_dim dimension. The rest of
            # the dimensions are collapsed into a single dimension to support
            # custom rms_norm cuda kernel.
            q = q.reshape(-1, self.head_dim)
            q = self.qk_norm(q.float()).reshape(-1, self.q_size).to(q.dtype)
            k = k.reshape(-1, self.head_dim)
            k = self.qk_norm(k.float()).reshape(-1, self.kv_size).to(k.dtype)

        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399)
        # to NoPE layers, where the inference-time temperature tuning function
        # is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        #
        # We should apply temperature tuning between (after) rotary / QK norm
        # and (before) attention.
        if self.attn_temperature_tuning and self.nope:
            attn_scale = self._get_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Llama4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: Llama4TextConfig | None = None,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_idx = extract_layer_index(prefix)
        self.global_layer = config.no_rope_layers[self.layer_idx] == 0
        self.hidden_size = config.hidden_size
        max_position_embeddings = config.max_position_embeddings

        self.self_attn = Llama4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        is_moe_layer = (
            config.interleave_moe_layer_step > 0
            and (self.layer_idx + 1) % config.interleave_moe_layer_step == 0
        )
        if is_moe_layer:
            self.feed_forward = Llama4MoE(
                vllm_config=vllm_config,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                hidden_act="silu",
                quant_config=quant_config,
                bias=False,
                prefix=f"{prefix}.feed_forward",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Llama4Model(LlamaModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[Llama4DecoderLayer] = Llama4DecoderLayer,
    ):
        self.num_experts = vllm_config.model_config.hf_config.num_local_experts
        self.n_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)


class Llama4ForCausalLM(LlamaForCausalLM, MixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # update temperature tuning config from generation config
        gen_config = vllm_config.model_config.try_get_generation_config()
        gen_config.update(vllm_config.model_config.override_generation_config)
        # enable temperature tuning by default when max_model_len > 32K
        default_attn_temperature_tuning = vllm_config.model_config.max_model_len > 32768
        vllm_config.model_config.hf_config.attn_temperature_tuning = gen_config.get(
            "attn_temperature_tuning", default_attn_temperature_tuning
        )

        super().__init__(
            vllm_config=vllm_config, prefix=prefix, layer_type=Llama4DecoderLayer
        )
        # Set MoE hyperparameters
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Llama4DecoderLayer)
            if isinstance(layer.feed_forward, Llama4MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.feed_forward
                self.moe_layers.append(layer.feed_forward.experts)

        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning("No Llama4MoE layer found in model.layers.")
        else:
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.feed_forward, Llama4MoE):
                moe = layer.feed_forward
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[Llama4DecoderLayer] = Llama4DecoderLayer,
    ):
        return Llama4Model(
            vllm_config=vllm_config, prefix=prefix, layer_type=layer_type
        )
