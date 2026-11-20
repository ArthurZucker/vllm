# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import islice

import torch
import torch.nn as nn
from transformers import DbrxConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class DbrxRouter(nn.Module):
    """A Router implementation for DBRX that returns logits for each expert
    per token.
    """

    def __init__(
        self,
        config: DbrxConfig,
        params_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.d_model = config.d_model
        self.layer = GateLinear(
            self.d_model,
            self.num_total_experts,
            params_dtype=params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.layer(hidden_states)
        return router_logits


class DbrxExperts(RoutedExperts):
    def __init__(
        self,
        *args,
        config: DbrxConfig,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.d_model = config.d_model
        self.tp_size = self.moe_config.tp_size
        self.intermediate_size = self.config.ffn_config.ffn_hidden_size // self.tp_size


class DbrxMoE(nn.Module):
    """A tensor-parallel MoE implementation for DBRX.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: DbrxConfig,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.router = DbrxRouter(config, self.params_dtype)

        self.experts = FusedMoEFactory(
            num_experts=config.ffn_config.moe_num_experts,
            top_k=config.ffn_config.moe_top_k,
            hidden_size=config.d_model,
            intermediate_size=config.ffn_config.ffn_hidden_size,
            params_dtype=self.params_dtype,
            renormalize=True,
            quant_config=quant_config,
            tp_size=get_tensor_model_parallel_world_size(),
            prefix=prefix,
            routed_experts_cls=DbrxExperts,
            routed_experts_args={"config": config},
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.d_model)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.router(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


class DbrxAttention(nn.Module):
    def __init__(
        self,
        config: DbrxConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.total_num_kv_heads = config.attn_config.kv_n_heads
        self.clip_qkv = config.attn_config.clip_qkv
        rope_parameters = {
            "rope_type": "default",
            "rope_theta": int(config.attn_config.rope_theta),
        }
        self.max_position = config.max_seq_len

        # pylint: disable=invalid-name
        self.Wqkv = QKVParallelLinear(
            self.d_model,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.Wqkv",
        )
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_world_size
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size
        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v)
        hidden_states, _ = self.out_proj(attn_output)
        return hidden_states


class DbrxFusedNormAttention(nn.Module):
    def __init__(
        self,
        config: DbrxConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.attn = DbrxAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.attn"
        )
        self.norm_1 = nn.LayerNorm(self.d_model)
        self.norm_2 = nn.LayerNorm(self.d_model)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        x = self.attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
        )
        hidden_states = residual + x
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        return hidden_states, residual


class DbrxBlock(nn.Module):
    def __init__(
        self,
        config: DbrxConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm_attn_norm = DbrxFusedNormAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.norm_attn_norm"
        )
        self.ffn = DbrxMoE(config, quant_config, prefix=f"{prefix}.ffn")

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, residual = self.norm_attn_norm(
            position_ids=position_ids,
            hidden_states=hidden_states,
        )
        hidden_states = self.ffn(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class DbrxModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.quant_config = quant_config
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.start_layer, self.end_layer, self.blocks = make_layers(
            config.n_layers,
            lambda prefix: DbrxBlock(config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.blocks",
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=1e-5)
        for module in self.modules():
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                # Remove the bias term in Linear and LayerNorm.
                module.register_parameter("bias", None)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.d_model
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors
            hidden_states = intermediate_tensors["hidden_states"]
        for block in islice(self.blocks, self.start_layer, self.end_layer):
            hidden_states = block(position_ids, hidden_states)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class DbrxForCausalLM(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        if config.tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported for Dbrx models.")
        self.quant_config = quant_config

        self.transformer = DbrxModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "transformer")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.transformer(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
