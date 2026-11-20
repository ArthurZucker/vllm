# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GraniteMoeHybrid model."""

# Added by the IBM Team, 2025
from itertools import islice

import torch
from torch import nn
from transformers import GraniteMoeHybridConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .granitemoe import GraniteMoeMoE
from .granitemoeshared import GraniteMoeSharedMLP
from .interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
from .utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class GraniteMoeHybridMambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier

        self.mamba = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=config.mamba_expand * config.hidden_size,
            use_conv_bias=config.mamba_conv_bias,
            use_bias=config.mamba_proj_bias,
            n_groups=config.mamba_n_groups,
            num_heads=config.mamba_n_heads,
            head_dim=config.mamba_d_head,
            rms_norm_eps=config.rms_norm_eps,
            activation=config.hidden_act,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.block_sparse_moe = None
        if getattr(config, "num_local_experts", 0) > 0:
            self.block_sparse_moe = GraniteMoeMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )

        self.shared_mlp = (
            None
            if getattr(config, "shared_intermediate_size", 0) == 0
            else GraniteMoeSharedMLP(
                config, quant_config=quant_config, prefix=f"{prefix}.shared_mlp"
            )
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        output = self.mamba(hidden_states)
        hidden_states = residual + output * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            if self.block_sparse_moe is not None:
                hidden_states = self.block_sparse_moe(hidden_states)
            # else: skip
        else:
            # create a copy since block_sparse_moe modifies in-place
            if self.block_sparse_moe is not None:
                moe_hidden_states = hidden_states.clone()
                moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
                hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
                del moe_hidden_states
            else:
                hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states, residual


class GraniteMoeHybridAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier

        self.self_attn = GraniteMoeHybridAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.block_sparse_moe = None
        if getattr(config, "num_local_experts", 0) > 0:
            self.block_sparse_moe = GraniteMoeMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )

        self.shared_mlp = (
            None
            if getattr(config, "shared_intermediate_size", 0) == 0
            else GraniteMoeSharedMLP(
                config, quant_config=quant_config, prefix=f"{prefix}.shared_mlp"
            )
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
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            if self.block_sparse_moe is not None:
                hidden_states = self.block_sparse_moe(hidden_states)
            # else: skip
        else:
            # create a copy since block_sparse_moe modifies in-place
            if self.block_sparse_moe is not None:
                moe_hidden_states = hidden_states.clone()
                moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
                hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
                del moe_hidden_states
            else:
                hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states, residual


class GraniteMoeHybridAttention(nn.Module):
    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.causal = True
        self.hidden_size = config.hidden_size
        self.attention_bias = config.attention_bias
        self.attention_multiplier = config.attention_multiplier
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.total_num_kv_heads = config.num_key_value_heads

        # TensorParallel logic
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_key_value_heads = max(1, self.total_num_kv_heads // tp_size)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=self.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=self.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb: nn.Module | None
        if config.position_embedding_type == "rope":
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=True,
            )
        else:
            self.rotary_emb = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.attention_multiplier,
            num_kv_heads=self.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=-1,
        )

        if self.rotary_emb is not None:
            query, key = self.rotary_emb(positions, query, key)

        hidden_states = self.attn(query, key, value)
        del query, key, value

        hidden_states = self.o_proj(hidden_states)[0]
        return hidden_states


ALL_DECODER_LAYER_TYPES = {
    "full_attention": GraniteMoeHybridAttentionDecoderLayer,
    "linear_attention": GraniteMoeHybridMambaDecoderLayer,
}


@support_torch_compile
class GraniteMoeHybridModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )
        self.embedding_multiplier = config.embedding_multiplier

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES[config.layer_types[layer_idx]]
            return layer_class(
                config,
                layer_idx,
                model_config,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states *= self.embedding_multiplier
            residual = None
        else:
            if intermediate_tensors is None:
                raise RuntimeError("Intermediate tensors may not be None!")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions, hidden_states=hidden_states, residual=residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GraniteMoeHybridForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    SupportsQuant,
):
    packed_modules_mapping: dict[str, list[str]] = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "conv1d": ["conv1d"],
        "in_proj": ["in_proj"],
        "input_linear": ["input_linear"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache

        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        intermediate_size = hf_config.mamba_expand * hf_config.hidden_size

        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=hf_config.mamba_n_groups,
            num_heads=hf_config.mamba_n_heads,
            head_dim=hf_config.mamba_d_head,
            state_size=hf_config.mamba_d_state,
            conv_kernel=hf_config.mamba_d_conv,
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba2_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = GraniteMoeHybridModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            config.vocab_size,
            scale=1 / self.config.logits_scaling,
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
