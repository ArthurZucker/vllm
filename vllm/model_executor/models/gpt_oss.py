# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import nn
from transformers import GptOssConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.utils import rocm_unquantized_gemm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    make_attention_sink,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    EagleModelMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

_GPT_OSS_STREAMED_EXPERT_SUFFIX_TO_SHARD = {
    "w13_weight": "gpt_oss_w13",
    "w2_weight": "gpt_oss_w2",
    "w13_bias": "gpt_oss_w13",
    "w2_bias": "gpt_oss_w2",
    "w13_weight_scale": "gpt_oss_w13",
    "w2_weight_scale": "gpt_oss_w2",
    "w13_weight_packed": "gpt_oss_w13",
    "w2_weight_packed": "gpt_oss_w2",
}


def _get_weight_loader(param: torch.Tensor) -> Callable[..., object]:
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    if not callable(weight_loader):
        raise TypeError("weight_loader must be callable")
    return weight_loader


class OAIAttention(nn.Module):
    # Override to switch RoPE convention. gpt-oss uses NeoX (chunk halves);
    # privacy-filter and similar derivatives use GPT-J (interleaved pairs).
    rope_is_neox_style: bool = True

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            dtype=torch.float32,
            rope_parameters={
                "rope_theta": config.rope_parameters["rope_theta"],
                "rope_type": "yarn",
                "factor": config.rope_parameters["factor"],
                "original_max_position_embeddings": config.rope_parameters[
                    "original_max_position_embeddings"
                ],
                "beta_fast": config.rope_parameters["beta_fast"],
                "beta_slow": config.rope_parameters["beta_slow"],
                "truncate": config.rope_parameters.get("truncate", True),
            },
            is_neox_style=self.rope_is_neox_style,
        )

        tp_size = get_tensor_model_parallel_world_size()

        self.sinks = make_attention_sink(config.num_attention_heads // tp_size)

        self.q_size = self.num_attention_heads * self.head_dim // tp_size
        self.kv_size = self.num_key_value_heads * self.head_dim // tp_size
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.num_local_attention_heads = config.num_attention_heads // tp_size
        self.num_local_key_value_heads = config.num_key_value_heads // tp_size

        self.attn = self._build_attention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _build_attention(
        self,
        config: GptOssConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> Attention:
        # Override to swap in an encoder-only attention or alter the
        # per-layer sliding-window policy.
        # Only apply sliding window to every other layer
        sliding_window = config.sliding_window if self.layer_idx % 2 == 0 else None
        return Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class GptOssRoutedExperts(RoutedExperts):
    """Load one GPT-OSS expert at a time without assembling the global stack."""

    @staticmethod
    def _narrow_for_rank(
        loaded_weight: torch.Tensor,
        dim: int,
        rank: int,
        size: int,
    ) -> torch.Tensor:
        start = rank * size
        available = loaded_weight.shape[dim] - start
        return loaded_weight.narrow(dim, start, min(size, max(available, 0)))

    @staticmethod
    def _copy_to_expert(expert_data: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if expert_data.numel() == loaded_weight.numel() == 1:
            expert_data.copy_(loaded_weight.reshape(()))
            return
        while loaded_weight.ndim > expert_data.ndim and loaded_weight.shape[-1] == 1:
            loaded_weight = loaded_weight.squeeze(-1)
        slices = tuple(slice(0, size) for size in loaded_weight.shape)
        expert_data[slices].copy_(loaded_weight)


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.is_sequence_parallel = (
            parallel_config.use_sequence_parallel_moe
            and vllm_config.lora_config is None
        )

        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.experts_per_token = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.router = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=True,
            prefix=f"{prefix}.router",
            return_bias=False,
        )
        assert config.intermediate_size % self.world_size == 0
        self.experts = FusedMoEFactory(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            ckpt_names=("w1", "w2", "w3"),
            apply_router_weight_on_input=False,
            has_bias=True,
            activation="swigluoai",
            is_sequence_parallel=self.is_sequence_parallel,
            routed_experts_cls=GptOssRoutedExperts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        if self.is_sequence_parallel:
            x = sequence_parallel_chunk(x)

        if current_platform.is_rocm():
            g = rocm_unquantized_gemm(
                self, x[:, : self.hidden_size], self.router.weight, self.router.bias
            )
        else:
            g = self.router(x)
        x = self.experts(hidden_states=x, router_logits=g)[:, : self.hidden_size]

        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x.contiguous(), 0)
            x = x[:num_tokens]
        return x


class TransformerBlock(torch.nn.Module):
    # Override to swap attention/MLP without re-implementing the block.
    attention_cls: type[nn.Module] = OAIAttention
    mlp_cls: type[nn.Module] = MLPBlock

    def __init__(
        self,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        self.layer_idx = extract_layer_index(prefix)
        self.attn = self.attention_cls(
            config,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
            cache_config=cache_config,
        )
        self.mlp = self.mlp_cls(vllm_config, self.layer_idx, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.attn(hidden_states, positions)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        output = self.mlp(hidden_states)
        return output, residual


@support_torch_compile
class GptOssModel(nn.Module, EagleModelMixin):
    # Override to swap in an alternative TransformerBlock subclass.
    block_cls: type[nn.Module] = TransformerBlock
    supports_aux_hidden_states_over_pp = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.parallel_config = vllm_config.parallel_config
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: self.block_cls(
                vllm_config,
                prefix=prefix,
                quant_config=self.quant_config,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)
        # its expert loading writes BF16 into the experts outside their loaders
        if any(getattr(m, "quantizes_on_load", False) for m in self.modules()):
            raise NotImplementedError(
                "gpt-oss does not support online quantization with "
                "VLLM_RELOAD_MODE=direct"
            )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                x = inputs_embeds
            else:
                x = self.embed_input_ids(input_ids)

            residual = None
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, x, residual
        )
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            x, residual = layer(x, positions, residual)
            self._maybe_add_hidden_state(aux_hidden_states, i + 1, x, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": x,
                    "residual": residual,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )
        x, _ = self.norm(x, residual)

        if len(aux_hidden_states) > 0:
            return x, aux_hidden_states
        return x

    @staticmethod
    def _get_streamed_expert_info(
        name: str,
        params_dict: dict[str, torch.nn.Parameter],
    ) -> tuple[int, str, str] | None:
        """Parse `...experts.<expert_id>.<fused_param>` checkpoint keys."""
        if ".mlp.experts." not in name:
            return None
        suffix = name.rsplit(".", 1)[-1]
        shard_id = _GPT_OSS_STREAMED_EXPERT_SUFFIX_TO_SHARD.get(suffix)
        if shard_id is None:
            return None

        prefix, expert_suffix = name.split(".mlp.experts.", maxsplit=1)
        expert_id_str, separator, param_suffix = expert_suffix.partition(".")
        if not separator or not expert_id_str.isdigit():
            return None
        expert_id = int(expert_id_str)
        for base_layer_prefix in ("", "base_layer."):
            fused_name = (
                f"{prefix}.mlp.experts.{base_layer_prefix}routed_experts.{param_suffix}"
            )
            if fused_name in params_dict:
                return expert_id, fused_name, shard_id
        return None


class GptOssForCausalLM(
    nn.Module, SupportsPP, SupportsEagle, SupportsEagle3, SupportsLoRA
):
    is_3d_moe_weight: typing.ClassVar[bool] = True
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        self.model = GptOssModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
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
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)
