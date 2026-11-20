# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Jurassic model."""

from typing import Any

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SwigluStepAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    MoERunner,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .interfaces import MixtureOfExperts, SupportsPP
from .utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _get_layer_type(config: PreTrainedConfig, layer_idx: int) -> str:
    # Upstream configs move the trailing MTP layers to `mtp_layer_types`
    mtp_layer_types = getattr(config, "mtp_layer_types", None) or []
    return [*config.layer_types, *mtp_layer_types][layer_idx]


def _get_num_attention_heads(
    config: PreTrainedConfig, layer_idx: int, layer_type: str
) -> int:
    if layer_type == "sliding_attention":
        if num_heads := getattr(config, "num_sliding_attention_heads", None):
            return num_heads
        # Remote code configs
        other_setting = getattr(config, "attention_other_setting", None) or {}
        if other_setting.get("attention_type") == layer_type:
            return other_setting["num_attention_heads"]
    if getattr(config, "is_heterogeneous", False):
        return config.per_layer_config[layer_idx].num_attention_heads
    return config.num_attention_heads


def _get_rope_parameters(
    config: PreTrainedConfig, layer_idx: int, layer_type: str
) -> dict[str, Any]:
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if isinstance(rope_parameters.get(layer_type), dict):
        rope_parameters = dict(rope_parameters[layer_type])
        rope_parameters.setdefault("partial_rotary_factor", 1.0)
        return rope_parameters
    # Remote code configs keep per-layer lists
    yarn_only_types = getattr(config, "yarn_only_types", None)
    if yarn_only_types and layer_type not in yarn_only_types:
        rope_parameters = {}
    rope_parameters = dict(rope_parameters)
    rope_parameters.setdefault("rope_type", "default")
    rope_theta = config.rope_theta
    if isinstance(rope_theta, list):
        rope_theta = rope_theta[layer_idx]
    rope_parameters["rope_theta"] = rope_theta
    partial_rotary_factors = getattr(config, "partial_rotary_factors", None)
    rope_parameters["partial_rotary_factor"] = (
        partial_rotary_factors[layer_idx] if partial_rotary_factors else 1.0
    )
    return rope_parameters


def _is_moe_layer(config: PreTrainedConfig, layer_idx: int) -> bool:
    if (mlp_layer_types := getattr(config, "mlp_layer_types", None)) is not None:
        mtp_mlp_layer_types = getattr(config, "mtp_mlp_layer_types", None) or []
        mlp_layer_types = [*mlp_layer_types, *mtp_mlp_layer_types]
        return layer_idx < len(mlp_layer_types) and (
            mlp_layer_types[layer_idx] == "sparse"
        )
    # Remote code configs
    if (moe_layers_enum := getattr(config, "moe_layers_enum", None)) is not None:
        return layer_idx in {int(i) for i in moe_layers_enum.strip().split(",")}
    return 0 < layer_idx < config.num_hidden_layers


class Step3p5MLP(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.limit = None
        layer_idx = extract_layer_index(prefix)
        swiglu_limits_shared = config.swiglu_limits_shared or []
        # Upstream configs drop the entries for the trailing MTP layers
        if layer_idx < len(swiglu_limits_shared) and swiglu_limits_shared[layer_idx]:
            self.limit = swiglu_limits_shared[layer_idx]
            self.act_fn = SwigluStepAndMul(limit=self.limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output


class Step3p5Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        # Step3p5 specific args
        sliding_window: int | None = None,
        use_head_wise_attn_gate: bool = False,
        use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)
        rope_parameters = dict(rope_parameters or {})
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", 10000)
        rope_parameters.setdefault("partial_rotary_factor", 1.0)

        self.rank = get_tensor_model_parallel_rank()
        self.partial_rotary_factor = rope_parameters["partial_rotary_factor"]
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
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_parameters["rope_theta"]
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, rms_norm_eps)
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        if use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        self.use_rope = use_rope

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
        )

        self.max_position_embeddings = max_position
        assert self.partial_rotary_factor == 1 or self.partial_rotary_factor == 0.5
        self.rotary_dim = (
            self.head_dim if self.partial_rotary_factor == 1 else self.head_dim // 2
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm inline similar to Qwen3 MOE attention
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)
            output = (
                attn_output.view(*attn_output.shape[:-1], self.num_heads, self.head_dim)
                * extra_dims.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        output, _ = self.o_proj(attn_output)
        return output


class FusedMoEBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)

        self.ep_size = get_ep_group().device_group.size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        # Router logits are accumulated in FP32 for higher precision.
        self.gate = GateLinear(
            config.hidden_size,
            config.moe_num_experts,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.use_moe_router_bias = config.use_moe_router_bias
        assert self.use_moe_router_bias, "Only support use_moe_router_bias is true."
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.need_fp32_gate = config.need_fp32_gate
        assert self.need_fp32_gate, (
            "Router logits must use FP32 precision for numerical stability."
        )

        activation = "silu"
        swiglu_limits = config.swiglu_limits or []
        swiglu_limit = (
            swiglu_limits[self.layer_idx]
            if self.layer_idx < len(swiglu_limits)
            else None
        )
        if swiglu_limit not in (None, 0):
            assert swiglu_limit is not None
            swiglu_limit = float(swiglu_limit)
            assert swiglu_limit == 7.0, (
                "Swiglu limit in fused moe block only support 7.0 now."
            )
            activation = "swiglustep"
            logger.debug(
                "step3p5 layer_idx: %s, activation: %s, limit: %s",
                self.layer_idx,
                activation,
                swiglu_limit,
            )

        self.share_expert = Step3p5MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.share_expert_dim,
            hidden_act="silu",
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.share_expert",
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.share_expert,
            gate=self.gate,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            activation=activation,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "moe_router_activation", "sigmoid"),
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )

        return final_hidden_states.view(num_tokens, hidden_dim)


class Step3p5DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        if cache_config is not None:
            cache_config.sliding_window = None
        att_impl_type = getattr(config, "att_impl_type", "GQA")
        if att_impl_type != "GQA":
            raise ValueError(f"Unsupported attention implementation: {att_impl_type}")
        layer_type = _get_layer_type(config, layer_idx)
        use_rope_layers = getattr(config, "use_rope_layers", None)
        assert config.max_position_embeddings is not None
        self.self_attn = Step3p5Attention(
            hidden_size=self.hidden_size,
            num_heads=_get_num_attention_heads(config, layer_idx, layer_type),
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_attention_groups,
            rope_parameters=_get_rope_parameters(config, layer_idx, layer_type),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            sliding_window=getattr(config, "sliding_window", None)
            if layer_type == "sliding_attention"
            else None,
            use_head_wise_attn_gate=getattr(config, "use_head_wise_attn_gate", False),
            use_rope=use_rope_layers[layer_idx] if use_rope_layers else True,
            prefix=f"{prefix}.self_attn",
        )
        self.use_moe = False
        self.use_fused_all_reduce = (
            get_tensor_model_parallel_world_size() > 1
            and get_dp_group().world_size == 1
        )
        if self.use_fused_all_reduce:
            logger.warning_once("Enable custom fused all reduce...")
        else:
            logger.warning_once("Disable custom fused all reduce...")

        if _is_moe_layer(config, layer_idx):
            self.moe = FusedMoEBlock(
                vllm_config,
                prefix=f"{prefix}.moe",
            )
            self.use_moe = True
        else:
            self.mlp = Step3p5MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.prefix = prefix

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_moe:
            ffn_output = self.moe(hidden_states)
        else:
            ffn_output = self.mlp(hidden_states)
        hidden_states = ffn_output + residual
        return hidden_states


@support_torch_compile
class Step3p5Model(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config

        self.moe_num_experts = config.moe_num_experts

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step3p5DecoderLayer(
                vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        return hidden_states


class Step3p5ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    # Required so quantization exclude lists match fused module prefixes.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.model = Step3p5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.moe_layers: list[MoERunner] = []
        example_layer: FusedMoEBlock | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Step3p5DecoderLayer)
            if hasattr(layer, "moe") and isinstance(layer.moe, FusedMoEBlock):
                example_layer = layer.moe
                self.moe_layers.append(layer.moe.experts)

        assert len(self.moe_layers) > 0, "No MoE layers found in the model."
        assert example_layer is not None
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.norm(hidden_states)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.moe_layers:
            assert isinstance(layer, FusedMoEBlock)
            layer.n_local_physical_experts = num_local_physical_experts
            layer.n_physical_experts = num_physical_experts
            layer.n_redundant_experts = self.num_redundant_experts
            layer.experts.update_expert_map()
