# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
    make_wna16_moe_kernel,
    make_wna16_moe_quant_config,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    normalize_and_validate_gptq_desc_act,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    INT4_DTYPE,
    INT8_DTYPE,
    QuantKey,
    kInt4Static32GroupScale,
    kInt4StaticGroupScale,
    kInt8StaticGroupScale,
)
from vllm.model_executor.model_loader.weight_ops import WholeOp
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform


class MoeWNA16Config(QuantizationConfig):
    """Config class for MOE WNA16 (W8A16/W4A16) quantization."""

    def __init__(
        self,
        linear_quant_method: str,
        weight_bits: int,
        group_size: int,
        has_zp: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: list[str] | None,
        full_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // self.weight_bits
        self.lm_head_quantized = lm_head_quantized
        self.linear_quant_method = linear_quant_method
        self.full_config = full_config
        # Avoid circular import
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        if self.linear_quant_method == "gptq":
            normalize_and_validate_gptq_desc_act(
                full_config.get("desc_act", False),
                group_size,
                full_config.get("dynamic") or {},
            )
        elif self.linear_quant_method in ("awq", "awq_marlin"):
            capability_tuple = current_platform.get_device_capability()
            device_capability = (
                -1 if capability_tuple is None else capability_tuple.to_int()
            )
            awq_min_capability = AutoAWQConfig.get_min_capability()
            if device_capability < awq_min_capability:
                raise ValueError(
                    "The quantization method moe_wna16 + awq is not supported "
                    "for the current GPU. "
                    f"Minimum capability: {awq_min_capability}. "
                    f"Current capability: {device_capability}."
                )
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        if modules_to_not_convert is None:
            self.modules_to_not_convert = []
        else:
            self.modules_to_not_convert = modules_to_not_convert

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "moe_wna16"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Config":
        linear_quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        if linear_quant_method == "gptq":
            has_zp = not cls.get_from_keys(config, ["sym"])
            modules_to_not_convert = []
        elif linear_quant_method in ("awq", "awq_marlin"):
            has_zp = cls.get_from_keys(config, ["zero_point"])
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        return cls(
            linear_quant_method,
            weight_bits,
            group_size,
            has_zp,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        can_convert = cls.is_moe_wna16_compatible(hf_quant_cfg)
        if can_convert and user_quant == "moe_wna16":
            return cls.get_name()
        return None

    @classmethod
    def is_moe_wna16_compatible(cls, quant_config: dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        desc_act = quant_config.get("desc_act", False)
        try:
            normalize_and_validate_gptq_desc_act(
                desc_act,
                quant_config.get("group_size"),
                quant_config.get("dynamic") or {},
            )
            supported_act_order = True
        except ValueError:
            supported_act_order = False

        capability_tuple = current_platform.get_device_capability()
        device_capability = (
            -1 if capability_tuple is None else capability_tuple.to_int()
        )
        # Avoid circular import
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        awq_min_capability = AutoAWQConfig.get_min_capability()

        gptq_compatible = (
            quant_method == "gptq" and supported_act_order and num_bits in [4, 8]
        )
        awq_compatible = (
            quant_method == "awq"
            and num_bits == 4
            and device_capability >= awq_min_capability
        )

        return gptq_compatible or awq_compatible

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            # Avoid circular import
            from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
            from vllm.model_executor.layers.quantization.auto_gptq import (
                AutoGPTQConfig,
            )

            linear_config: QuantizationConfig
            if self.linear_quant_method == "gptq":
                linear_config = AutoGPTQConfig.from_config(self.full_config)
            elif self.linear_quant_method in ("awq", "awq_marlin"):
                linear_config = AutoAWQConfig.from_config(self.full_config)
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")

            # The delegate is rebuilt from the raw HF quantization dict, which
            # lists shard names and never fused ones. Without the mapping it
            # resolves fused layers (qkv_proj, gate_up_proj) to
            # UnquantizedLinearMethod and the checkpoint's qweight has nowhere
            # to load.
            linear_config.packed_modules_mapping = self.packed_modules_mapping
            return linear_config.get_quant_method(layer, prefix)
        elif isinstance(layer, RoutedExperts):
            return MoeWNA16Method(self, layer.moe_config)
        return None


def is_layer_skipped_quant(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class MoeWNA16Method(FusedMoEMethodBase):
    """Linear method for MOE WNA16 (W8A16/W4A16) quantization.

    Args:
        quant_config: The MOE WNA16 (W8A16/W4A16) quantization config.

    """

    def __init__(self, quant_config: MoeWNA16Config, moe: "FusedMoEConfig") -> None:
        super().__init__(moe)
        self.quant_config = quant_config

        num_bits = self.quant_config.weight_bits
        group_size = self.quant_config.group_size

        if num_bits == 4:
            quant_type = INT4_DTYPE
            if group_size == 32:
                scale = kInt4Static32GroupScale
            else:
                scale = kInt4StaticGroupScale
        elif num_bits == 8:
            quant_type = INT8_DTYPE
            scale = kInt8StaticGroupScale
        else:
            raise ValueError("MoeWNA16Method only supports int4 and int8 now.")

        weight_key = QuantKey(quant_type, scale)

        # Select WNA16 MoE backend via oracle.
        # handle ZP?
        self.wna16_backend, self.experts_cls = select_wna16_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            quant_config=self.quant_config,
            may_have_zp=self.quant_config.has_zp,
            may_have_bias=False,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.quant_config = self.quant_config
        bit8_pack_factor = self.quant_config.bit8_pack_factor
        group_size = self.quant_config.group_size
        group_size_div_factor = 1

        # make intermediate_size and hidden_size divisible by group_size
        # we reduce the group size to ensure that
        # and we would repeat the loaded_weight later
        while intermediate_size_per_partition % group_size or hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        layer.group_size = group_size
        layer.group_size_div_factor = group_size_div_factor

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": False})

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        if self.quant_config.has_zp:
            w13_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards
                    * intermediate_size_per_partition
                    // bit8_pack_factor,
                    hidden_size // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)

            w2_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size // bit8_pack_factor,
                    intermediate_size_per_partition // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)

        if self.quant_config.linear_quant_method == "gptq":
            # some param are unused, but we need to init them in order to
            # load weights
            invalid_param_keys: list[str] = []
            if not self.quant_config.has_zp:
                invalid_param_keys += ["w13_qzeros", "w2_qzeros"]
            for key in invalid_param_keys:
                param = torch.nn.Parameter(
                    torch.empty((0,), dtype=torch.int32), requires_grad=False
                )
                layer.register_parameter(key, param)
                set_weight_attrs(param, extra_weight_attrs)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        if self.wna16_backend == WNA16MoEBackend.HUMMING:
            from vllm.model_executor.layers.quantization.utils.humming import (
                get_humming_moe_quant_config,
            )

            return get_humming_moe_quant_config(
                layer,
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )

        has_zp = self.quant_config.has_zp
        return make_wna16_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w1_zp=layer.w13_qzeros if has_zp else None,
            w2_zp=layer.w2_qzeros if has_zp else None,
            group_size=layer.group_size,
            num_bits=self.quant_config.weight_bits,
        )

    def _setup_kernel(self, layer: RoutedExperts):
        assert self.experts_cls is not None
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.wna16_backend,
            routing_tables=layer._expert_routing_tables(),
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        has_zp = self.quant_config.has_zp
        converted = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_backend,
            layer=layer,
            quant_config=self.quant_config,
            w13=layer.w13_qweight,
            w2=layer.w2_qweight,
            w13_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w13_qzeros=layer.w13_qzeros if has_zp else None,
            w2_qzeros=layer.w2_qzeros if has_zp else None,
        )

        if converted is None:
            # Backend rewrote the layer's params in place (e.g. Humming).
            self._setup_kernel(layer)
            return

        (
            w13_qweight,
            w2_qweight,
            w13_scales,
            w2_scales,
            w13_qzeros,
            w2_qzeros,
            _,  # w13_bias
            _,  # w2_bias
        ) = converted

        # Replace common parameters
        replace_parameter(layer, "w13_qweight", w13_qweight)
        replace_parameter(layer, "w2_qweight", w2_qweight)
        replace_parameter(layer, "w13_scales", w13_scales)
        replace_parameter(layer, "w2_scales", w2_scales)
        layer.w13_weight = layer.w13_qweight
        layer.w2_weight = layer.w2_qweight

        if has_zp:
            assert w13_qzeros is not None and w2_qzeros is not None
            replace_parameter(layer, "w13_qzeros", w13_qzeros)
            replace_parameter(layer, "w2_qzeros", w2_qzeros)

        self._setup_kernel(layer)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
            routing_sink=layer.routing_sink,
        )

    def checkpoint_quantizer(self, layer: torch.nn.Module):
        """GPTQ / AWQ experts load into this layout (`MoeWNA16Unpack`)"""
        from vllm.model_executor.models.transformers.conversion import (
            VllmLayoutQuantizer,
        )

        names = [
            f"{p}_{t}" for p in ("w13", "w2") for t in ("qweight", "qzeros", "scales")
        ]
        return VllmLayoutQuantizer(MoeWNA16Unpack, names)


class MoeWNA16Unpack(WholeOp):
    """One expert's GPTQ / AWQ tensor in the layout `MoeWNA16Method` computes with:
    rows the output channels, int4 packed two to a byte, qzeros / scales repeated
    to the layer's (reduced) group size. Whole: sharded after, by the experts param"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        name = kwargs["full_layer_name"]
        layer = kwargs["model"].get_submodule(name.rpartition(".")[0])
        kind = next(k for k in ("qweight", "qzeros", "scales") if name.endswith(k))
        return {
            pattern: [_unpack(layer, t, kind) for t in tensors]
            for pattern, tensors in input_dict.items()
        }


def _unpack(layer: torch.nn.Module, tensor: torch.Tensor, kind: str) -> torch.Tensor:
    config = layer.quant_config
    if config.linear_quant_method in ("awq", "awq_marlin"):
        assert config.weight_bits == 4
        tensor = (
            unpack_int4(tensor, kind == "qweight") if kind != "scales" else tensor.T
        )
    elif config.linear_quant_method == "gptq":
        assert config.weight_bits in (4, 8)
        if kind == "qweight":
            tensor = tensor.T.contiguous().view(torch.uint8)
        elif kind == "qzeros":  # GPTQ stores zeros minus one: as AWQ's
            tensor = tensor.view(torch.uint8)
            int4 = config.weight_bits == 4
            tensor = _convert_gptq_int4_qzeros(tensor).T if int4 else tensor.T + 1
        else:
            tensor = tensor.T
    if layer.group_size_div_factor > 1 and kind != "qweight":
        tensor = tensor.repeat_interleave(layer.group_size_div_factor, 1)
    return tensor


def unpack_int4(
    tensor: torch.Tensor, is_weight: bool, reorder: bool = True, sign_flip: bool = False
) -> torch.Tensor:
    """int4 weight `(k, n // 8) int32 -> (n, k // 2) uint8`, zeros
    `(k // group, n // 8) -> (n // 2, k // group)`"""
    size0 = tensor.size(0)
    tensor = tensor.view(torch.uint8)
    shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
    tensor = (tensor[:, :, None] >> shifter) & 0xF
    tensor = tensor.view(-1, 8)
    if reorder:  # AutoAWQ's pack order (awq/utils/quant_utils.py)
        tensor = tensor[:, [0, 4, 1, 5, 2, 6, 3, 7]]
    if sign_flip:
        tensor = tensor ^ 0x8
    tensor = tensor.view(size0, -1).T.contiguous()
    if is_weight:
        return tensor[:, 1::2] * 16 + tensor[:, ::2]
    return tensor[1::2, :] * 16 + tensor[::2, :]


def _convert_gptq_int4_qzeros(tensor: torch.Tensor) -> torch.Tensor:
    shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
    tensor = ((tensor[:, :, None] >> shifter) & 0xF) + 1
    return tensor[:, :, 0] + tensor[:, :, 1] * 16
