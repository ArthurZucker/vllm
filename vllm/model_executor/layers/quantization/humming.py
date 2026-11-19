# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial
from math import gcd
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from transformers.core_model_loading import ConversionOps

import vllm.utils.humming as _hm
from vllm import envs
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.humming import (
    check_and_fallback_input_schema,
    convert_to_humming_moe_kernel_format,
    get_humming_linear_compute_config,
    get_humming_moe_quant_config,
    humming_update_schema_hadamard_block_size,
    input_schema_to_quant_key,
    make_humming_moe_kernel,
    resolve_humming_layer_config,
    select_humming_moe_experts,
    weight_schema_to_quant_key,
)
from vllm.model_executor.layers.quantization.utils.humming.linear import (
    get_live_param,
    kernel_parameters,
    replace_layer_parameters,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper
    from vllm.utils.humming import (
        BaseInputSchema,
        BaseWeightSchema,
        HummingInputSchema,
        HummingWeightSchema,
    )


def prepare_param(tensor, name, extra_attrs):
    extra_attrs = extra_attrs.copy()
    scale_type = extra_attrs.pop("scale_type", None)
    param_cls_name_map = {
        "block": BlockQuantScaleParameter,
        "tensor": PerTensorScaleParameter,
        "group": GroupQuantScaleParameter,
        "channel": ChannelQuantScaleParameter,
        "input_scale": PerTensorScaleParameter,
    }

    param_cls: type[BasevLLMParameter]
    if "packed_dim" in extra_attrs:
        param_cls = PackedvLLMParameter
    elif scale_type in param_cls_name_map:
        param_cls = param_cls_name_map[scale_type]
    elif "output_dim" in extra_attrs and "input_dim" in extra_attrs:
        param_cls = ModelWeightParameter
    elif "input_dim" in extra_attrs:
        param_cls = RowvLLMParameter
    elif "output_dim" in extra_attrs:
        param_cls = ChannelQuantScaleParameter
    else:
        param_cls = BasevLLMParameter

    kwargs_keys = [
        "input_dim",
        "output_dim",
        "packed_dim",
        "packed_factor",
    ]
    cls_kwargs = {}
    for key in extra_attrs.copy():
        if key in kwargs_keys:
            cls_kwargs[key] = extra_attrs.pop(key)

    param = param_cls(data=tensor, **cls_kwargs)
    set_weight_attrs(param, extra_attrs)

    param.param_name = name
    param.ignore_warning = True
    if scale_type in ["tensor", "input_scale"]:
        param.needs_scalar_to_array = True

    return param


def prepare_moe_param(tensor: torch.Tensor, name: str, extra_attrs: dict[str, Any]):
    param = torch.nn.Parameter(tensor, requires_grad=False)
    if "scale_type" in extra_attrs:
        extra_attrs["quant_method"] = extra_attrs["scale_type"]

    if "input_dim" in extra_attrs and "output_dim" in extra_attrs:
        input_dim = extra_attrs["input_dim"]
        output_dim = extra_attrs["output_dim"]
        extra_attrs["is_transposed"] = input_dim < output_dim

    set_weight_attrs(param, extra_attrs)
    param.param_name = name
    return param


def compressed_tensors_get_config(config: dict[str, Any], key: str):
    assert key in ["weights", "input_activations"]
    target_group_config = None
    for group_config in config["config_groups"].values():
        if "Linear" in group_config["targets"]:
            if "weights" not in group_config:
                return None
            if key not in group_config or group_config[key] is None:
                return None
            target_group_config = group_config[key].copy()
            break

    if target_group_config is None:
        return None
    target_group_config["quant_method"] = config["quant_method"]
    if config["quant_method"] == "compressed-tensors":
        target_group_config["format"] = config["format"]
    elif config["quant_method"] == "modelopt":
        target_group_config["quant_algo"] = config["quant_algo"]
    return target_group_config


class HummingLoadedWeight(ConversionOps):
    """A checkpoint tensor in its Humming schema's layout
    (`process_loaded_weight`, e.g. BitNet's 2-bit repack). Whole, on each
    source (one expert, one of q / k / v): sharded after. A full precision `weight`
    stays as it is"""

    needs_full = True

    def __init__(self, quantizer=None):
        self.quantizer = quantizer

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        name = kwargs["full_layer_name"]
        model = kwargs["model"]
        schema = model.get_submodule(name.rpartition(".")[0]).quant_method.weight_schema
        param_name = model.get_parameter(name).param_name

        def process(tensor: torch.Tensor) -> torch.Tensor:
            if param_name == "weight" and tensor.dtype in _FLOAT_DTYPES:
                return tensor
            return schema.process_loaded_weight(tensor=tensor, name=param_name)

        return {p: [process(t) for t in v] for p, v in input_dict.items()}


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _loaded_weight_quantizer(schema, layer: torch.nn.Module):
    """`HummingLoadedWeight` on the layer's Humming params, unless its schema
    takes the checkpoint's tensors as they are"""
    if type(schema).process_loaded_weight is _hm.BaseWeightSchema.process_loaded_weight:
        return None
    from vllm.model_executor.models.transformers.conversion import (
        VllmLayoutQuantizer,
    )

    names = [
        n for n, p in layer.named_parameters(recurse=False) if hasattr(p, "param_name")
    ]
    return VllmLayoutQuantizer(HummingLoadedWeight, names)


class HummingConfig(QuantizationConfig):
    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, full_config: dict[str, Any] | None = None):
        self.full_config: dict[str, Any] = full_config or {}

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "humming"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HummingConfig":
        return cls(full_config=config)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if user_quant == "humming" and hf_config is not None:
            model_type = hf_config.model_type
            quant_method = hf_quant_cfg.get("quant_method", None)
            if model_type == "gpt_oss" and quant_method == "mxfp4":
                msg = (
                    "For gpt-oss model, use '--moe-backend humming' "
                    "instead of '--quantization humming'."
                )
                raise ValueError(msg)
        return "humming" if user_quant == "humming" else None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.hf_to_vllm_mapper = hf_to_vllm_mapper

    def is_layer_skipped(self, config: dict[str, Any], prefix: str):
        keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
        ignored_layers = self.get_from_keys_or(config, keys, []) or []
        if hasattr(self, "hf_to_vllm_mapper"):
            ignored_layers = self.hf_to_vllm_mapper.apply_list(ignored_layers)

        for module_name in ignored_layers:
            # compressed-tensors style entries may be regex patterns prefixed
            # with "re:" (e.g. "re:vision_tower.*"). These must be regex-matched;
            # plain substring matching never matches the literal "re:..." string,
            # so ignored layers get silently quantized. Non-"re:" entries keep the
            # existing substring behavior (e.g. bitsandbytes modules_to_not_convert).
            if module_name.startswith("re:"):
                if re.match(module_name[3:], prefix):
                    return True
            elif module_name in prefix:
                return True
        if "lm_head" in prefix:
            return True

        for regex in config.get("dynamic", {}):
            if regex[:1] != "-":
                continue
            if re.match(regex[2:], prefix):
                return True

        return False

    def get_layer_weight_schema(self, config: dict[str, Any], prefix: str):
        if self.is_layer_skipped(config, prefix):
            return None

        if config["quant_method"] in ["compressed-tensors", "modelopt"]:
            group_config = compressed_tensors_get_config(config, "weights")
            if group_config is None:
                return None
            config = group_config

        layer_config = resolve_humming_layer_config(config, prefix)

        if "quant_method" in layer_config:
            return _hm.BaseWeightSchema.from_config(layer_config)
        return None

    def get_layer_input_config(self, config: dict[str, Any], prefix: str):
        if self.is_layer_skipped(config, prefix):
            return None
        if config["quant_method"] in ["compressed-tensors", "modelopt"]:
            group_config = compressed_tensors_get_config(config, "input_activations")
            if group_config is None:
                return None
            config = group_config

        config = resolve_humming_layer_config(config, prefix)
        if config.get("quant_method", None) in _hm.BaseInputSchema.INPUT_SCHEMA_MAP:
            return config
        return None

    def get_quant_config_for_layer(
        self, prefix: str, layer_type: str
    ) -> "HummingLayerQuantizationConfig | None":
        weight_schema: BaseWeightSchema | None = None
        force_weight_schema: HummingWeightSchema | None = None

        if self.full_config:
            weight_schema = self.get_layer_weight_schema(self.full_config, prefix)

        is_online_quant = False
        online_quant_config = envs.VLLM_HUMMING_ONLINE_QUANT_CONFIG or {}
        if not self.full_config or online_quant_config.get("force_requant", False):
            online_quant_config["quant_method"] = "humming"
            schema = self.get_layer_weight_schema(online_quant_config, prefix)
            if not self.full_config:
                weight_schema = schema
                is_online_quant = True
            else:
                force_weight_schema = schema

        if weight_schema is not None:
            input_schema = None
            force_input_schema = None
            allow_input_schema_fallback = True

            if self.full_config:
                input_config = self.get_layer_input_config(self.full_config, prefix)
                if input_config is not None:
                    allow_input_schema_fallback = input_config.pop(
                        "allow_fallback", True
                    )
                    input_schema = _hm.BaseInputSchema.from_config(input_config)

            if envs.VLLM_HUMMING_INPUT_QUANT_CONFIG:
                quant_config = envs.VLLM_HUMMING_INPUT_QUANT_CONFIG.copy()
                quant_config["quant_method"] = "humming"
                input_config = self.get_layer_input_config(quant_config, prefix)
                if input_config is not None:
                    allow_input_schema_fallback = input_config.pop(
                        "allow_fallback", False
                    )
                    force_input_schema = _hm.BaseInputSchema.from_config(input_config)
                if input_schema is None:
                    input_schema = force_input_schema

            if force_weight_schema is not None and force_input_schema is None:
                force_input_schema = _hm.HummingInputSchema()

            return HummingLayerQuantizationConfig(
                weight_schema=weight_schema,
                input_schema=input_schema,
                force_weight_schema=force_weight_schema,
                force_input_schema=force_input_schema,
                is_online_quant=is_online_quant,
                allow_input_schema_fallback=allow_input_schema_fallback,
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        layer_type = "other"
        if isinstance(layer, RoutedExperts):
            layer_type = "moe"
        elif isinstance(layer, LinearBase):
            layer_type = "linear"

        quant_config = self.get_quant_config_for_layer(prefix, layer_type)
        if quant_config is None:
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            elif isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            input_size = layer.input_size
            partition_size = getattr(layer, "input_size_per_partition", input_size)
            is_online_quant = quant_config.is_online_quant
            if is_online_quant and (input_size % 32 != 0 or partition_size % 32 != 0):
                return UnquantizedLinearMethod()
            return HummingLinearMethod(quant_config)
        elif isinstance(layer, RoutedExperts):
            return HummingMoEMethod(quant_config, layer.moe_config)
        return None


class HummingLayerQuantizationConfig(HummingConfig):
    def __init__(
        self,
        weight_schema: "BaseWeightSchema",
        input_schema: "BaseInputSchema | None" = None,
        force_weight_schema: "HummingWeightSchema | None" = None,
        force_input_schema: "HummingInputSchema | None" = None,
        is_online_quant: bool = False,
        allow_input_schema_fallback: bool = True,
    ):
        self.weight_schema = weight_schema
        if input_schema is None:
            input_schema = _hm.HummingInputSchema()
        self.input_schema = input_schema
        self.force_weight_schema = force_weight_schema
        self.force_input_schema = force_input_schema
        self.is_online_quant = is_online_quant
        self.allow_input_schema_fallback = allow_input_schema_fallback

    @classmethod
    def from_config(cls, config):
        weight_schema = _hm.BaseWeightSchema.from_config(config)
        return cls(weight_schema)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        raise NotImplementedError


class HummingLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: HummingLayerQuantizationConfig):
        self.quant_config = quant_config
        self.weight_schema = quant_config.weight_schema
        self.input_schema = quant_config.input_schema
        self.force_weight_schema = quant_config.force_weight_schema
        self.force_input_schema = quant_config.force_input_schema
        self.is_online_quant = self.quant_config.is_online_quant

    def checkpoint_quantizer(self, layer: torch.nn.Module):
        if self.is_online_quant:  # a full precision checkpoint
            from vllm.model_executor.models.transformers.conversion import (
                VllmQuantizeQuantizer,
            )

            quantize = partial(self.quantize_weight, layer)
            return VllmQuantizeQuantizer({"weight": (quantize, [], True)})
        return _loaded_weight_quantizer(self.weight_schema, layer)

    def quantize_weight(
        self, layer: torch.nn.Module, weight: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The layer's params (`{name: tensor}`) of its full precision `weight`
        (this rank's), each of its shards (q / k / v) quantized on its own"""
        schema = self.weight_schema
        assert isinstance(schema, _hm.HummingWeightSchema)
        shards = [
            schema.quant_tensor(w, schema, layer.param_dtype)
            for w in weight.split(layer.output_partition_sizes)
        ]
        return {
            name: torch.cat(
                [torch.atleast_1d(s[name]) for s in shards],
                getattr(getattr(layer, name), "output_dim", 0),
            )
            for name, tensor in shards[0].items()
            if tensor is not None and tensor.nelement() > 0
        }

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.param_dtype = params_dtype
        layer.input_size = input_size
        layer.output_size = output_size
        layer.input_size_per_partition = input_size_per_partition
        layer.output_partition_sizes_sum = sum(output_partition_sizes)
        layer.output_partition_sizes = output_partition_sizes

        input_schema = self.force_input_schema or self.input_schema
        input_schema = input_schema.to_humming_schema(params_dtype)
        for name in ("weight_schema", "force_weight_schema"):
            schema = getattr(self, name)
            if getattr(schema, "hadamard_block_size", None) == -1:
                schema = humming_update_schema_hadamard_block_size(
                    weight_schema=schema,
                    input_schema=input_schema,
                    shape_k=input_size_per_partition,
                )
                setattr(self, name, schema)

        for key in ["weight_block_size", "block_structure"]:
            block_size = getattr(self.weight_schema, key, None)
            if block_size is not None:
                layer.weight_block_size = block_size

        weight_tensor_attrs = self.weight_schema.get_tensors_attrs(
            shape_n=layer.output_partition_sizes_sum,
            shape_k=layer.input_size_per_partition,
            param_dtype=params_dtype,
            stack_size=len(layer.output_partition_sizes),
        )

        input_tensor_attrs = self.input_schema.get_tensors_attrs(
            shape_k=layer.input_size_per_partition,
            param_dtype=params_dtype,
            stack_size=len(layer.output_partition_sizes),
        )

        tensors_attrs = weight_tensor_attrs | input_tensor_attrs

        for name, attrs in tensors_attrs.items():
            tensor = torch.empty(attrs["shape"], dtype=attrs["dtype"])
            extra_attrs = attrs.get("extra_attrs", {}).copy()
            extra_attrs.update(extra_weight_attrs)
            param = prepare_param(tensor, name, extra_attrs)
            setattr(layer, name, param)

        if self.force_input_schema is not None:
            self.input_schema = self.force_input_schema

        if not hasattr(layer, "weight"):
            param = prepare_param(torch.tensor(0), "weight", extra_weight_attrs)
            layer.weight = param

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # convert from checkpoint format to humming format
        is_humming_weight = isinstance(self.weight_schema, _hm.HummingWeightSchema)
        is_humming_input = isinstance(self.input_schema, _hm.HummingInputSchema)

        if not is_humming_weight or not is_humming_input:
            self.weight_schema, tensors = self.weight_schema.convert_humming(
                tensors=layer.state_dict(),
                shape_n_stacks=layer.output_partition_sizes,
                shape_k_stacks=[layer.input_size_per_partition],
                param_dtype=layer.param_dtype,
            )

            self.input_schema, input_tensors = self.input_schema.convert_humming(
                tensors=layer.state_dict(),
                shape_n_stacks=layer.output_partition_sizes,
                shape_k_stacks=[layer.input_size_per_partition],
                param_dtype=layer.param_dtype,
            )
            tensors.update(input_tensors)
            replace_layer_parameters(layer, tensors, list(layer._parameters))
            del tensors

        # force requant (origin quant setting -> fp16/bf16 -> new_quant setting)
        assert isinstance(self.weight_schema, _hm.HummingWeightSchema)
        force_requant = self.force_weight_schema is not None
        if force_requant and self.weight_schema != self.force_weight_schema:
            source_tensors = kernel_parameters(layer)
            tensors = self.weight_schema.requant_tensors(
                tensors=source_tensors,
                target_weight_schema=self.force_weight_schema,
                param_dtype=layer.param_dtype,
            )
            name = self.input_schema.static_tensor_scale_name
            if name is not None:
                tensors[name] = source_tensors[name]

            self.weight_schema = self.force_weight_schema

            names = [name for name in layer._parameters if name != "bias"]
            replace_layer_parameters(layer, tensors, names)
            del tensors

        self.input_schema = check_and_fallback_input_schema(
            weight_schema=self.weight_schema,
            input_schema=self.input_schema,
            param_dtype=layer.param_dtype,
            allow_fallback=self.quant_config.allow_input_schema_fallback,
        )
        source_tensors = kernel_parameters(layer)
        self.layer_config, tensors = _hm.prepare_and_transform_humming_tensors(
            tensors=source_tensors,
            shape_n=layer.output_partition_sizes_sum,
            shape_k=layer.input_size_per_partition,
            weight_schema=self.weight_schema,
            input_schema=self.input_schema,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            has_bias=layer.has_bias,
            torch_dtype=layer.param_dtype,
            device=layer.weight.device,
        )
        name = self.input_schema.static_tensor_scale_name
        if name is not None:
            tensors[name] = source_tensors[name]
        if "bias" in tensors:
            zero_bias = torch.zeros_like(tensors["bias"])
            layer.register_buffer("zero_bias", zero_bias)
        replace_layer_parameters(layer, tensors, list(layer._parameters))

        self.compute_config = get_humming_linear_compute_config()
        self.locks = _hm.create_humming_locks(layer.weight.device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flatten_inputs = x.reshape(-1, x.size(-1))
        output = _hm.humming_forward(
            self.layer_config,
            inputs=flatten_inputs,
            weight=layer.weight,
            weight_scale=get_live_param(layer, "weight_scale"),
            zero_point=get_live_param(layer, "zero_point"),
            bias=get_live_param(layer, "bias" if bias is not None else "zero_bias"),
            weight_scale_2=get_live_param(layer, "weight_scale_2"),
            input_scale=get_live_param(layer, "input_scale"),
            input_scale_2=get_live_param(layer, "input_scale_2"),
            hadamard_block_size=self.weight_schema.hadamard_block_size,
            locks=self.locks,
            compute_config=self.compute_config,
        )
        output = output.view(*x.shape[:-1], output.size(-1))
        return output


class HummingMoEMethod(FusedMoEMethodBase):
    def __init__(
        self, quant_config: HummingLayerQuantizationConfig, moe: "FusedMoEConfig"
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.weight_schema = quant_config.weight_schema
        self.input_schema = quant_config.input_schema
        self.force_weight_schema = quant_config.force_weight_schema
        self.force_input_schema = quant_config.force_input_schema

        # Derive QuantKeys from humming schemas.
        # Prefer force schemas (the final format after requant) over base.
        weight_key = weight_schema_to_quant_key(
            self.force_weight_schema or self.weight_schema, moe.in_dtype
        )
        runtime_input_schema = check_and_fallback_input_schema(
            weight_schema=self.force_weight_schema or self.weight_schema,
            input_schema=self.force_input_schema or self.input_schema,
            param_dtype=moe.in_dtype,
            allow_fallback=quant_config.allow_input_schema_fallback,
            # Keep the BF16 fallback on SM100-SM119 (see the MoE weight
            # processing call site in utils/humming/moe.py).
            allow_fp8_sm100=False,
        )
        activation_key = input_schema_to_quant_key(runtime_input_schema, moe.in_dtype)

        # Select Humming MoE experts
        self.experts_cls = select_humming_moe_experts(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
        )

    def checkpoint_quantizer(self, layer: torch.nn.Module):
        return _loaded_weight_quantizer(self.weight_schema, layer)

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.param_dtype = params_dtype
        layer.intermediate_size = intermediate_size_per_partition
        input_schema = self.force_input_schema or self.input_schema
        input_schema = input_schema.to_humming_schema(params_dtype)
        shape_k = gcd(hidden_size, intermediate_size_per_partition)
        for name in ("weight_schema", "force_weight_schema"):
            schema = getattr(self, name)
            if getattr(schema, "hadamard_block_size", None) == -1:
                schema = humming_update_schema_hadamard_block_size(
                    weight_schema=schema,
                    input_schema=input_schema,
                    shape_k=shape_k,
                )
                setattr(self, name, schema)

        # sublayer: a layer contains multiple sets of weights for quantized GEMM
        # (e.g., weight, weight_scale, etc.).
        # The weight names of sublayer start with the prefix "{sublayer_name}_"
        w13_output_size = intermediate_size_per_partition * self.moe.w13_num_shards
        layer.sublayer_configs = {
            "w13": {
                "shape_n": w13_output_size,
                "shape_k": hidden_size,
                "tensors_attrs": self.weight_schema.get_padded_tensors_attrs(
                    shape_n=w13_output_size,
                    shape_k=hidden_size,
                    num_experts=num_experts,
                    param_dtype=params_dtype,
                    has_bias=self.moe.has_bias,
                ),
            },
            "w2": {
                "shape_n": hidden_size,
                "shape_k": intermediate_size_per_partition,
                "tensors_attrs": self.weight_schema.get_padded_tensors_attrs(
                    shape_n=hidden_size,
                    shape_k=intermediate_size_per_partition,
                    num_experts=num_experts,
                    param_dtype=params_dtype,
                    has_bias=self.moe.has_bias,
                ),
            },
        }

        for sublayer_name, configs in layer.sublayer_configs.items():
            for name, attrs in configs["tensors_attrs"].items():
                tensor = torch.empty(attrs["shape"], dtype=attrs["dtype"])
                param = torch.nn.Parameter(tensor, requires_grad=False)
                extra_attrs = attrs.get("extra_attrs", {}).copy()
                extra_attrs.update(extra_weight_attrs)
                param = prepare_moe_param(tensor, name, extra_attrs)
                setattr(layer, f"{sublayer_name}_{name}", param)

        if self.force_input_schema is not None:
            self.input_schema = self.force_input_schema

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        return get_humming_moe_quant_config(
            layer,
            self.humming_configs,
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(self, "processed", False):
            return
        self.processed = True

        # Convert weights to Humming kernel format
        self.humming_configs = convert_to_humming_moe_kernel_format(
            layer=layer,
            sublayer_configs=layer.sublayer_configs,
            weight_schema=self.weight_schema,
            input_schema=self.input_schema,
            force_weight_schema=self.force_weight_schema,
            allow_input_schema_fallback=self.quant_config.allow_input_schema_fallback,
        )

        # Build the MoE kernel
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_humming_moe_kernel(
            self.moe_quant_config,
            self.moe,
            self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply Humming-quantized MoE computation using the standard kernel flow.

        This method uses FusedMoEKernel.apply() which orchestrates:
        1. Preparation (quantization if needed - skipped for Humming via
           expects_unquantized_inputs=True to prevent double quantization)
        2. Expert computation (via experts.apply())
        3. Finalization (weight application & reduction - no-op for Humming
           since it's already done internally)

        Humming handles quantization, weight application, and reduction
        internally in the experts implementation.

        Humming experts consume the weights and quantization tensors passed
        through the standard modular kernel interface.
        """
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=False,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
