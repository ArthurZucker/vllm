# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load checkpoints with Transformers' loader: its conversions and quantization plan.

```
checkpoint key ─▶ plan: the layer's quantizer
                   ├─ vLLM has the kernels: tensors kept serialized (`load_serialized`)
                   └─ only Transformers reads it: dequantized by its ops
               ─▶ model converters (qkv split, gate_up concat, ...), scales alongside
               ─▶ `ops_loader.load_with_ops`: sharded on read, copied into vLLM's params
```
"""

from functools import partial

import torch
from transformers.core_model_loading import ConversionOps
from transformers.quantizers import (
    HfQuantizer,
    register_quantization_config,
    register_quantizer,
)
from transformers.quantizers.auto import (
    AUTO_QUANTIZATION_CONFIG_MAPPING,
    AUTO_QUANTIZER_MAPPING,
    AutoHfQuantizer,
    AutoQuantizationConfig,
)
from transformers.quantizers.schema_readers import (
    MODELOPT_ALGOS,
    read_quantization_schema,
)
from transformers.utils.quantization_config import QuantizationConfigMixin

from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.model_executor.layers.quantization.modelopt import LINEAR_ALGOS


class VllmFormatConfig(QuantizationConfigMixin):
    """A format vLLM reads and Transformers does not: the checkpoint's own config"""

    def __init__(self, quant_method: str, **kwargs):
        self.quant_method = quant_method
        self.__dict__.update(kwargs)


class VllmFormatQuantizer(HfQuantizer):
    """Its tensors go to vLLM's kernels as they are: no load op, no renames"""

    requires_calibration = False
    load_serialized = True
    replaces_modules = True  # vLLM's own modules: mixes with the plan's other formats

    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        return False


class VllmLayoutQuantizer(VllmFormatQuantizer):
    """A format vLLM loads into its own layout: `op` (a `ConversionOps` taking the
    quantizer) on each of the tensors `names`, whole, before the model's ops (e.g.
    GPTQ / AWQ experts repacked before they merge)"""

    load_serialized = False
    load_before_model_ops = True

    def __init__(self, op: type, names: list[str]):
        super().__init__(VllmFormatConfig("vllm_layout"))
        self.op, self.serialized_names = op, {"weight": list(names)}

    def get_load_operations(self):
        return {"deserialize": self.op}

    def get_weight_conversions(self):
        return []  # the model's converters read the tensors


class QuantizeOnLoad(ConversionOps):
    """A full precision tensor quantized as it loads (an online quantization): its
    quantizer's function of the param, whose outputs fill the param and its scales.
    Experts are stacked: one call each, unless the function takes them stacked"""

    def __init__(self, quantizer: "VllmQuantizeQuantizer"):
        self.quantizer = quantizer

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        key = kwargs["full_layer_name"]
        prefix, _, name = key.rpartition(".")
        quantize, scale_names, stacked = self.quantizer.targets[name]
        layer = kwargs["model"].get_submodule(prefix)
        value = next(iter(input_dict.values()))
        tensor = value[0] if isinstance(value, list) else value
        tensor = tensor.to(getattr(layer, name).device, layer.params_dtype).contiguous()
        if tensor.dim() == 3 and not stacked:  # experts, one at a time
            values, *scales = (torch.stack(o) for o in zip(*map(quantize, tensor)))
        else:
            values, *scales = quantize(tensor)
        out = {key: values}
        for scale_name, scale in zip(scale_names, scales):
            param = getattr(layer, scale_name)
            if (
                scale.numel() != param.numel()
            ):  # one a row, in each of the param's slots
                scale = scale.reshape(len(scale), *[1] * (param.dim() - 1))
                scale = scale.expand(param.shape).contiguous()
            out[f"{prefix}.{scale_name}"] = scale
        return out


class VllmQuantizeQuantizer(VllmFormatQuantizer):
    """Quantizes a full precision checkpoint as it loads: `targets` maps a param
    name to `(quantize, its scales' names, takes stacked experts)`"""

    load_serialized = False

    def __init__(self, targets: dict[str, tuple]):
        super().__init__(VllmFormatConfig("vllm_online"), pre_quantized=False)
        self.targets = targets

    def param_needs_quantization(self, model, param_name, **kwargs):
        return param_name.rpartition(".")[2] in self.targets

    def get_quantize_ops(self):
        return QuantizeOnLoad(self)


def checkpoint_plan(model) -> HfQuantizer | None:
    """The layers whose quant method loads its own layout (`checkpoint_quantizer`),
    each with its quantizer; the others take the checkpoint's tensors as they are"""
    from transformers.quantizers.auto import QuantizationPlanHfQuantizer

    plan = {
        name: quantizer
        for name, module in model.named_modules()
        if (method := getattr(module, "quant_method", None)) is not None
        and hasattr(method, "checkpoint_quantizer")
        and (quantizer := method.checkpoint_quantizer(module)) is not None
    }
    if not plan:
        return None
    hf_quantizer = QuantizationPlanHfQuantizer(plan)
    hf_quantizer.map_module_to_quantizer(model)
    return hf_quantizer


def register_vllm_formats():
    """Every format vLLM reads that Transformers does not, in its registries"""
    for method in QUANTIZATION_METHODS:
        if method not in AUTO_QUANTIZER_MAPPING:
            register_quantizer(method)(VllmFormatQuantizer)
        if method not in AUTO_QUANTIZATION_CONFIG_MAPPING:
            register_quantization_config(method)(VllmFormatConfig)
    # ModelOpt's per-layer algos, in its `quantized_layers`
    for algo in LINEAR_ALGOS:
        MODELOPT_ALGOS.setdefault(
            algo, partial(VllmFormatConfig, "modelopt", quant_algo=algo)
        )


def dequantizable(quant_cfg: dict) -> bool:
    """Whether Transformers dequantizes this format, which vLLM has no kernels for"""
    method = quant_cfg.get("quant_method")
    if method in QUANTIZATION_METHODS or method not in AUTO_QUANTIZER_MAPPING:
        return False
    quantizer = AUTO_QUANTIZER_MAPPING[method](
        AutoQuantizationConfig.from_dict(quant_cfg)
    )
    return "dequantize" in quantizer.get_load_operations()


def quantization_plan(quant_cfg: dict | None, serialized: bool) -> HfQuantizer | None:
    """Transformers' quantizer of the checkpoint, one per layer.

    `serialized`: vLLM has the kernels, which take the tensors as they are. Else
    Transformers' ops dequantize them.
    """
    if not quant_cfg:
        return None
    register_vllm_formats()
    plan = read_quantization_schema(quant_cfg)
    hf_quantizer = AutoHfQuantizer.from_plan(
        plan or {"*": quant_cfg}, None, True, checkpoint_names=plan is not None
    )
    for quantizer in hf_quantizer.all_quantizers:
        quantizer.load_serialized = serialized
        quantizer.quantization_config.dequantize = not serialized
    return hf_quantizer
