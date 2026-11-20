# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers' quantizers for vLLM's formats: a per-layer plan of which tensors
vLLM's kernels take as stored, and which Transformers' ops (de)quantize on load."""

import re
from functools import partial

import torch
from transformers.core_model_loading import (
    Concatenate,
    ConversionOps,
    WeightConverter,
)
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
from vllm.model_executor.model_loader.weight_ops import _first


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

    def get_weight_conversions(self):
        return []  # the model's converters read the tensors


class VllmLayoutQuantizer(VllmFormatQuantizer):
    """`op` on the tensors `names`, whole, before the model's ops (e.g. GPTQ / AWQ
    experts repacked before they merge)"""

    load_serialized = False
    load_before_model_ops = True

    def __init__(self, op: type, names: list[str]):
        super().__init__(VllmFormatConfig("vllm_layout"))
        self.op, self.serialized_names = op, {"weight": list(names)}

    def get_load_operations(self):
        return {"deserialize": self.op}


class QuantizeOnLoad(ConversionOps):
    """Online quantization: `quantize(tensor)` fills the param and its scales (or
    returns `{param name: tensor}`); experts one at a time unless taken stacked"""

    def __init__(self, quantizer: "VllmQuantizeQuantizer"):
        self.quantizer = quantizer

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        key = kwargs["full_layer_name"]
        prefix, _, name = key.rpartition(".")
        quantize, scale_names, stacked = self.quantizer.targets[name]
        layer = kwargs["model"].get_submodule(prefix)
        tensor = _first(input_dict).to(getattr(layer, name).device, layer.params_dtype)
        tensor = tensor.contiguous()
        if tensor.dim() == 3 and not stacked:  # experts, one at a time
            values, *scales = (torch.stack(o) for o in zip(*map(quantize, tensor)))
        elif isinstance(quantized := quantize(tensor), dict):
            return {f"{prefix}.{n}": t for n, t in quantized.items()}
        else:
            values, *scales = quantized
        out = {key: values}
        for scale_name, scale in zip(scale_names, scales):
            param = getattr(layer, scale_name)
            if scale.numel() != param.numel():  # one a row, in each of its slots
                scale = scale.reshape(len(scale), *[1] * (param.dim() - 1))
                scale = scale.expand(param.shape).contiguous()
            out[f"{prefix}.{scale_name}"] = scale
        return out


class VllmQuantizeQuantizer(VllmFormatQuantizer):
    """`targets`: param name -> `(quantize, scale names, takes stacked experts)`"""

    load_serialized = False

    def __init__(self, targets: dict[str, tuple]):
        super().__init__(VllmFormatConfig("vllm_online"), pre_quantized=False)
        self.targets = targets

    def param_needs_quantization(self, model, param_name, **kwargs):
        return param_name.rpartition(".")[2] in self.targets

    def get_quantize_ops(self):
        return QuantizeOnLoad(self)


class DequantizeOnLoad(ConversionOps):
    """Each weight dequantized with its scales (read along by `ReadAlong`) before the
    model's ops; the scales and `dropped_names` are consumed"""

    def __init__(self, hf_quantizer: "VllmRequantizeQuantizer"):
        self.hf_quantizer = hf_quantizer

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        weight, *scales = self.hf_quantizer.serialized_names["weight"]
        dropped = (*scales, *self.hf_quantizer.dropped_names)
        dequantize = self.hf_quantizer.dequantize_tensor
        out = {}
        for key, value in input_dict.items():
            stem = key.removesuffix(weight)
            if stem != key:
                if missing := [s for s in scales if stem + s not in input_dict]:
                    raise ValueError(f"{key}: no {missing} to dequantize it with")
                ss = [input_dict[stem + s] for s in scales]
                out[key] = (
                    [dequantize(*t) for t in zip(value, *ss)]
                    if isinstance(value, list)
                    else dequantize(value, *ss)
                )
            elif not key.endswith(dropped):
                out[key] = value  # not quantized, e.g. a bias
        return out


class VllmRequantizeQuantizer(VllmQuantizeQuantizer):
    """`dequantize(weight, *scales)` before the model's ops (`names`: weight, then
    scales; `dropped`: read along and dropped), then `targets` requantize"""

    requantize = True
    load_before_model_ops = True  # the scales are read along by `ReadAlong`

    def __init__(
        self,
        targets: dict[str, tuple],
        dequantize,
        names: list[str],
        dropped: list[str] | tuple[str, ...] = (),
    ):
        super().__init__(targets)
        self.pre_quantized, self.dequantize_tensor = True, dequantize
        self.dropped_names = tuple(dropped)
        # the params the plan's ops fill, and the checkpoint's names
        self.serialized_names = {t: [t] for t in targets} | {"weight": list(names)}

    def get_load_operations(self):
        return {"deserialize": DequantizeOnLoad}  # what the plan reads, pre-quantized


class ReadAlong(WeightConverter):
    """A converter reading each weight's scales along, for the requantized `modules`
    only (q, k, v each dequantized before their concat); other keys fall through"""

    __slots__ = ("modules", "along", "shards")

    def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
        renamed, pattern = super().rename_source_key(source_key)
        if pattern is None or renamed.rpartition(".")[0] in self.modules:
            return renamed, pattern
        return source_key, None


def read_scales_along(conversions: list, hf_quantizer) -> list:
    """`ReadAlong` copies of the model's converters, plus one for unconverted layers
    (e.g. `o_proj`), first. Sources start with `(?:)`: Transformers finds a converter
    by its source strings, which must differ from the model converter's."""
    from vllm.model_executor.model_loader.conversion_mapping import Fused

    along: dict[str, None] = {}  # every requantizing format's names, in order
    modules: set[str] = set()
    for name, quantizer in hf_quantizer.quantizers.items():
        if getattr(quantizer, "requantize", False):
            names = (*quantizer.serialized_names["weight"], *quantizer.dropped_names)
            along |= dict.fromkeys(names)
            # an unprefixed checkpoint key names the module without its body
            modules.update((name, name.partition(".")[2]) if "." in name else (name,))
    if not modules:
        return conversions

    def read_along(sources, targets, operations, converter=None) -> ReadAlong:
        new = ReadAlong(sources, targets, operations=operations)
        new.modules, new.along = modules, len(along)
        new.shards = getattr(converter, "shards", None)
        if converter is not None:
            new.scope_prefix = converter.scope_prefix
            new.base_model_prefix = converter.base_model_prefix
            new.force_cpu = converter.force_cpu
        return new

    widened = []
    for converter in conversions:
        if not isinstance(converter, WeightConverter):
            continue
        sources = converter._original_source_patterns
        targets = converter._original_target_patterns
        if isinstance(converter, Fused):  # module names: into the fused weight
            if any(s.endswith("bias") for s in sources):
                continue  # not quantized
            # `.q_proj`, `.q_proj.` or `.q_proj.weight` (escaped): the module
            sources = [re.sub(r"\\\.(weight)?$", "", s) for s in sources]
            sources = [rf"(?:){s}\.{n}" for s in sources for n in along]
            targets = [re.sub(r"\.(weight)?$", "", t) + ".weight" for t in targets]
        elif all(s.endswith("weight") for s in sources):
            sources = [
                f"(?:){s.removesuffix('weight')}{n}" for s in sources for n in along
            ]
        else:
            continue
        ops = list(converter.operations)
        widened.append(read_along(sources, targets, ops, converter))
    alone = [rf"(?<![^.]){n}" for n in along]  # `.weight`, not `.q_proj_weight`
    widened.append(read_along(alone, "weight", [Concatenate(dim=0)]))
    return [*widened, *conversions]


def checkpoint_plan(model) -> HfQuantizer | None:
    """A plan of the layers whose quant method has a `checkpoint_quantizer`"""
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
    """Per-layer quantizers; not `serialized`: Transformers' ops dequantize"""
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
