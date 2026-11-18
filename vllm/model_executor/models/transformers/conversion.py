# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load checkpoints with Transformers' loader: its conversions and quantization plan.

```
checkpoint key ─▶ plan: the layer's quantizer
                   ├─ vLLM has the kernels: tensors kept serialized (`load_serialized`)
                   └─ only Transformers reads it: dequantized by its ops
               ─▶ model converters (qkv split, gate_up concat, ...), scales alongside
               ─▶ vLLM `weight_loader`s
```
"""

from collections import Counter
from collections.abc import Iterable, Iterator
from copy import deepcopy
from functools import partial

import torch
from transformers.core_model_loading import (
    WeightConverter,
    WeightTransform,
    rename_source_key,
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


def streamable_converters(conversions: list[WeightTransform]) -> list[WeightConverter]:
    """The converters to run on the stream: those taking one key per source pattern.

    Wildcard sources (`experts.*.gate_proj`) are expert merges, left out: `FusedMoE`'s
    `weight_loader` shards each checkpoint expert straight into place, without
    stacking all of them first.
    """
    return [
        conversion
        for conversion in conversions
        if isinstance(conversion, WeightConverter)
        and not any("*" in pattern for pattern in conversion.source_patterns)
    ]


def convert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    converters: list[WeightConverter],
    config=None,
    hf_quantizer: HfQuantizer | None = None,
    checkpoint_keys: Iterable[str] = (),
    renamings: list[WeightTransform] = (),
) -> Iterator[tuple[str, torch.Tensor]]:
    """`weights`, through Transformers' renamings, converters and quantization plan.

    A converter's tensors are held until it has them all, then converted: as many
    as `checkpoint_keys` name for its target, else one per source pattern. Keys no
    converter matches pass through. Scales follow their weight: a `.weight` pattern
    also matches `.weight_scale`, into a target of its own.
    """

    def rename(name):
        name = hf_quantizer.rename_layer(name) if hf_quantizer else name
        return rename_source_key(name, renamings, converters)

    expected = Counter(
        target for target, pattern in map(rename, checkpoint_keys) if pattern
    )
    pending: dict[str, WeightConverter] = {}
    seen: Counter[str] = Counter()
    for name, tensor in weights:
        target, pattern = rename(name)
        if pattern is None:
            yield target, tensor  # renamed, if a renaming matched
            continue
        if target not in pending:
            converter = next(c for c in converters if pattern in c.source_patterns)
            converter = deepcopy(converter)
            if hf_quantizer is not None:  # the layer's load op, if any
                converter = hf_quantizer.update_weight_transform(
                    converter, None, target
                )
            pending[target] = converter
        pending[target].add_tensor(target, name, pattern, tensor)
        seen[target] += 1
        if seen[target] == expected.get(target, len(pending[target].source_patterns)):
            yield from _convert(pending.pop(target), target, config)
    # A group missing a source, e.g. a scale only some of its weights have
    for target, converter in pending.items():
        yield from _convert(converter, target, config)


def _convert(converter: WeightConverter, target: str, config):
    for name, tensor in converter.convert(target, config=config).items():
        yield name, tensor[0] if isinstance(tensor, list) else tensor
