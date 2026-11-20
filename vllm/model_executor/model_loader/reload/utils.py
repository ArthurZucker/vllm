# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch

from .types import LayerReloadingInfo, LayerTensors

__all__ = [
    "get_layer_tensors",
    "get_layer_params_buffers",
    "get_layer_size",
    "get_tensor_load_numel",
    "has_device_tensors",
    "get_info_size",
]


def get_layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get all parameters and buffers from a module as a dict."""
    params, buffers = get_layer_params_buffers(layer)
    return params | buffers


def get_layer_params_buffers(layer: torch.nn.Module) -> LayerTensors:
    """Get all parameters and buffers of a module as a tuple of dicts."""
    return (
        {name: param for name, param in layer._parameters.items() if param is not None},
        {name: buffer for name, buffer in layer._buffers.items() if buffer is not None},
    )


def get_tensor_load_numel(tensor: torch.Tensor) -> int:
    """Count checkpoint elements, excluding padding declared by the weight creator."""
    return getattr(tensor, "weight_loader_numel", tensor.numel())


def get_layer_size(layer: torch.nn.Module) -> int:
    """Calculate total number of elements across loadable tensors in a layer.

    Excludes SKIP_LOAD_TENSORS (e.g. _expert_map) which are never loaded during
    layerwise reload.
    """
    from .meta import SKIP_LOAD_TENSORS

    return sum(
        get_tensor_load_numel(tensor)
        for name, tensor in get_layer_tensors(layer).items()
        if name not in SKIP_LOAD_TENSORS
    )


def has_device_tensors(values: Iterable[object]) -> bool:
    """Return True if any of the loaded values is on an accelerator device."""
    return any(
        isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
        for value in values
    )


def get_info_size(info: LayerReloadingInfo) -> int:
    """Bytes of device memory held by a layer's buffered writes."""
    return sum(
        value.nbytes
        for _, _, value in info.loaded_weights
        if value.device.type not in ("meta", "cpu")
    )
