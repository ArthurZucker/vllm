# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

_requantization_source: ContextVar[bool] = ContextVar(
    "requantization_source", default=False
)


@contextmanager
def skip_checkpoint_backend_selection() -> Generator[None]:
    """Skip backend selection for checkpoint quantization methods in this scope.

    Covers checkpoint methods used as requantization sources (only their
    dequantize op is used) and unquantized checkpoint methods discarded when an
    online target replaces them, so oracles need not probe a compute backend.
    """
    token = _requantization_source.set(True)
    try:
        yield
    finally:
        _requantization_source.reset(token)


def is_requantization_source() -> bool:
    return _requantization_source.get()


class OnlineQuantMixin:
    """An online method's activation key, and its requantization source: a
    checkpoint method whose weights it requantizes as they load (the source's
    `dequantize_op` on each checkpoint weight before the model's ops, this
    method's quantize op after them)."""

    # Only the MXFP4 and FP8 per-channel (PTPC) online linear methods honor an
    # activation override for now.
    accepts_activation_override: bool = False

    @classmethod
    def resolve_activation_quant_key(
        cls, activation_quant_key: QuantKey | None, explicit: bool
    ) -> QuantKey | None:
        """Final activation key for a requested (or default) key."""
        # TODO: this should be removed and handled by oracles once oracles are
        # refactored.
        if explicit and not cls.accepts_activation_override:
            raise NotImplementedError(
                f"activation override is not supported for online {cls.__name__}"
            )
        if activation_quant_key is not None and activation_quant_key.scale.static:
            # Online quantization has no calibrated static activation scales.
            raise NotImplementedError(
                f"online {cls.__name__} cannot honour "
                f"activation={activation_quant_key}: static activation scales are "
                "not supported. Use a dynamic activation key, or 'activation: null'."
            )

        return activation_quant_key

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requantization_source: QuantizeMethodBase | None = None

    def set_requantization_source(self, source_method: QuantizeMethodBase) -> None:
        self.requantization_source = source_method

    def quantizer(self, layer: torch.nn.Module, targets: dict[str, tuple]):
        """The plan's quantizer of `targets`, after the source's dequantize op"""
        from vllm.model_executor.models.transformers.conversion import (
            VllmQuantizeQuantizer,
            VllmRequantizeQuantizer,
        )

        if self.requantization_source is None:
            return VllmQuantizeQuantizer(targets)
        dequantize, names, dropped = self.requantization_source.dequantize_op(layer)
        return VllmRequantizeQuantizer(targets, dequantize, names, dropped)
