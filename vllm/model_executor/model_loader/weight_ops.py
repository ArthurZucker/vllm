# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conversion ops vLLM adds to Transformers' (`transformers.core_model_loading`)."""

import torch
from transformers.core_model_loading import Concatenate


class OutputConcatenate(Concatenate):
    """`Concatenate` along the target param's `output_dim` (an AWQ `qweight` packs its
    output along dim 1); 0-dim tensors (per-tensor scales) are stacked"""

    def __init__(self):
        super().__init__(dim=0)

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        model, name = kwargs.get("model"), kwargs.get("full_layer_name", "")
        try:
            param = model.get_parameter(name)
        except (AttributeError, TypeError):
            param = None
        self.dim = getattr(param, "output_dim", None) or 0
        return super().convert(input_dict, source_patterns, target_patterns, **kwargs)
