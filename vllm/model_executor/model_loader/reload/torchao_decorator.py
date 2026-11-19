# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import ModelConfig

__all__ = ["set_torchao_reload_attrs"]


def set_torchao_reload_attrs(model: torch.nn.Module, model_config: ModelConfig):
    model._do_torchao_reload = True
    model._model_config = model_config
