# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for model adapters (adapters.py)."""

import pytest
from transformers import Gemma3Config, PreTrainedConfig, Qwen2Config

from vllm.model_executor.models.adapters import _resolve_num_labels

pytestmark = pytest.mark.cpu_test


def _composite_config(outer_labels=None, inner_labels=None):
    """Build a multimodal-style config whose text_config is a separate object."""
    config = Gemma3Config(text_config={"num_hidden_layers": 1})
    if outer_labels is not None:
        config.num_labels = outer_labels
    if inner_labels is not None:
        config.get_text_config().num_labels = inner_labels
    return config


def test_resolve_num_labels_text_only_config():
    config = Qwen2Config(num_labels=7)
    assert config.get_text_config() is config
    assert _resolve_num_labels(config, config.get_text_config()) == 7


def test_resolve_num_labels_defaults_when_undeclared():
    config = _composite_config()
    assert (
        _resolve_num_labels(config, config.get_text_config())
        == PreTrainedConfig().num_labels
    )


def test_resolve_num_labels_declared_on_outer_config():
    """Multimodal checkpoints keep id2label/problem_type on the top-level config."""
    config = _composite_config(outer_labels=20)
    assert config.get_text_config().num_labels == PreTrainedConfig().num_labels
    assert _resolve_num_labels(config, config.get_text_config()) == 20


def test_resolve_num_labels_declared_on_text_config():
    """Overrides written into text_config keep working."""
    config = _composite_config(inner_labels=5)
    assert _resolve_num_labels(config, config.get_text_config()) == 5


def test_resolve_num_labels_outer_wins_when_both_declared():
    config = _composite_config(outer_labels=20, inner_labels=5)
    assert _resolve_num_labels(config, config.get_text_config()) == 20
