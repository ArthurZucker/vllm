# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A merged layer's checkpoint, fused on disk, fills its shards and leaves the
output padding zero."""

import pytest
import torch

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.model_loader.ops_loader import load_weights

HIDDEN = 8

pytestmark = [pytest.mark.cpu_test, pytest.mark.usefixtures("dist_init")]


@pytest.mark.parametrize("padding_size", [0, 3])
def test_merged_checkpoint_bias_keeps_output_padding_zero(padding_size):
    layer = MergedColumnParallelLinear(HIDDEN, [4, 4], padding_size=padding_size)
    weight = torch.arange(8 * HIDDEN, dtype=layer.weight.dtype).reshape(8, HIDDEN)
    bias = torch.arange(8, dtype=layer.bias.dtype)
    with torch.no_grad():
        layer.weight.fill_(float("nan"))
        layer.bias.fill_(float("nan"))

    load_weights(layer, iter([("weight", weight), ("bias", bias)]))

    torch.testing.assert_close(layer.weight[:8], weight, rtol=0, atol=0)
    torch.testing.assert_close(layer.bias[:8], bias, rtol=0, atol=0)
    assert not layer.weight[8:].any()
    assert not layer.bias[8:].any()
