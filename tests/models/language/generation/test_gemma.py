# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.core_model_loading import WeightConverter, rename_source_key

from vllm.model_executor.model_loader.conversion_mapping import _gemma4, rename_mapper
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

from ....utils import multi_gpu_test
from ...utils import check_logprobs_close

MODELS = ["google/gemma-2-2b", "google/gemma-3-4b-it"]


@pytest.mark.cpu_test
def test_gemma4_attention_mapper() -> None:
    """Layers with a qkv_proj pack q/k/v; `attention_k_eq_v` full-attention
    layers also load K as the V shard (their own converter repeats it)."""
    config = SimpleNamespace(
        num_hidden_layers=3,
        num_kv_shared_layers=1,
        attention_k_eq_v=True,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )
    transforms = _gemma4(SimpleNamespace(config=config))
    renamings = [t for t in transforms if not isinstance(t, WeightConverter)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]
    names = [
        f"model.layers.{i}.self_attn.{tensor}.weight"
        for i in range(2)
        for tensor in ("q_proj", "k_proj", "k_norm")
    ] + ["model.layers.0.mlp.up_proj.weight"]

    mapped = []
    for name in names:
        target, pattern = rename_source_key(name, renamings, converters)
        converter = next((c for c in converters if pattern in c.source_patterns), None)
        shard = None if converter is None else converter.source_patterns.index(pattern)
        mapped.append((target, shard, converter))

    assert [(target, shard) for target, shard, _ in mapped] == [
        ("model.layers.0.self_attn.qkv_proj.weight", 0),
        ("model.layers.0.self_attn.qkv_proj.weight", 1),
        ("model.layers.0.self_attn.k_norm.weight", None),
        ("model.layers.1.self_attn.qkv_proj.weight", 0),
        ("model.layers.1.self_attn.qkv_proj.weight", 1),
        ("model.layers.1.self_attn.k_norm.weight", None),
        ("model.layers.0.mlp.gate_up_proj.weight", 1),
    ]
    # the k_eq_v layer's q/k go through one converter, which repeats K as V
    assert mapped[3][2] is mapped[4][2] is not mapped[0][2]
    assert len(mapped[3][2].source_patterns) == 2


@pytest.mark.cpu_test
def test_gemma4_expert_names_strip_language_model_prefix() -> None:
    """The text-only path reuses the conditional wrapper's checkpoint naming,
    so fused and per-expert tensors reach the experts under `model.*`."""
    prefix = "model.language_model.layers.0."
    weights = [
        (prefix + name, torch.empty(0))
        for name in (
            "experts.gate_up_proj",
            "experts.3.down_proj.weight_packed",
            "router.per_expert_scale",
        )
    ]

    mapped = [name for name, _ in rename_mapper(Gemma4ForCausalLM).apply(weights)]

    assert mapped == [
        "model.layers.0.experts.gate_up_proj",
        "model.layers.0.experts.3.down_proj.weight_packed",
        "model.layers.0.router.per_expert_scale",
    ]


@pytest.mark.parametrize("model", MODELS)
def test_dummy_loader(vllm_runner, monkeypatch, model: str) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        # Fork fails here when no earlier file switched the process to spawn.
        m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        with vllm_runner(
            model,
            load_format="dummy",
        ) as llm:
            if model == "google/gemma-3-4b-it":
                normalizers = llm.llm.collective_rpc(
                    lambda self: (
                        self.model_runner.model.language_model.model.normalizer.cpu().item()
                    )  # noqa: E501
                )
                config = llm.llm.llm_engine.model_config.hf_config.text_config
            else:
                normalizers = llm.llm.collective_rpc(
                    lambda self: self.model_runner.model.model.normalizer.cpu().item()
                )
                config = llm.llm.llm_engine.model_config.hf_config
            assert np.allclose(normalizers, config.hidden_size**0.5, rtol=2e-3)


@multi_gpu_test(num_gpus=2)
def test_gemma4_per_layer_embeddings_pipeline_parallel(
    vllm_runner, example_prompts, monkeypatch
) -> None:
    """Stages after the first must use the per-layer embeddings they receive
    from the previous stage, so PP=2 matches PP=1."""
    # The base model continues raw prompts; the -it model ends the turn at once.
    model = "google/gemma-4-E2B"
    text_only = {"image": 0, "audio": 0, "video": 0}
    with vllm_runner(model, limit_mm_per_prompt=text_only) as vllm_model:
        pp1_outputs = vllm_model.generate_greedy_logprobs(example_prompts, 32, 5)

    # KV-shared layers must sit on the same stage as the layers whose KV they
    # read, which the default split breaks. Put layers 13-34 (the two KV
    # source layers 13 and 14 and all 20 KV-shared layers) on stage 2.
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "13,22")
    with vllm_runner(
        model, pipeline_parallel_size=2, limit_mm_per_prompt=text_only
    ) as vllm_model:
        pp2_outputs = vllm_model.generate_greedy_logprobs(example_prompts, 32, 5)

    check_logprobs_close(
        outputs_0_lst=pp1_outputs,
        outputs_1_lst=pp2_outputs,
        name_0="pp1",
        name_1="pp2",
    )
