# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/blob/ac95e7b1678c08ea6f4a2581a051e432ed63e9b7/modeling_ming_v3_asr_encoder.py
# https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/blob/ac95e7b1678c08ea6f4a2581a051e432ed63e9b7/modeling_whisper_encoder.py
# Copyright 2026 Ant Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reusable Ming audio tower and checkpoint-compatible projector."""

import torch
from torch import nn
from transformers import WhisperConfig

from vllm.config import VllmConfig
from vllm.transformers_utils.configs.ming_v3_asr import MingAudioConfig

from .whisper import WhisperEncoder


def _get_audio_projector_params(config: MingAudioConfig) -> list[tuple[int, int, int]]:
    kernels = config.downsampler_kernel_size_list
    strides = config.downsampler_stride_list
    paddings = config.downsampler_padding_list
    if not kernels or not (len(kernels) == len(strides) == len(paddings)):
        raise ValueError("Audio convolution lists must have equal positive length")
    params = list(zip(kernels, strides, paddings))
    if any(
        kernel <= 0 or stride <= 0 or padding < 0 for kernel, stride, padding in params
    ):
        raise ValueError(
            "Audio kernels and strides must be positive and padding nonnegative"
        )
    return params


def get_audio_token_count(
    config: MingAudioConfig, num_frames: int, chunk_frames: int
) -> int:
    """Count projected frames with independent encoder positions for each chunk."""
    if num_frames <= 0:
        raise ValueError("Audio must contain at least one mel frame")
    params = _get_audio_projector_params(config)

    def output_length(length: int) -> int:
        length = (length + 1) // 2
        for kernel, stride, padding in params:
            length = (length + 2 * padding - kernel) // stride + 1
        if length <= 0:
            raise ValueError("Audio chunk is too short for its projector")
        return length

    num_chunks, tail = divmod(num_frames, chunk_frames)
    return (num_chunks * output_length(chunk_frames) if num_chunks else 0) + (
        output_length(tail) if tail else 0
    )


class MingAudioEncoder(nn.Module):
    def __init__(
        self, *, vllm_config: VllmConfig, config: MingAudioConfig, prefix: str
    ) -> None:
        super().__init__()
        encoder_config = WhisperConfig(
            num_mel_bins=config.n_mels,
            d_model=config.n_audio_state,
            encoder_attention_heads=config.n_audio_head,
            encoder_layers=config.n_audio_layer,
            encoder_ffn_dim=4 * config.n_audio_state,
            max_source_positions=config.n_audio_ctx,
            activation_function="gelu",
            pos_embed="learned",
        )
        self.encoder = WhisperEncoder(
            vllm_config=vllm_config.with_hf_config(encoder_config),
            prefix=prefix,
            enable_pp=False,
            init_in_fp32=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder([features.transpose(0, 1)])[0]


class Transpose(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.transpose(-1, -2)


class MingAudioProjector(nn.Module):
    def __init__(self, config: MingAudioConfig, hidden_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = config.n_audio_state
        for kernel, stride, padding in _get_audio_projector_params(config):
            layers.extend(
                [
                    nn.Conv1d(width, hidden_size, kernel, stride, padding),
                    nn.GELU(),
                ]
            )
            width = hidden_size
        layers.extend([Transpose(), nn.Linear(hidden_size, hidden_size)])
        self.proj = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.proj(embeddings.transpose(-1, -2))
