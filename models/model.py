# Copyright 2020 LMNT, Inc. All Rights Reserved.
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
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from math import log as ln

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond:  # conditional model
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else:  # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None:  # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        if self.params.unconditional:  # use unconditional model
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
                          uncond=params.unconditional)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram=None):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
               (spectrogram is not None and self.spectrogram_upsampler is not None)
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler:  # use conditional model
            spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


class Conv1dW(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, noise_level):
        """
    Arguments:
      x:
          (shape: [N,C,T], dtype: float32)
      noise_level:
          (shape: [N], dtype: float32)

    Returns:
      noise_level:
          (shape: [N,C,T], dtype: float32)
    """
        N = x.shape[0]
        T = x.shape[2]
        return (x + self._build_encoding(noise_level)[:, :, None])

    def _build_encoding(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FiLM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoding = PositionalEncoding(input_size)
        self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.input_conv.bias)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, noise_scale):
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.encoding(x, noise_scale)
        shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
        return shift, scale


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.block1 = Conv1dW(input_size, hidden_size, 1)
        self.block2 = nn.ModuleList([
            Conv1dW(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
            Conv1dW(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1])
        ])
        self.block3 = nn.ModuleList([
            Conv1dW(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
            Conv1dW(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3])
        ])

    def forward(self, x, film_shift, film_scale):
        block1 = F.interpolate(x, size=x.shape[-1] * self.factor)
        block1 = self.block1(block1)

        block2 = F.leaky_relu(x, 0.2)
        block2 = F.interpolate(block2, size=x.shape[-1] * self.factor)
        block2 = self.block2[0](block2)
        block2 = film_shift + film_scale * block2
        block2 = F.leaky_relu(block2, 0.2)
        block2 = self.block2[1](block2)

        x = block1 + block2

        block3 = film_shift + film_scale * x
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[0](block3)
        block3 = film_shift + film_scale * block3
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[1](block3)

        x = x + block3
        return x


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1dW(input_size, hidden_size, 1)
        self.conv = nn.ModuleList([
            Conv1dW(input_size, hidden_size, 3, dilation=1, padding=1),
            Conv1dW(hidden_size, hidden_size, 3, dilation=2, padding=2),
            Conv1dW(hidden_size, hidden_size, 3, dilation=4, padding=4),
        ])

    def forward(self, x):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        for layer in self.conv:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual


class WaveGrad(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.downsample = nn.ModuleList([
            Conv1dW(1, 32, 5, padding=2),
            DBlock(32, 128, 2),
            DBlock(128, 128, 2),
            DBlock(128, 256, 3),
            DBlock(256, 512, 5),
        ])
        self.film = nn.ModuleList([
            FiLM(32, 128),
            FiLM(128, 128),
            FiLM(128, 256),
            FiLM(256, 512),
            FiLM(512, 512),
        ])
        self.upsample = nn.ModuleList([
            UBlock(768, 512, 5, [1, 2, 1, 2]),
            UBlock(512, 512, 5, [1, 2, 1, 2]),
            UBlock(512, 256, 3, [1, 2, 4, 8]),
            UBlock(256, 128, 2, [1, 2, 4, 8]),
            UBlock(128, 128, 2, [1, 2, 4, 8]),
        ])
        self.first_conv = Conv1d(128, 768, 3, padding=1)
        self.last_conv = Conv1d(128, 1, 3, padding=1)

    def forward(self, audio, spectrogram, noise_scale):
        x = audio.unsqueeze(1)
        downsampled = []
        for film, layer in zip(self.film, self.downsample):
            x = layer(x)
            downsampled.append(film(x, noise_scale))
        print('Downsample shape:', x.shape)
        x = self.first_conv(spectrogram)
        for layer, (film_shift, film_scale) in zip(self.upsample, reversed(downsampled)):
            x = layer(x, film_shift, film_scale)
        x = self.last_conv(x)
        return x

    class Discriminator(nn.Module):

        def __init__(self, params):
            super().__init__()
            self.params = params
            self.downsample = nn.ModuleList([
                Conv1dW(1, 32, 5, padding=2),
                DBlock(32, 128, 2),
                DBlock(128, 128, 2),
                DBlock(128, 256, 3),
                DBlock(256, 512, 5),
            ])
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(512, 2)

        def forward(self, audio, spectrogram, noise_scale):
            x = audio.unsqueeze(1)
            for layer in self.downsample:
                x = layer(x)
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
