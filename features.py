# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Feature-related functions.

References:
    - https://github.com/bigpon/QPPWG
    - https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG
    - https://github.com/k2kobayashi/sprocket

"""

import sys
from logging import getLogger

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.nn.functional import interpolate

# A logger for this file
logger = getLogger(__name__)

import copy
def convert_continuos_f0(f0):
    """Convert F0 to continuous F0
    Args:
        f0 (ndarray): original f0 sequence with the shape (T)
    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)
    # get start and end of f0
    if (f0 == 0).all():
        logger.warn("all of the f0 values are 0.")
        return uv, f0, False
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    # padding start and end of f0 sequence
    cf0 = copy.deepcopy(f0)
    start_idx = np.where(cf0 == start_f0)[0][0]
    end_idx = np.where(cf0 == end_f0)[0][-1]
    cf0[:start_idx] = start_f0
    cf0[end_idx:] = end_f0
    # get non-zero frame index
    nz_frames = np.where(cf0 != 0)[0]
    # perform linear interpolation
    f = interp1d(nz_frames, cf0[nz_frames])
    cf0 = f(np.arange(0, cf0.shape[0]))

    return uv, cf0, True


def validate_length(xs, ys=None, hop_size=None):
    """Validate length

    Args:
        xs (ndarray): numpy array of features
        ys (ndarray): numpy array of audios
        hop_size (int): upsampling factor

    Returns:
        (ndarray): length adjusted features

    """
    min_len_x = min([x.shape[0] for x in xs])
    if ys is not None:
        min_len_y = min([y.shape[0] for y in ys])
        if min_len_y < min_len_x * hop_size:
            min_len_x = min_len_y // hop_size
        if min_len_y > min_len_x * hop_size:
            min_len_y = min_len_x * hop_size
        ys = [y[:min_len_y] for y in ys]
    xs = [x[:min_len_x] for x in xs]

    return xs + ys if ys is not None else xs


def dilated_factor(batch_f0, fs, dense_factor):
    """Pitch-dependent dilated factor

    Args:
        batch_f0 (ndarray): the f0 sequence (T)
        fs (int): sampling rate
        dense_factor (int): the number of taps in one cycle

    Return:
        dilated_factors(np array):
            float array of the pitch-dependent dilated factors (T)

    """
    batch_f0[batch_f0 == 0] = fs / dense_factor
    dilated_factors = np.ones(batch_f0.shape) * fs / dense_factor / batch_f0
    assert np.all(dilated_factors > 0)

    return dilated_factors


class SignalGenerator:
    """Input signal generator module."""

    def __init__(
        self,
        sample_rate=16000,
        hop_size=320,
        sine_amp=0.1,
        noise_amp=0.003,
        signal_types=["sine"],
    ):
        """Initialize WaveNetResidualBlock module.

        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of input F0.
            sine_amp (float): Sine amplitude for NSF-based sine generation.
            noise_amp (float): Noise amplitude for NSF-based sine generation.
            signal_types (list): List of input signal types for generator.

        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.signal_types = signal_types
        self.sine_amp = sine_amp
        self.noise_amp = noise_amp

        for signal_type in signal_types:
            if not signal_type in ["noise", "sine", "sines", "uv"]:
                logger.error(f"{signal_type} is not supported type for generator input.")
                raise Exception(f"{signal_type} is not supported type for generator input.")
        logger.info(f"Use {signal_types} for generator input signals.")

    @torch.no_grad()
    def __call__(self, f0):
        signals = []
        for typ in self.signal_types:
            if "noise" == typ:
                signals.append(self.random_noise(f0))
            if "sine" == typ:
                signals.append(self.sinusoid(f0))
            if "sines" == typ:
                signals.append(self.sinusoids(f0))
            if "uv" == typ:
                signals.append(self.vuv_binary(f0))

        input_batch = signals[0]
        for signal in signals[1:]:
            input_batch = torch.cat([input_batch, signal], axis=1)

        return input_batch

    @torch.no_grad()
    def random_noise(self, f0):
        """Calculate noise signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Gaussian noise signals (B, 1, T).

        """
        B, _, T = f0.size()
        noise = torch.randn((B, 1, T * self.hop_size), device=f0.device)

        return noise

    @torch.no_grad()
    def sinusoid(self, f0):
        """Calculate sine signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Sines generated following NSF (B, 1, T).

        """
        B, _, T = f0.size()
        vuv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        radious = (interpolate(f0, T * self.hop_size) / self.sample_rate) % 1
        sine = vuv * torch.sin(torch.cumsum(radious, dim=2) * 2 * np.pi) * self.sine_amp
        if self.noise_amp > 0:
            noise_amp = vuv * self.noise_amp + (1.0 - vuv) * self.noise_amp / 3.0
            noise = torch.randn((B, 1, T * self.hop_size), device=f0.device) * noise_amp
            sine = sine + noise

        return sine

    @torch.no_grad()
    def sinusoids(self, f0):
        """Calculate sines.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Sines generated following NSF (B, 1, T).

        """
        B, _, T = f0.size()
        vuv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        f0 = interpolate(f0, T * self.hop_size)
        sines = torch.zeros_like(f0, device=f0.device)
        harmonics = 5  # currently only fixed number of harmonics is supported
        for i in range(harmonics):
            radious = (f0 * (i + 1) / self.sample_rate) % 1
            sines += torch.sin(torch.cumsum(radious, dim=2) * 2 * np.pi)
        sines = self.sine_amp * sines * vuv / harmonics
        if self.noise_amp > 0:
            noise_amp = vuv * self.noise_amp + (1.0 - vuv) * self.noise_amp / 3.0
            noise = torch.randn((B, 1, T * self.hop_size), device=f0.device) * noise_amp
            sines = sines + noise

        return sines

    @torch.no_grad()
    def vuv_binary(self, f0):
        """Calculate V/UV binary sequences.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: V/UV binary sequences (B, 1, T).

        """
        _, _, T = f0.size()
        uv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)

        return uv
