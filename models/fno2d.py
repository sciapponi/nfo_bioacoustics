"""
2D Fourier Neural Operator for spectrograms.

The model processes 2D Time-Frequency spectrograms as continuous surfaces,
using FFT over both time and frequency axes.
"""
import torch
import torch.nn as nn
from typing import Optional


class FNO2DLayer(nn.Module):
    """
    2D Fourier Neural Operator layer for spectrograms.
    
    Treats spectrogram as a continuous 2D surface:
    - Time axis (x): temporal evolution
    - Frequency axis (y): spectral structure
    
    Spectral weights shape: (out_channels, in_channels, modes_time, modes_freq)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_time: int = 32,
        n_modes_freq: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_time = n_modes_time
        self.n_modes_freq = n_modes_freq

        # Spectral weights for 2D convolution in Fourier space
        # Shape: (out_channels, out_channels, modes_time, modes_freq)
        # Already lifted to out_channels in __init__
        self.spectral_weights = nn.Parameter(
            torch.randn(
                out_channels, out_channels, n_modes_time, n_modes_freq,
                dtype=torch.complex64
            ) * 0.02
        )

        # Lifting: project from in_channels -> out_channels once
        self.lifting = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Bypass (W path): pointwise linear in spatial domain
        self.bypass = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        # Optional output projection
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, height, width) spectrogram
               where height is frequency bins, width is time steps
        Returns:
            (batch, out_channels, height, width)
        """
        batch_size, _, h, w = x.shape

        # --- Lift once ---
        x_lifted = self.lifting(x)

        # --- Spectral (K) path: 2D FFT ---
        # FFT over both spatial dimensions
        x_fft = torch.fft.rfft2(x_lifted, norm='forward')
        
        n_freqs_time = x_fft.shape[-2]  # modes in time direction
        n_freqs_freq = x_fft.shape[-1]   # modes in frequency direction
        
        modes_time = min(self.n_modes_time, n_freqs_time)
        modes_freq = min(self.n_modes_freq, n_freqs_freq)

        # Spectral convolution: truncate to learned modes
        # Weights: (out_ch, out_ch, modes_t, modes_f)
        # x_fft: (batch, out_ch, modes_t, modes_f)
        # Spectral convolution: einsum over input channel and both modal axes
        # weights: (out_ch, in_ch, modes_time, modes_freq)
        # x_fft:   (batch, in_ch, modes_time, modes_freq)
        # result:  (batch, out_ch, modes_time, modes_freq)
        x_spectral = torch.einsum(
            'o i t f, b i t f -> b o t f',
            self.spectral_weights[..., :modes_time, :modes_freq],
            x_fft[..., :modes_time, :modes_freq],
        )

        # Pad back to full frequency resolution
        x_fft_padded = torch.zeros(
            batch_size, self.out_channels, n_freqs_time, n_freqs_freq,
            dtype=x_fft.dtype, device=x.device,
        )
        x_fft_padded[..., :modes_time, :modes_freq] = x_spectral

        # Back to spatial domain
        x_k = torch.fft.irfft2(x_fft_padded, s=(h, w), norm='forward')

        # --- Bypass (W) path ---
        x_w = self.bypass(x_lifted)

        # --- Combine, activate, project ---
        x_out = torch.relu(x_k + x_w)
        x_out = self.projection(x_out)

        return x_out


class FNO2DStack(nn.Module):
    """Stack of 2D FNO layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 2,
        n_modes_time: int = 32,
        n_modes_freq: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.layers.append(
                FNO2DLayer(
                    in_ch, out_channels, n_modes_time, n_modes_freq
                )
            )

    def forward(
        self, x: torch.Tensor, sample_rate: Optional[float] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, height, width) 2D spectrogram
            sample_rate: ignored (for API compatibility)
        Returns:
            (batch, out_channels, height, width)
        """
        for layer in self.layers:
            x = layer(x)
        return x
