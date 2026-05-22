"""
Simplified 1D Fourier Neural Operator for audio - CUDA stable version
"""
import torch
import torch.nn as nn
from typing import Optional


class FNOLayer(nn.Module):
    """
    1D Fourier Neural Operator layer.

    Follows the standard FNO formulation:
        u_{l+1} = σ( K(u_l) + W(u_l) )

    where K is spectral convolution (FFT path) and W is the
    pointwise bypass (time-domain residual path).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes

        # Spectral weights: both dims are out_channels because we lift
        # first, so x entering the spectral conv already has out_channels.
        # Shape: (out_channels, out_channels, n_modes)
        self.spectral_weights = nn.Parameter(
            torch.randn(out_channels, out_channels, n_modes, dtype=torch.complex64) * 0.02
        )

        # Lifting: project from in_channels -> out_channels once, up front.
        self.lifting = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        # Bypass (W path): pointwise linear in time domain, applied to the
        # already-lifted representation so both paths share the same shape.
        self.bypass = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        # Optional output projection (keeps the layer composable).
        self.projection = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         x: (batch, in_channels, seq_len)
    #     Returns:
    #         (batch, out_channels, seq_len)
    #     """
    #     batch_size, _, seq_len = x.shape

    #     # --- Lift once: (batch, in_channels, seq_len) -> (batch, out_channels, seq_len)
    #     x_lifted = self.lifting(x)

    #     # --- Spectral (K) path ---
    #     # FFT along time axis
    #     x_fft = torch.fft.rfft(x_lifted, dim=-1, norm='ortho')
    #     n_freqs = x_fft.shape[-1]
    #     modes = min(self.n_modes, n_freqs)

    #     # Spectral convolution on the truncated modes
    #     # weights: (out_ch, out_ch, modes), x_fft: (batch, out_ch, modes)
    #     x_spectral = torch.einsum(
    #         'oim,bim->bom',
    #         self.spectral_weights[..., :modes],
    #         x_fft[..., :modes],
    #     )

    #     # Zero-pad back to full frequency resolution; use input dtype to
    #     # stay compatible with AMP / mixed precision.
    #     x_fft_padded = torch.zeros(
    #         batch_size, self.out_channels, n_freqs,
    #         dtype=x_fft.dtype, device=x.device,
    #     )
    #     x_fft_padded[..., :modes] = x_spectral

    #     # Back to time domain
    #     x_k = torch.fft.irfft(x_fft_padded, n=seq_len, dim=-1, norm='ortho')

    #     # --- Bypass (W) path ---
    #     x_w = self.bypass(x_lifted)

    #     # --- Combine, activate, project ---
    #     x_out = torch.relu(x_k + x_w)
    #     x_out = self.projection(x_out)

    #     return x_out
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = x.shape
        x_lifted = self.lifting(x)

        # --- Spectral (K) path ---
        # CRITICAL FIX: Changed norm to 'forward' to ensure discretization invariance
        x_fft = torch.fft.rfft(x_lifted, dim=-1, norm='forward')
        n_freqs = x_fft.shape[-1]
        modes = min(self.n_modes, n_freqs)

        x_spectral = torch.einsum(
            'oim,bim->bom',
            self.spectral_weights[..., :modes],
            x_fft[..., :modes],
        )

        x_fft_padded = torch.zeros(
            batch_size, self.out_channels, n_freqs,
            dtype=x_fft.dtype, device=x.device,
        )
        x_fft_padded[..., :modes] = x_spectral

        # CRITICAL FIX: Matching norm='forward' requires norm='forward' on inverse
        x_k = torch.fft.irfft(x_fft_padded, n=seq_len, dim=-1, norm='forward')

        # --- Bypass (W) path ---
        x_w = self.bypass(x_lifted)

        x_out = torch.relu(x_k + x_w)
        x_out = self.projection(x_out)

        return x_out

class FNOStack(nn.Module):
    """Stack of FNO layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 1,
        n_modes: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.layers.append(FNOLayer(in_ch, out_channels, n_modes))

    def forward(self, x: torch.Tensor, sample_rate: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
            sample_rate: accepted for API compatibility, currently unused.
        Returns:
            (batch, out_channels, seq_len)
        """
        for layer in self.layers:
            x = layer(x)
        return x