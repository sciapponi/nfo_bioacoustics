#!/usr/bin/env python3
"""Sample-rate invariant bioacoustic classifier.

Design goals:
  - No torchaudio.
  - Sample rate is used only to construct physical time/frequency features.
  - The classifier head does not receive a sample-rate embedding.
  - A gradient-reversal sample-rate adversary can be used during training to
    discourage the embedding from encoding nominal sample rate.

Input:
  audio: Tensor[B, 1, T], Tensor[B, T], or list of Tensor[1, T] / Tensor[T]
  sample_rate: int, list[int], or Tensor[B]

Output:
  {
    "logits": {dataset_name: Tensor[B, classes_per_dataset]},
    "embedding": Tensor[B, embedding_dim],
    "sr_logits": Tensor[B, num_sr_buckets] or None,
  }
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DATASET_NAMES = ("whales", "frogs", "birds", "bats")


def _as_audio_list(audio: Any) -> list[torch.Tensor]:
    if isinstance(audio, (list, tuple)):
        out = []
        for x in audio:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            elif x.dim() == 2:
                pass
            elif x.dim() == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                raise ValueError(f"Unsupported audio item shape: {tuple(x.shape)}")
            out.append(x.float())
        return out

    if audio.dim() == 2:
        # Tensor[B, T]
        return [audio[i : i + 1].float() for i in range(audio.shape[0])]
    if audio.dim() == 3:
        # Tensor[B, C, T]
        return [audio[i].float() for i in range(audio.shape[0])]
    raise ValueError(f"Unsupported audio shape: {tuple(audio.shape)}")


def _as_sr_list(sample_rate: Any, batch_size: int) -> list[int]:
    if isinstance(sample_rate, int):
        return [int(sample_rate)] * batch_size
    if isinstance(sample_rate, float):
        return [int(round(sample_rate))] * batch_size
    if isinstance(sample_rate, torch.Tensor):
        if sample_rate.ndim == 0:
            return [int(sample_rate.item())] * batch_size
        return [int(x) for x in sample_rate.detach().cpu().tolist()]
    if isinstance(sample_rate, (list, tuple)):
        return [int(x) for x in sample_rate]
    raise ValueError(f"Unsupported sample_rate type: {type(sample_rate)}")


def next_power_of_two(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def gradient_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReverseFunction.apply(x, float(lambd))


class PhysicalMultiScaleSTFT(nn.Module):
    """Multi-scale log-STFT frontend on a fixed physical frequency grid.

    The STFT windows are specified in milliseconds, then converted to samples
    using each sample's own sample rate. Frequency bins are interpolated onto a
    common log-Hz grid. This avoids fixed-sample-count filters, but it does not
    pretend missing bandwidth exists: frequencies above Nyquist are set to zero.
    The training script uses dual-view consistency and SR adversarial loss to
    discourage the model from using those missing-bandwidth patterns as shortcuts.
    """

    def __init__(
        self,
        freq_bins: int = 160,
        time_bins: int = 256,
        freq_min_hz: float = 1.0,
        freq_max_hz: float = 120_000.0,
        window_ms: Sequence[float] = (32.0, 64.0, 128.0),
        hop_fraction: float = 0.25,
        min_window_samples: int = 16,
        max_window_samples: int = 8192,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.freq_bins = int(freq_bins)
        self.time_bins = int(time_bins)
        self.freq_min_hz = float(freq_min_hz)
        self.freq_max_hz = float(freq_max_hz)
        self.window_ms = tuple(float(x) for x in window_ms)
        self.hop_fraction = float(hop_fraction)
        self.min_window_samples = int(min_window_samples)
        self.max_window_samples = int(max_window_samples)
        self.eps = float(eps)

        target_freqs = torch.logspace(
            math.log10(self.freq_min_hz),
            math.log10(self.freq_max_hz),
            steps=self.freq_bins,
        )
        freq_coord = torch.linspace(-1.0, 1.0, steps=self.freq_bins).view(1, self.freq_bins, 1)
        time_coord = torch.linspace(-1.0, 1.0, steps=self.time_bins).view(1, 1, self.time_bins)
        self.register_buffer("target_freqs", target_freqs, persistent=False)
        self.register_buffer("freq_coord", freq_coord, persistent=False)
        self.register_buffer("time_coord", time_coord, persistent=False)

    def _interp_frequency_axis(
        self,
        spec: torch.Tensor,
        src_freqs: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        # spec: [F_src, T]
        target = self.target_freqs.to(device=spec.device, dtype=spec.dtype)
        src = src_freqs.to(device=spec.device, dtype=spec.dtype)

        idx_hi = torch.searchsorted(src, target).clamp(min=1, max=src.numel() - 1)
        idx_lo = idx_hi - 1
        f_lo = src[idx_lo]
        f_hi = src[idx_hi]
        denom = (f_hi - f_lo).clamp_min(self.eps)
        alpha = ((target - f_lo) / denom).clamp(0.0, 1.0)

        lo = spec.index_select(0, idx_lo)
        hi = spec.index_select(0, idx_hi)
        out = lo * (1.0 - alpha[:, None]) + hi * alpha[:, None]

        valid = (target <= (float(sr) * 0.5)) & (target >= src[0])
        out = out * valid[:, None].to(out.dtype)
        return out

    def _normalize_valid_region(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        # x: [F, T]. Normalize only frequencies available under Nyquist.
        target = self.target_freqs.to(device=x.device, dtype=x.dtype)
        valid = target <= (float(sr) * 0.5)
        if not bool(valid.any()):
            return x
        vals = x[valid]
        mean = vals.mean()
        std = vals.std().clamp_min(1e-5)
        x = (x - mean) / std
        x = x * valid[:, None].to(x.dtype)
        return x

    def _one_scale(self, audio: torch.Tensor, sr: int, window_ms: float) -> torch.Tensor:
        # audio: [C, T]
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        x = audio[0]
        sr = int(sr)
        win = int(round(sr * float(window_ms) / 1000.0))
        win = max(self.min_window_samples, min(self.max_window_samples, win))
        win = min(win, int(x.numel())) if x.numel() > 0 else win
        win = max(4, win)
        hop = max(1, int(round(win * self.hop_fraction)))
        n_fft = next_power_of_two(win)
        n_fft = max(n_fft, win)

        if x.numel() < win:
            x = F.pad(x, (0, win - x.numel()))

        window = torch.hann_window(win, device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=True,
            return_complex=True,
        )
        mag = torch.log1p(stft.abs())  # [F_src, T_src]
        src_freqs = torch.linspace(0.0, float(sr) * 0.5, steps=mag.shape[0], device=x.device, dtype=x.dtype)
        mag = self._interp_frequency_axis(mag, src_freqs, sr=sr)  # [F_target, T_src]
        mag = self._normalize_valid_region(mag, sr=sr)
        mag = F.interpolate(
            mag.unsqueeze(0).unsqueeze(0),
            size=(self.freq_bins, self.time_bins),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return mag

    def forward(self, audio: Any, sample_rate: Any) -> torch.Tensor:
        audio_list = _as_audio_list(audio)
        sr_list = _as_sr_list(sample_rate, len(audio_list))
        specs = []
        for x, sr in zip(audio_list, sr_list):
            scale_specs = [self._one_scale(x, sr=sr, window_ms=w) for w in self.window_ms]
            spec = torch.stack(scale_specs, dim=0)  # [S, F, T]
            freq_coord = self.freq_coord.to(device=spec.device, dtype=spec.dtype).expand(1, -1, self.time_bins)
            time_coord = self.time_coord.to(device=spec.device, dtype=spec.dtype).expand(1, self.freq_bins, -1)
            specs.append(torch.cat([spec, freq_coord, time_coord], dim=0))
        return torch.stack(specs, dim=0)  # [B, C, F, T]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int] = (1, 1), dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != (1, 1):
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class SpectralEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 48,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ConvBlock(c, c, stride=(1, 1), dropout=dropout),
            ConvBlock(c, c * 2, stride=(2, 2), dropout=dropout),
            ConvBlock(c * 2, c * 4, stride=(2, 2), dropout=dropout),
            ConvBlock(c * 4, c * 4, stride=(2, 2), dropout=dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(c * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        mean = x.mean(dim=(-2, -1))
        maxv = x.amax(dim=(-2, -1))
        return self.proj(torch.cat([mean, maxv], dim=-1))


class SRInvariantBioacousticClassifier(nn.Module):
    def __init__(
        self,
        datasets: Sequence[str] = DATASET_NAMES,
        classes_per_dataset: int = 8,
        freq_bins: int = 160,
        time_bins: int = 256,
        freq_min_hz: float = 1.0,
        freq_max_hz: float = 120_000.0,
        window_ms: Sequence[float] = (32.0, 64.0, 128.0),
        base_channels: int = 48,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        sr_adversary_buckets: int = 8,
    ) -> None:
        super().__init__()
        self.datasets = tuple(str(x) for x in datasets)
        self.classes_per_dataset = int(classes_per_dataset)
        self.frontend = PhysicalMultiScaleSTFT(
            freq_bins=freq_bins,
            time_bins=time_bins,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            window_ms=window_ms,
        )
        in_ch = len(tuple(window_ms)) + 2
        self.encoder = SpectralEncoder(
            in_channels=in_ch,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(embedding_dim, self.classes_per_dataset) for name in self.datasets}
        )
        self.sr_adversary = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, int(sr_adversary_buckets)),
        )

    def encode(self, audio: Any, sample_rate: Any) -> torch.Tensor:
        spec = self.frontend(audio, sample_rate)
        return self.encoder(spec)

    def forward(
        self,
        audio: Any,
        sample_rate: Any,
        grl_lambda: float = 0.0,
        return_sr_logits: bool = True,
    ) -> dict[str, Any]:
        z = self.encode(audio, sample_rate)
        logits = {name: head(z) for name, head in self.heads.items()}
        sr_logits = None
        if return_sr_logits:
            z_adv = gradient_reverse(z, grl_lambda) if grl_lambda > 0 else z.detach()
            sr_logits = self.sr_adversary(z_adv)
        return {"logits": logits, "embedding": z, "sr_logits": sr_logits}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test the sample-rate invariant bioacoustic model.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--sample-rates", type=int, nargs="+", default=[250, 22050])
    p.add_argument("--classes-per-dataset", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    audios = []
    rates = []
    for i in range(args.batch_size):
        sr = args.sample_rates[i % len(args.sample_rates)]
        t = torch.linspace(0, args.seconds, int(sr * args.seconds))
        x = 0.1 * torch.sin(2 * math.pi * 20.0 * t)
        audios.append(x.unsqueeze(0))
        rates.append(sr)
    model = SRInvariantBioacousticClassifier(classes_per_dataset=args.classes_per_dataset)
    out = model(audios, rates, grl_lambda=1.0)
    print("embedding", tuple(out["embedding"].shape))
    for name, logits in out["logits"].items():
        print(name, tuple(logits.shape))
    print("sr_logits", tuple(out["sr_logits"].shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
