from __future__ import annotations

import argparse
import math
from typing import Any, Sequence

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
        return [audio[i : i + 1].float() for i in range(audio.shape[0])]
    if audio.dim() == 3:
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


def _group_count(channels: int, max_groups: int = 8) -> int:
    channels = int(channels)
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


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
    """
    Converts raw audio with arbitrary sample rates to a fixed physical Hz-time grid.
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
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        x = audio[0]
        sr = int(sr)
        win = int(round(sr * float(window_ms) / 1000.0))
        win = max(self.min_window_samples, min(self.max_window_samples, win))
        win = min(win, int(x.numel())) if x.numel() > 0 else win
        win = max(4, win)
        hop = max(1, int(round(win * self.hop_fraction)))
        n_fft = max(next_power_of_two(win), win)

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
        mag = torch.log1p(stft.abs())
        src_freqs = torch.linspace(0.0, float(sr) * 0.5, steps=mag.shape[0], device=x.device, dtype=x.dtype)
        mag = self._interp_frequency_axis(mag, src_freqs, sr=sr)
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
            spec = torch.stack(scale_specs, dim=0)
            freq_coord = self.freq_coord.to(device=spec.device, dtype=spec.dtype).expand(1, -1, self.time_bins)
            time_coord = self.time_coord.to(device=spec.device, dtype=spec.dtype).expand(1, self.freq_bins, -1)
            specs.append(torch.cat([spec, freq_coord, time_coord], dim=0))
        return torch.stack(specs, dim=0)


class SineLayer(nn.Module):
    """SIREN layer: Linear followed by sin(omega_0 * x)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class CoordinateSIREN(nn.Module):
    """
    Learns continuous frequency-time coordinate features.
    """

    def __init__(
        self,
        out_channels: int = 16,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            SineLayer(2, hidden_features, is_first=True, omega_0=first_omega_0)
        ]
        for _ in range(int(hidden_layers)):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        final = nn.Linear(hidden_features, out_channels)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
            final.weight.uniform_(-bound, bound)
            if final.bias is not None:
                final.bias.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.out_channels = int(out_channels)

    def forward(self, coord_channels: torch.Tensor) -> torch.Tensor:
        if coord_channels.dim() != 4 or coord_channels.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, F, T] coordinates, got {tuple(coord_channels.shape)}")
        b, _, f, t = coord_channels.shape
        coords = coord_channels.permute(0, 2, 3, 1).reshape(b * f * t, 2)
        y = self.net(coords)
        return y.view(b, f, t, self.out_channels).permute(0, 3, 1, 2).contiguous()


class SpectralConv2d(nn.Module):
    """2D Fourier Neural Operator spectral convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_freq: int = 24,
        modes_time: int = 24,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_freq = int(modes_freq)
        self.modes_time = int(modes_time)
        scale = 1.0 / math.sqrt(max(1, self.in_channels * self.out_channels))
        self.weight_pos_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_freq, modes_time))
        self.weight_pos_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_freq, modes_time))
        self.weight_neg_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_freq, modes_time))
        self.weight_neg_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_freq, modes_time))

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcft,coft->boft", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f, t = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            b,
            self.out_channels,
            f,
            t // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )

        mf = min(self.modes_freq, f)
        mt = min(self.modes_time, t // 2 + 1)
        w_pos = torch.complex(
            self.weight_pos_real[:, :, :mf, :mt],
            self.weight_pos_imag[:, :, :mf, :mt],
        )
        out_ft[:, :, :mf, :mt] = self._compl_mul2d(x_ft[:, :, :mf, :mt], w_pos)

        if mf > 0 and f > mf:
            w_neg = torch.complex(
                self.weight_neg_real[:, :, :mf, :mt],
                self.weight_neg_imag[:, :, :mf, :mt],
            )
            out_ft[:, :, -mf:, :mt] = self._compl_mul2d(x_ft[:, :, -mf:, :mt], w_neg)

        return torch.fft.irfft2(out_ft, s=(f, t), norm="ortho")


class FNOBlock2d(nn.Module):
    """Residual FNO block: global spectral mixing plus local 1x1 channel mixing."""

    def __init__(
        self,
        channels: int,
        modes_freq: int = 24,
        modes_time: int = 24,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_freq=modes_freq, modes_time=modes_time)
        self.local = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.local(x)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return x + y


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int] = (1, 1), dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != (1, 1):
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_ch), out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class FnoSirenSpectralEncoder(nn.Module):
    """
    CNN encoder augmented with:
    - Coordinate SIREN features over the physical frequency-time grid.
    - FNO blocks for global spectral mixing in the learned feature maps.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 48,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        fno_modes_freq: int = 24,
        fno_modes_time: int = 24,
        siren_channels: int = 16,
        siren_hidden_features: int = 64,
        siren_hidden_layers: int = 2,
        siren_first_omega_0: float = 30.0,
        siren_hidden_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.in_channels = int(in_channels)
        self.siren_channels = int(siren_channels)
        self.coord_siren = CoordinateSIREN(
            out_channels=siren_channels,
            hidden_features=siren_hidden_features,
            hidden_layers=siren_hidden_layers,
            first_omega_0=siren_first_omega_0,
            hidden_omega_0=siren_hidden_omega_0,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + siren_channels, c, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            FNOBlock2d(c, modes_freq=fno_modes_freq, modes_time=fno_modes_time, dropout=dropout),
            ConvBlock(c, c * 2, stride=(2, 2), dropout=dropout),
            FNOBlock2d(c * 2, modes_freq=max(4, fno_modes_freq // 2), modes_time=max(4, fno_modes_time // 2), dropout=dropout),
            ConvBlock(c * 2, c * 4, stride=(2, 2), dropout=dropout),
            FNOBlock2d(c * 4, modes_freq=max(4, fno_modes_freq // 4), modes_time=max(4, fno_modes_time // 4), dropout=dropout),
            ConvBlock(c * 4, c * 4, stride=(2, 2), dropout=dropout),
            FNOBlock2d(c * 4, modes_freq=max(4, fno_modes_freq // 8), modes_time=max(4, fno_modes_time // 8), dropout=dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(c * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 2:
            raise ValueError("Expected the last two frontend channels to be frequency and time coordinates.")
        coord = x[:, -2:, :, :]
        coord_features = self.coord_siren(coord)
        x = torch.cat([x, coord_features], dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        mean = x.mean(dim=(-2, -1))
        maxv = x.amax(dim=(-2, -1))
        return self.proj(torch.cat([mean, maxv], dim=-1))


# ADD CNN 


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
        fno_modes_freq: int = 24,
        fno_modes_time: int = 24,
        siren_channels: int = 16,
        siren_hidden_features: int = 64,
        siren_hidden_layers: int = 2,
        siren_first_omega_0: float = 30.0,
        siren_hidden_omega_0: float = 30.0,
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
        self.encoder = FnoSirenSpectralEncoder(
            in_channels=in_ch,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            fno_modes_freq=fno_modes_freq,
            fno_modes_time=fno_modes_time,
            siren_channels=siren_channels,
            siren_hidden_features=siren_hidden_features,
            siren_hidden_layers=siren_hidden_layers,
            siren_first_omega_0=siren_first_omega_0,
            siren_hidden_omega_0=siren_hidden_omega_0,
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
    p = argparse.ArgumentParser(description="Test FNO/SIREN bioacoustic model.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--sample-rates", type=int, nargs="+", default=[250, 22050])
    p.add_argument("--classes-per-dataset", type=int, default=8)
    p.add_argument("--freq-bins", type=int, default=160)
    p.add_argument("--time-bins", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--fno-modes-freq", type=int, default=24)
    p.add_argument("--fno-modes-time", type=int, default=24)
    p.add_argument("--siren-channels", type=int, default=16)
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

    model = SRInvariantBioacousticClassifier(
        classes_per_dataset=args.classes_per_dataset,
        freq_bins=args.freq_bins,
        time_bins=args.time_bins,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        fno_modes_freq=args.fno_modes_freq,
        fno_modes_time=args.fno_modes_time,
        siren_channels=args.siren_channels,
    )
    out = model(audios, rates, grl_lambda=1.0)
    print("embedding", tuple(out["embedding"].shape))
    for name, logits in out["logits"].items():
        print(name, tuple(logits.shape))
    print("sr_logits", tuple(out["sr_logits"].shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
