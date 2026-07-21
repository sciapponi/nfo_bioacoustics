from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Generic tensor/audio helpers. No external audio backend is used in this module.
# -----------------------------------------------------------------------------


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_report(module: nn.Module) -> dict[str, int]:
    report = {name: count_trainable_parameters(child) for name, child in module.named_children()}
    report["total"] = count_trainable_parameters(module)
    return report


def _as_audio_list(audio: Any) -> list[torch.Tensor]:
    if isinstance(audio, (list, tuple)):
        out: list[torch.Tensor] = []
        for x in audio:
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
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
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
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


def group_count(channels: int, max_groups: int = 8) -> int:
    channels = int(channels)
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def fft_resample_1d(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Anti-aliased 1D resampling for one mono waveform [1, N] or [N].

    This version deliberately avoids CUDA FFT. The previous implementation used
    torch.fft.rfft/irfft on the input device; during the staged WhaleNet feature
    extraction this can trigger cuFFT internal errors on some CUDA/PyTorch
    combinations. The default path uses scipy.signal.resample_poly on CPU and
    returns the tensor to the original device. That is stable for offline feature
    extraction and for fixed front-end preprocessing because gradients with
    respect to the input waveform are not needed.
    """
    if int(orig_sr) == int(target_sr):
        return wav

    squeeze = False
    if wav.dim() == 1:
        wav_in = wav.unsqueeze(0)
        squeeze = True
    elif wav.dim() == 2:
        wav_in = wav
    else:
        raise ValueError(f"Expected waveform with shape [N] or [C, N], got {tuple(wav.shape)}")

    n = int(wav_in.shape[-1])
    if n <= 1:
        out = wav_in.clone()
        return out.squeeze(0) if squeeze else out

    new_n = max(1, int(round(n * float(target_sr) / float(orig_sr))))
    if new_n == n:
        return wav

    device = wav_in.device
    dtype = wav_in.dtype

    # Stable anti-aliased polyphase path. scipy is listed in requirements.
    try:
        from math import gcd
        from scipy.signal import resample_poly  # type: ignore

        g = gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        x_np = wav_in.detach().cpu().float().numpy()
        y_np = resample_poly(x_np, up=up, down=down, axis=-1)
        if y_np.shape[-1] > new_n:
            y_np = y_np[..., :new_n]
        elif y_np.shape[-1] < new_n:
            pad = new_n - y_np.shape[-1]
            y_np = np.pad(y_np, [(0, 0)] * (y_np.ndim - 1) + [(0, pad)], mode="constant")
        y = torch.from_numpy(np.asarray(y_np)).to(device=device, dtype=dtype)
        return y.squeeze(0) if squeeze else y
    except Exception:
        # Last-resort fallback that avoids CUDA FFT. It is not as good as
        # polyphase resampling, but it is stable and keeps the script running if
        # scipy is unavailable or rejects a file.
        x = wav_in.detach().float().cpu().unsqueeze(0)
        y = F.interpolate(x, size=new_n, mode="linear", align_corners=False).squeeze(0)
        y = y.to(device=device, dtype=dtype)
        return y.squeeze(0) if squeeze else y


def lowpass_fft_1d(wav: torch.Tensor, sr: int, cutoff_hz: float) -> torch.Tensor:
    """Ideal FFT low-pass used for robustness evaluation and controlled augmentation."""
    if cutoff_hz <= 0:
        return torch.zeros_like(wav)
    nyq = 0.5 * float(sr)
    if cutoff_hz >= nyq:
        return wav
    if wav.dim() == 1:
        x = wav.unsqueeze(0)
        squeeze = True
    else:
        x = wav
        squeeze = False
    n = int(x.shape[-1])
    X = torch.fft.rfft(x.float(), dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / float(sr)).to(device=x.device, dtype=x.dtype)
    mask = (freqs <= float(cutoff_hz)).to(X.dtype)
    y = torch.fft.irfft(X * mask, n=n, dim=-1)
    return y.squeeze(0) if squeeze else y


def make_random_upsampled_view(
    audio: Sequence[torch.Tensor],
    sample_rates: Sequence[int],
    min_factor: float = 1.05,
    max_factor: float = 1.50,
    round_to: int = 1000,
    max_hz: int = 0,
) -> tuple[list[torch.Tensor], list[int]]:
    out_audio: list[torch.Tensor] = []
    out_sr: list[int] = []
    for x, sr in zip(audio, sample_rates):
        factor = random.uniform(float(min_factor), float(max_factor))
        target = max(int(sr), int(round(float(sr) * factor)))
        if round_to > 0 and target > 4 * round_to:
            target = max(1, int(round(target / round_to) * round_to))
        if max_hz > 0:
            target = min(target, int(max_hz))
        target = max(target, int(sr))
        out_audio.append(fft_resample_1d(x.float(), int(sr), target))
        out_sr.append(target)
    return out_audio, out_sr


# -----------------------------------------------------------------------------
# Sampling-frequency-independent wavelet/scattering frontend.
# -----------------------------------------------------------------------------


class SFIWaveletScatteringFrontend(nn.Module):
    """Continuous-parameter log-frequency scattering-style frontend.

    The filters are specified in physical units: center frequency in Hz, temporal
    analysis windows in milliseconds, and modulation rates in Hz. For each input
    sample rate, the same physical filters are evaluated on that recording's STFT
    grid. Frequencies above the safe Nyquist limit are omitted by a mask and are
    never used by the encoder.

    Compared with a fixed-sample STFT or fixed-size spectrogram image, this avoids
    making a 1024-sample kernel mean different physical durations for different
    species and sensors.

    Output:
      coeffs: [B, F, C, T]
      valid_mask: [B, F] where True means the physical frequency row is observable
      freqs_hz: [F]
    """

    def __init__(
        self,
        freq_min_hz: float = 5.0,
        freq_max_hz: float = 120_000.0,
        freq_bins: int = 96,
        time_bins: int = 96,
        q_factor: float = 12.0,
        window_ms: Sequence[float] = (16.0, 64.0, 256.0),
        modulation_rates_hz: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
        nyquist_rolloff: float = 0.92,
        normalize: bool = True,
        min_window_samples: int = 16,
        max_window_samples: int = 8192,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.freq_min_hz = float(freq_min_hz)
        self.freq_max_hz = float(freq_max_hz)
        self.freq_bins = int(freq_bins)
        self.time_bins = int(time_bins)
        self.q_factor = float(q_factor)
        self.window_ms = tuple(float(x) for x in window_ms if float(x) > 0)
        if not self.window_ms:
            raise ValueError("At least one analysis window in ms is required")
        self.modulation_rates_hz = tuple(float(x) for x in modulation_rates_hz if float(x) > 0)
        self.nyquist_rolloff = float(nyquist_rolloff)
        self.normalize = bool(normalize)
        self.min_window_samples = int(min_window_samples)
        self.max_window_samples = int(max_window_samples)
        self.eps = float(eps)
        freqs = torch.logspace(
            math.log10(max(self.freq_min_hz, 1e-3)),
            math.log10(max(self.freq_max_hz, self.freq_min_hz + 1e-3)),
            steps=self.freq_bins,
        )
        self.register_buffer("target_freqs_hz", freqs, persistent=False)

    @property
    def out_channels(self) -> int:
        return len(self.window_ms) * (1 + len(self.modulation_rates_hz))

    def _stft_energy_for_window(self, wav: torch.Tensor, sr: int, window_ms: float) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(wav.numel())
        win = int(round(float(sr) * float(window_ms) / 1000.0))
        win = max(self.min_window_samples, min(self.max_window_samples, win))
        win = min(win, n) if n > 0 else win
        win = max(4, win)
        hop = max(1, int(round(win * 0.25)))
        n_fft = max(next_power_of_two(win), win)
        if n < win:
            wav = F.pad(wav, (0, win - n))
        window = torch.hann_window(win, device=wav.device, dtype=wav.dtype)
        spec = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=True,
            return_complex=True,
        ).abs().pow(2.0)
        freqs = torch.linspace(0.0, float(sr) * 0.5, spec.shape[0], device=wav.device, dtype=wav.dtype)
        return spec, freqs

    def _project_to_physical_log_filters(self, spec: torch.Tensor, src_freqs: torch.Tensor, target_freqs: torch.Tensor) -> torch.Tensor:
        # Log-Gaussian constant-Q projection. This is a wavelet-like filterbank in
        # frequency; q_factor controls relative bandwidth.
        safe_src = src_freqs.clamp_min(self.freq_min_hz * 0.25)
        log_ratio = torch.log(safe_src[None, :] / target_freqs[:, None].clamp_min(self.eps))
        sigma_log = max(1.0 / self.q_factor, 1e-3)
        H = torch.exp(-0.5 * (log_ratio / sigma_log).pow(2))
        H = H * (src_freqs[None, :] > 0.0).to(spec.dtype)
        H = H / H.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.matmul(H, spec)

    def _modulation_channels(self, env: torch.Tensor, duration: float) -> list[torch.Tensor]:
        # env: [F_valid, T]
        channels: list[torch.Tensor] = [env]
        if not self.modulation_rates_hz:
            return channels
        T = int(env.shape[-1])
        centered = env - env.mean(dim=-1, keepdim=True)
        E = torch.fft.rfft(centered, dim=-1)
        rate_axis = torch.fft.rfftfreq(T, d=duration / float(T)).to(device=env.device, dtype=env.dtype)
        safe_rate = rate_axis.clamp_min(1.0 / max(duration, self.eps))
        for alpha in self.modulation_rates_hz:
            log_mod = torch.log(safe_rate / float(alpha))
            Hm = torch.exp(-0.5 * (log_mod / 0.50).pow(2))
            Hm = Hm * (rate_axis > 0.0).to(env.dtype)
            mod = torch.fft.irfft(E * Hm[None, :].to(E.dtype), n=T, dim=-1).abs()
            channels.append(torch.log1p(mod))
        return channels

    def _one(self, wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        else:
            wav = wav.flatten()
        wav = wav.float()
        n = int(wav.numel())
        if n < 8:
            wav = F.pad(wav, (0, 8 - n))
            n = int(wav.numel())
        duration = max(float(n) / float(sr), self.eps)
        target_freqs = self.target_freqs_hz.to(device=wav.device, dtype=wav.dtype)
        valid = target_freqs <= (float(sr) * 0.5 * self.nyquist_rolloff)
        if not bool(valid.any()):
            valid = valid.clone()
            valid[0] = True
        valid_freqs = target_freqs[valid]

        all_channels: list[torch.Tensor] = []
        for w_ms in self.window_ms:
            spec, src_freqs = self._stft_energy_for_window(wav, int(sr), float(w_ms))
            band_energy = self._project_to_physical_log_filters(spec, src_freqs, valid_freqs)
            env = torch.sqrt(band_energy.clamp_min(self.eps))
            env = torch.log1p(env)
            env = F.interpolate(env.unsqueeze(0).unsqueeze(0), size=(env.shape[0], self.time_bins), mode="bilinear", align_corners=False)[0, 0]
            all_channels.extend(self._modulation_channels(env, duration))

        coeff_valid = torch.stack(all_channels, dim=1)  # [F_valid, C, T]
        if self.normalize:
            mu = coeff_valid.mean()
            std = coeff_valid.std(unbiased=False).clamp_min(1e-5)
            coeff_valid = (coeff_valid - mu) / std
        coeff = wav.new_zeros((self.freq_bins, self.out_channels, self.time_bins))
        coeff[valid] = coeff_valid.to(coeff.dtype)
        return coeff, valid

    def forward(self, audio: Any, sample_rate: Any) -> dict[str, torch.Tensor]:
        audio_list = _as_audio_list(audio)
        sr_list = _as_sr_list(sample_rate, len(audio_list))
        coeffs: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for wav, sr in zip(audio_list, sr_list):
            c, m = self._one(wav, int(sr))
            coeffs.append(c)
            masks.append(m.to(device=c.device))
        return {
            "coeffs": torch.stack(coeffs, dim=0),
            "valid_mask": torch.stack(masks, dim=0),
            "freqs_hz": self.target_freqs_hz.to(device=coeffs[0].device, dtype=coeffs[0].dtype),
        }


class MaskedFrequencyEncoder(nn.Module):
    """Encode per-frequency temporal coefficients, then aggregate valid rows."""

    def __init__(
        self,
        in_channels: int,
        token_dim: int = 96,
        embedding_dim: int = 192,
        transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.time_net = nn.Sequential(
            nn.Conv1d(in_channels, token_dim, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(group_count(token_dim), token_dim),
            nn.GELU(),
            nn.Conv1d(token_dim, token_dim, kernel_size=5, padding=2, groups=group_count(token_dim), bias=False),
            nn.GroupNorm(group_count(token_dim), token_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.stat_proj = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.freq_pos = nn.Sequential(
            nn.Linear(5, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.out = nn.Sequential(
            nn.Linear(token_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(x.dtype).unsqueeze(-1)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        very_neg = torch.finfo(x.dtype).min / 4.0
        y = x.masked_fill(~mask.unsqueeze(-1), very_neg)
        return y.amax(dim=1)

    def forward(self, coeffs: torch.Tensor, valid_mask: torch.Tensor, freqs_hz: torch.Tensor) -> torch.Tensor:
        # coeffs: [B, F, C, T]
        B, Freq, C, T = coeffs.shape
        x = coeffs.reshape(B * Freq, C, T)
        y = self.time_net(x)
        mean = y.mean(dim=-1)
        mx = y.amax(dim=-1)
        std = y.std(dim=-1, unbiased=False)
        tokens = self.stat_proj(torch.cat([mean, mx, std], dim=-1)).view(B, Freq, self.token_dim)

        lf = torch.log(freqs_hz.float().clamp_min(1.0))
        lf = (lf - lf.mean()) / lf.std(unbiased=False).clamp_min(1e-6)
        pos_in = torch.stack([
            lf,
            torch.sin(math.pi * lf),
            torch.cos(math.pi * lf),
            torch.sin(2.0 * math.pi * lf),
            torch.cos(2.0 * math.pi * lf),
        ], dim=-1).to(device=coeffs.device, dtype=coeffs.dtype)
        tokens = tokens + self.freq_pos(pos_in).unsqueeze(0).to(tokens.dtype)
        tokens = tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        h = self.transformer(tokens, src_key_padding_mask=~valid_mask)
        h = h.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        pooled = torch.cat([self._masked_mean(h, valid_mask), self._masked_max(h, valid_mask)], dim=-1)
        return self.out(pooled)


class TwoLayerHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(32, min(512, in_dim * 2)))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SFIScatNet(nn.Module):
    """Native-sample-rate scattering-style classifier."""

    def __init__(
        self,
        num_classes: int,
        freq_min_hz: float = 5.0,
        freq_max_hz: float = 120_000.0,
        freq_bins: int = 96,
        time_bins: int = 96,
        q_factor: float = 12.0,
        modulation_rates_hz: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
        window_ms: Sequence[float] = (16.0, 64.0, 256.0),
        token_dim: int = 96,
        embedding_dim: int = 192,
        transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.frontend = SFIWaveletScatteringFrontend(
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            freq_bins=freq_bins,
            time_bins=time_bins,
            q_factor=q_factor,
            modulation_rates_hz=modulation_rates_hz,
            window_ms=window_ms,
        )
        self.encoder = MaskedFrequencyEncoder(
            in_channels=self.frontend.out_channels,
            token_dim=token_dim,
            embedding_dim=embedding_dim,
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.classifier = TwoLayerHead(embedding_dim, int(num_classes), hidden_dim=head_hidden_dim, dropout=dropout)

    def forward(self, audio: Any, sample_rate: Any, return_hidden: bool = False) -> dict[str, Any]:
        front = self.frontend(audio, sample_rate)
        z = self.encoder(front["coeffs"], front["valid_mask"], front["freqs_hz"])
        logits = self.classifier(z)
        out: dict[str, Any] = {"logits": logits, "embedding": z}
        if return_hidden:
            out["hidden"] = front
        return out


# -----------------------------------------------------------------------------
# Baseline frontends and classifiers.
# -----------------------------------------------------------------------------


def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def make_mel_filterbank(n_fft: int, sample_rate: int, n_mels: int, f_min: float, f_max: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    f_max = min(float(f_max), float(sample_rate) * 0.5)
    fft_freqs = torch.linspace(0.0, float(sample_rate) * 0.5, n_fft // 2 + 1, device=device, dtype=dtype)
    mel_min = hz_to_mel(torch.tensor(float(f_min), device=device, dtype=dtype))
    mel_max = hz_to_mel(torch.tensor(float(f_max), device=device, dtype=dtype))
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2, device=device, dtype=dtype)
    hz_pts = mel_to_hz(mel_pts)
    fb = torch.zeros(n_mels, n_fft // 2 + 1, device=device, dtype=dtype)
    for m in range(n_mels):
        left, center, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (fft_freqs - left) / (center - left).clamp_min(1e-6)
        down = (right - fft_freqs) / (right - center).clamp_min(1e-6)
        fb[m] = torch.clamp(torch.minimum(up, down), min=0.0)
    fb = fb / fb.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return fb


class FixedMelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 47_600,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 200,
        image_time_bins: int = 96,
        f_min: float = 5.0,
        f_max: float = 24_000.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.image_time_bins = int(image_time_bins)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.normalize = bool(normalize)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def _batch_resample_pad(self, audio: Any, sample_rate: Any, device: torch.device) -> torch.Tensor:
        audio_list = _as_audio_list(audio)
        sr_list = _as_sr_list(sample_rate, len(audio_list))
        out: list[torch.Tensor] = []
        max_len = 0
        for wav, sr in zip(audio_list, sr_list):
            w = fft_resample_1d(wav.float().to(device), int(sr), self.sample_rate)
            if w.dim() == 2:
                w = w.mean(dim=0)
            out.append(w)
            max_len = max(max_len, int(w.shape[-1]))
        padded = []
        for w in out:
            if w.shape[-1] < max_len:
                w = F.pad(w, (0, max_len - w.shape[-1]))
            padded.append(w)
        return torch.stack(padded, dim=0)

    def forward(self, audio: Any, sample_rate: Any) -> torch.Tensor:
        device = self.window.device
        wav = self._batch_resample_pad(audio, sample_rate, device=device)
        if wav.shape[-1] < self.n_fft:
            wav = F.pad(wav, (0, self.n_fft - wav.shape[-1]))
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=wav.device, dtype=wav.dtype),
            center=True,
            return_complex=True,
        ).abs().pow(2.0)
        fb = make_mel_filterbank(
            self.n_fft,
            self.sample_rate,
            self.n_mels,
            self.f_min,
            self.f_max,
            wav.device,
            wav.dtype,
        )
        mel = torch.einsum("mf,bft->bmt", fb, spec)
        mel = torch.log1p(mel)
        mel = F.interpolate(mel.unsqueeze(1), size=(self.n_mels, self.image_time_bins), mode="bilinear", align_corners=False)
        if self.normalize:
            mean = mel.mean(dim=(-2, -1), keepdim=True)
            std = mel.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-5)
            mel = (mel - mean) / std
        return mel


class ResBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int] = (1, 1), dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != (1, 1):
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.gelu(self.net(x) + self.skip(x)))


class MelCNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mel_frontend: FixedMelFrontend,
        base_channels: int = 16,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.frontend = mel_frontend
        c = int(base_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            ResBlock2d(c, c, dropout=dropout),
            ResBlock2d(c, c * 2, stride=(2, 2), dropout=dropout),
            ResBlock2d(c * 2, c * 4, stride=(2, 2), dropout=dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(c * 4 * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = TwoLayerHead(embedding_dim, int(num_classes), hidden_dim=head_hidden_dim, dropout=dropout)

    def forward(self, audio: Any, sample_rate: Any, return_hidden: bool = False) -> dict[str, Any]:
        x = self.frontend(audio, sample_rate)
        h = self.encoder(x)
        pooled = torch.cat([h.mean(dim=(-2, -1)), h.amax(dim=(-2, -1))], dim=-1)
        z = self.proj(pooled)
        out: dict[str, Any] = {"logits": self.classifier(z), "embedding": z}
        if return_hidden:
            out["hidden"] = {"mel": x, "encoder": h}
        return out


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1, activation: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True) if activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden, kernel_size=1))
        layers.append(ConvBNAct(hidden, hidden, kernel_size=kernel_size, stride=stride, groups=hidden))
        layers.append(SqueezeExcite(hidden))
        layers.append(ConvBNAct(hidden, out_ch, kernel_size=1, activation=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + y if self.use_residual else y


class MelEfficientNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mel_frontend: FixedMelFrontend,
        width: int = 16,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.frontend = mel_frontend
        w = int(width)
        self.stem = ConvBNAct(1, w, kernel_size=3, stride=2)
        configs = [(1, w, 1, 1, 3), (4, w * 2, 2, 2, 3), (4, w * 3, 2, 2, 5), (4, w * 4, 2, 2, 3), (4, w * 5, 1, 1, 5)]
        blocks: list[nn.Module] = []
        in_ch = w
        for expand, out_ch, repeats, stride, kernel in configs:
            for r in range(repeats):
                blocks.append(MBConv(in_ch, out_ch, expand_ratio=expand, kernel_size=kernel, stride=stride if r == 0 else 1))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.proj = ConvBNAct(in_ch, embedding_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = TwoLayerHead(embedding_dim, int(num_classes), hidden_dim=head_hidden_dim, dropout=dropout)

    def forward(self, audio: Any, sample_rate: Any, return_hidden: bool = False) -> dict[str, Any]:
        x = self.frontend(audio, sample_rate)
        h = self.stem(x)
        h = self.blocks(h)
        h = self.proj(h)
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        z = self.drop(pooled)
        out: dict[str, Any] = {"logits": self.classifier(z), "embedding": z}
        if return_hidden:
            out["hidden"] = {"mel": x, "encoder": h}
        return out


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_freq: int = 8, modes_time: int = 8) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_freq = int(modes_freq)
        self.modes_time = int(modes_time)
        scale = 1.0 / math.sqrt(max(1, in_channels * out_channels))
        shape = (in_channels, out_channels, self.modes_freq, self.modes_time)
        self.wr = nn.Parameter(scale * torch.randn(*shape))
        self.wi = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Freq, Time = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = x_ft.new_zeros(B, self.out_channels, Freq, Time // 2 + 1)
        mf = min(self.modes_freq, Freq)
        mt = min(self.modes_time, Time // 2 + 1)
        w = torch.complex(self.wr[:, :, :mf, :mt], self.wi[:, :, :mf, :mt])
        out_ft[:, :, :mf, :mt] = torch.einsum("bcft,coft->boft", x_ft[:, :, :mf, :mt], w)
        return torch.fft.irfft2(out_ft, s=(Freq, Time), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(self, channels: int, modes_freq: int = 8, modes_time: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_freq=modes_freq, modes_time=modes_time)
        self.local = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.local(x)
        y = self.drop(F.gelu(self.norm(y)))
        return x + y


class MelCNNFNOClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mel_frontend: FixedMelFrontend,
        base_channels: int = 16,
        embedding_dim: int = 128,
        fno_depth: int = 2,
        modes_freq: int = 8,
        modes_time: int = 8,
        dropout: float = 0.1,
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.frontend = mel_frontend
        c = int(base_channels)
        self.lift = nn.Sequential(nn.Conv2d(1, c, 3, padding=1, bias=False), nn.GroupNorm(group_count(c), c), nn.GELU())
        self.blocks = nn.ModuleList([FNOBlock2d(c, modes_freq=modes_freq, modes_time=modes_time, dropout=dropout) for _ in range(int(fno_depth))])
        self.conv = nn.Sequential(
            ResBlock2d(c, c * 2, stride=(2, 2), dropout=dropout),
            ResBlock2d(c * 2, c * 4, stride=(2, 2), dropout=dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(c * 4 * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = TwoLayerHead(embedding_dim, int(num_classes), hidden_dim=head_hidden_dim, dropout=dropout)

    def forward(self, audio: Any, sample_rate: Any, return_hidden: bool = False) -> dict[str, Any]:
        x = self.frontend(audio, sample_rate)
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        h = self.conv(h)
        pooled = torch.cat([h.mean(dim=(-2, -1)), h.amax(dim=(-2, -1))], dim=-1)
        z = self.proj(pooled)
        out: dict[str, Any] = {"logits": self.classifier(z), "embedding": z}
        if return_hidden:
            out["hidden"] = {"mel": x, "encoder": h}
        return out


class SFIFNOClassifier(nn.Module):
    """Native-sample-rate SFI front-end with an FNO-style backend.

    This model tests whether the useful part is the physical-unit front-end rather
    than the small Transformer backend used in SFIScatNet. It receives the same
    SFI coefficients [frequency, feature-channel, time], masks frequency rows that
    are above the valid Nyquist range, and then applies Fourier Neural Operator
    blocks over the physical frequency-time plane.
    """

    def __init__(
        self,
        num_classes: int,
        freq_min_hz: float = 5.0,
        freq_max_hz: float = 120_000.0,
        freq_bins: int = 96,
        time_bins: int = 96,
        q_factor: float = 12.0,
        modulation_rates_hz: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
        window_ms: Sequence[float] = (16.0, 64.0, 256.0),
        base_channels: int = 16,
        embedding_dim: int = 192,
        fno_depth: int = 2,
        modes_freq: int = 8,
        modes_time: int = 8,
        dropout: float = 0.1,
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.frontend = SFIWaveletScatteringFrontend(
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            freq_bins=freq_bins,
            time_bins=time_bins,
            q_factor=q_factor,
            modulation_rates_hz=modulation_rates_hz,
            window_ms=window_ms,
        )
        c = int(base_channels)
        self.lift = nn.Sequential(
            nn.Conv2d(self.frontend.out_channels, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(c), c),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            FNOBlock2d(c, modes_freq=modes_freq, modes_time=modes_time, dropout=dropout)
            for _ in range(int(fno_depth))
        ])
        self.conv = nn.Sequential(
            ResBlock2d(c, c * 2, stride=(2, 2), dropout=dropout),
            ResBlock2d(c * 2, c * 4, stride=(2, 2), dropout=dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(c * 4 * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = TwoLayerHead(embedding_dim, int(num_classes), hidden_dim=head_hidden_dim, dropout=dropout)

    def forward(self, audio: Any, sample_rate: Any, return_hidden: bool = False) -> dict[str, Any]:
        front = self.frontend(audio, sample_rate)
        x = front["coeffs"].permute(0, 2, 1, 3).contiguous()  # [B, C, F, T]
        mask = front["valid_mask"].to(dtype=x.dtype, device=x.device).unsqueeze(1).unsqueeze(-1)
        x = x * mask
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
            h = h * mask
        h = self.conv(h)
        pooled = torch.cat([h.mean(dim=(-2, -1)), h.amax(dim=(-2, -1))], dim=-1)
        z = self.proj(pooled)
        out: dict[str, Any] = {"logits": self.classifier(z), "embedding": z}
        if return_hidden:
            out["hidden"] = {"sfi": front, "encoder": h}
        return out




# -----------------------------------------------------------------------------
# WhaleNet-style fixed-rate WST/Mel baselines.
# -----------------------------------------------------------------------------


def _center_crop_or_pad_samples(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """Center crop/pad a 1-D or [1,N] waveform to a fixed number of samples."""
    squeeze = False
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeeze = True
    n = int(wav.shape[-1])
    target_len = int(target_len)
    if n == target_len:
        out = wav
    elif n < target_len:
        left = (target_len - n) // 2
        right = target_len - n - left
        out = F.pad(wav, (left, right))
    else:
        start = (n - target_len) // 2
        out = wav[..., start:start + target_len]
    return out.squeeze(0) if squeeze else out


def _standardize_per_item(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dims = tuple(range(1, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


class WhaleNetSignalPreprocessor(nn.Module):
    """Paper-style fixed preprocessing: resample, center crop/pad, standardize.

    The uploaded baseline code first resamples every signal to a fixed rate, then
    center-crops or zero-pads to 8000 samples and standardizes the waveform before
    computing Mel/WST features. This module reproduces that behavior without using
    any external audio backend inside the model.
    """

    def __init__(self, sample_rate: int = 47600, signal_len: int = 8000, standardize: bool = True) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.signal_len = int(signal_len)
        self.standardize = bool(standardize)

    def forward(self, audio: Any, sample_rate: Any) -> torch.Tensor:
        audio_list = _as_audio_list(audio)
        sr_list = _as_sr_list(sample_rate, len(audio_list))
        device = audio_list[0].device if audio_list else torch.device('cpu')
        out: list[torch.Tensor] = []
        for wav, sr in zip(audio_list, sr_list):
            w = wav.float().to(device)
            if w.dim() == 2:
                w = w.mean(dim=0, keepdim=True)
            w = fft_resample_1d(w, int(sr), self.sample_rate)
            w = _center_crop_or_pad_samples(w, self.signal_len)
            if w.dim() == 2:
                w = w.squeeze(0)
            out.append(w)
        batch = torch.stack(out, dim=0)
        if self.standardize:
            batch = _standardize_per_item(batch)
        return batch


class WhaleNetMelFrontend(nn.Module):
    """Mel branch used by the WhaleNet-style baseline."""

    def __init__(
        self,
        sample_rate: int = 47600,
        signal_len: int = 8000,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 200,
        f_min: float = 0.0,
        f_max: float = 0.0,
        standardize_signal: bool = True,
        normalize_image: bool = True,
    ) -> None:
        super().__init__()
        self.pre = WhaleNetSignalPreprocessor(sample_rate=sample_rate, signal_len=signal_len, standardize=standardize_signal)
        self.sample_rate = int(sample_rate)
        self.signal_len = int(signal_len)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.f_min = float(f_min)
        self.f_max = float(f_max) if float(f_max) > 0 else float(sample_rate) * 0.5
        self.normalize_image = bool(normalize_image)
        self.register_buffer('window', torch.hann_window(self.n_fft), persistent=False)

    def forward(self, audio: Any, sample_rate: Any) -> torch.Tensor:
        wav = self.pre(audio, sample_rate).to(self.window.device)
        if wav.shape[-1] < self.n_fft:
            wav = F.pad(wav, (0, self.n_fft - wav.shape[-1]))
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=wav.device, dtype=wav.dtype),
            center=True,
            return_complex=True,
        ).abs().pow(2.0)
        fb = make_mel_filterbank(
            self.n_fft,
            self.sample_rate,
            self.n_mels,
            self.f_min,
            self.f_max,
            wav.device,
            wav.dtype,
        )
        mel = torch.einsum('mf,bft->bmt', fb, spec)
        mel = torch.log1p(mel).unsqueeze(1)
        if self.normalize_image:
            mel = _standardize_per_item(mel)
        return mel


class WhaleNetWSTFrontend(nn.Module):
    """Kymatio WST branch used by the WhaleNet-style baseline.

    Output is a 2-D image [B,1,C,T] for either order-1 or order-2 scattering
    coefficients. Each order is normalized per sample with median/std, matching
    the uploaded baseline code at the level of preprocessing logic.
    """

    def __init__(
        self,
        sample_rate: int = 47600,
        signal_len: int = 8000,
        J: int = 6,
        Q: int = 16,
        order: int = 1,
        standardize_signal: bool = True,
        median_norm: bool = True,
    ) -> None:
        super().__init__()
        self.pre = WhaleNetSignalPreprocessor(sample_rate=sample_rate, signal_len=signal_len, standardize=standardize_signal)
        self.sample_rate = int(sample_rate)
        self.signal_len = int(signal_len)
        self.J = int(J)
        self.Q = int(Q)
        self.order = int(order)
        self.median_norm = bool(median_norm)
        try:
            from kymatio.torch import Scattering1D  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError('The WhaleNet-style WST baseline requires kymatio. Install it with: pip install kymatio') from exc
        self.scattering = Scattering1D(J=self.J, shape=self.signal_len, Q=self.Q)
        meta = self.scattering.meta()
        idx = torch.as_tensor((meta['order'] == self.order).nonzero()[0], dtype=torch.long)
        if idx.numel() == 0:
            raise RuntimeError(f'Kymatio returned zero coefficients for WST order {self.order}')
        self.register_buffer('order_idx', idx, persistent=False)

    def forward(self, audio: Any, sample_rate: Any) -> torch.Tensor:
        wav = self.pre(audio, sample_rate).to(self.order_idx.device)
        sx = self.scattering(wav)
        sx = sx.index_select(1, self.order_idx.to(sx.device))
        if self.median_norm:
            med = sx.median(dim=2, keepdim=True).values.median(dim=1, keepdim=True).values
            std = sx.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-5)
            sx = (sx - med) / std
        return sx.unsqueeze(1)


class WhaleNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class WhaleNetResNetBranch(nn.Module):
    """Small ResNet branch from the uploaded WhaleNet-style code."""

    def __init__(self, num_classes: int, in_channels: int = 1, layers: Sequence[int] = (2, 2, 2)) -> None:
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, int(layers[0]), stride=1)
        self.layer2 = self._make_layer(32, int(layers[1]), stride=2)
        self.layer3 = self._make_layer(64, int(layers[2]), stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, int(num_classes))

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers: list[nn.Module] = [WhaleNetBasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(WhaleNetBasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.avg_pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))


class WhaleNetFusionMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden1: int = 256, hidden2: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(in_dim), int(hidden1))
        self.fc2 = nn.Linear(int(hidden1), int(hidden2))
        self.fc3 = nn.Linear(int(hidden2), int(num_classes))

    def forward_with_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h), h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_embedding(x)[0]


class WhaleNetBranchClassifier(nn.Module):
    def __init__(self, num_classes: int, branch: str, args: Any) -> None:
        super().__init__()
        branch = str(branch).lower()
        sr = int(getattr(args, 'whalenet_sr', 0) or getattr(args, 'baseline_sr', 47600))
        sig_len = int(getattr(args, 'whalenet_signal_len', 8000))
        if branch == 'mel':
            self.frontend = WhaleNetMelFrontend(
                sample_rate=sr,
                signal_len=sig_len,
                n_mels=int(getattr(args, 'whalenet_mel_bins', 64)),
                n_fft=int(getattr(args, 'whalenet_mel_n_fft', 400)),
                hop_length=int(getattr(args, 'whalenet_mel_hop_length', 200)),
            )
        elif branch in {'wst1', 'wst2'}:
            self.frontend = WhaleNetWSTFrontend(
                sample_rate=sr,
                signal_len=sig_len,
                J=int(getattr(args, 'whalenet_j', 6)),
                Q=int(getattr(args, 'whalenet_q', 16)),
                order=1 if branch == 'wst1' else 2,
            )
        else:
            raise ValueError(f'Unknown WhaleNet branch {branch}')
        self.branch = branch
        self.encoder = WhaleNetResNetBranch(num_classes=num_classes, in_channels=1)

    def forward(self, audio: Any, sample_rate: Any, return_hidden: bool = False) -> dict[str, Any]:
        x = self.frontend(audio, sample_rate)
        z = self.encoder.forward_features(x)
        logits = self.encoder.fc(z)
        out: dict[str, Any] = {'logits': logits, 'embedding': z}
        if return_hidden:
            out['hidden'] = {self.branch: x}
        return out


class WhaleNetWST12Classifier(nn.Module):
    """WhaleNet-style S1/S2 WST fusion baseline.

    This approximates the uploaded code path where separate ResNets are trained on
    order-1 and order-2 WST coefficients and their logits are merged by an MLP.
    Here the full module is trained in one run so it can share the same training
    harness and split as all other models.
    """

    def __init__(self, num_classes: int, args: Any) -> None:
        super().__init__()
        self.wst1 = WhaleNetBranchClassifier(num_classes, 'wst1', args)
        self.wst2 = WhaleNetBranchClassifier(num_classes, 'wst2', args)
        self.fuse = WhaleNetFusionMLP(2 * int(num_classes), int(num_classes))

    def forward(self, audio: Any, sample_rate: Any, return_hidden: bool = False) -> dict[str, Any]:
        o1 = self.wst1(audio, sample_rate, return_hidden=return_hidden)
        o2 = self.wst2(audio, sample_rate, return_hidden=return_hidden)
        features = torch.cat([o1['logits'], o2['logits']], dim=-1)
        logits, emb = self.fuse.forward_with_embedding(features)
        out: dict[str, Any] = {'logits': logits, 'embedding': emb}
        if return_hidden:
            out['hidden'] = {'wst1': o1.get('hidden'), 'wst2': o2.get('hidden'), 'branch_logits': features}
        return out


class WhaleNetFullClassifier(nn.Module):
    """WhaleNet-style Mel + WST ensemble baseline.

    The structure follows the uploaded implementation: Mel, first-order WST, and
    second-order WST are processed by separate small ResNets; WST logits are first
    merged, then the WST and Mel logits are fused by another MLP.
    """

    def __init__(self, num_classes: int, args: Any) -> None:
        super().__init__()
        self.mel = WhaleNetBranchClassifier(num_classes, 'mel', args)
        self.wst12 = WhaleNetWST12Classifier(num_classes, args)
        self.final_fuse = WhaleNetFusionMLP(2 * int(num_classes), int(num_classes))

    def forward(self, audio: Any, sample_rate: Any, return_hidden: bool = False) -> dict[str, Any]:
        om = self.mel(audio, sample_rate, return_hidden=return_hidden)
        ow = self.wst12(audio, sample_rate, return_hidden=return_hidden)
        features = torch.cat([ow['logits'], om['logits']], dim=-1)
        logits, emb = self.final_fuse.forward_with_embedding(features)
        out: dict[str, Any] = {'logits': logits, 'embedding': emb}
        if return_hidden:
            out['hidden'] = {'mel': om.get('hidden'), 'wst12': ow.get('hidden'), 'branch_logits': features}
        return out


@dataclass(frozen=True)
class ModelConfig:
    name: str
    kind: str


def build_model(name: str, num_classes: int, args: Any, device: torch.device) -> nn.Module:
    name = str(name).strip().lower()
    head_hidden = int(getattr(args, "head_hidden_dim", 0)) or None
    if name == "sfi_scatnet":
        model = SFIScatNet(
            num_classes=num_classes,
            freq_min_hz=args.sfi_freq_min_hz,
            freq_max_hz=args.sfi_freq_max_hz,
            freq_bins=args.sfi_freq_bins,
            time_bins=args.sfi_time_bins,
            q_factor=args.sfi_q_factor,
            modulation_rates_hz=parse_float_tuple(args.sfi_modulation_rates_hz),
            window_ms=parse_float_tuple(args.sfi_window_ms),
            token_dim=args.sfi_token_dim,
            embedding_dim=args.embedding_dim,
            transformer_layers=args.sfi_transformer_layers,
            num_heads=args.sfi_num_heads,
            dropout=args.dropout,
            head_hidden_dim=head_hidden,
        )
        return model.to(device)
    if name == "sfi_fno":
        model = SFIFNOClassifier(
            num_classes=num_classes,
            freq_min_hz=args.sfi_freq_min_hz,
            freq_max_hz=args.sfi_freq_max_hz,
            freq_bins=args.sfi_freq_bins,
            time_bins=args.sfi_time_bins,
            q_factor=args.sfi_q_factor,
            modulation_rates_hz=parse_float_tuple(args.sfi_modulation_rates_hz),
            window_ms=parse_float_tuple(args.sfi_window_ms),
            base_channels=args.baseline_channels,
            embedding_dim=args.embedding_dim,
            fno_depth=args.fno_depth,
            modes_freq=args.fno_modes_freq,
            modes_time=args.fno_modes_time,
            dropout=args.dropout,
            head_hidden_dim=head_hidden,
        )
        return model.to(device)

    mel = FixedMelFrontend(
        sample_rate=args.baseline_sr,
        n_mels=args.mel_bins,
        n_fft=args.mel_n_fft,
        hop_length=args.mel_hop_length,
        image_time_bins=args.mel_time_bins,
        f_min=args.mel_f_min_hz,
        f_max=args.mel_f_max_hz,
    )
    if name == "mel_cnn":
        return MelCNNClassifier(num_classes, mel, base_channels=args.baseline_channels, embedding_dim=args.embedding_dim, dropout=args.dropout, head_hidden_dim=head_hidden).to(device)
    if name == "mel_efficientnet":
        return MelEfficientNetClassifier(num_classes, mel, width=args.efficientnet_width, embedding_dim=args.embedding_dim, dropout=args.dropout, head_hidden_dim=head_hidden).to(device)
    if name == "mel_cnn_fno":
        return MelCNNFNOClassifier(
            num_classes,
            mel,
            base_channels=args.baseline_channels,
            embedding_dim=args.embedding_dim,
            fno_depth=args.fno_depth,
            modes_freq=args.fno_modes_freq,
            modes_time=args.fno_modes_time,
            dropout=args.dropout,
            head_hidden_dim=head_hidden,
        ).to(device)
    if name == "whalenet_mel":
        return WhaleNetBranchClassifier(num_classes, "mel", args).to(device)
    if name == "whalenet_wst1":
        return WhaleNetBranchClassifier(num_classes, "wst1", args).to(device)
    if name == "whalenet_wst2":
        return WhaleNetBranchClassifier(num_classes, "wst2", args).to(device)
    if name == "whalenet_wst12":
        return WhaleNetWST12Classifier(num_classes, args).to(device)
    if name == "whalenet_full":
        return WhaleNetFullClassifier(num_classes, args).to(device)
    raise ValueError(
        f"Unknown model {name!r}. Use sfi_fno, sfi_scatnet, mel_cnn, mel_efficientnet, mel_cnn_fno, "
        "whalenet_mel, whalenet_wst1, whalenet_wst2, whalenet_wst12, or whalenet_full."
    )


def parse_float_tuple(text: Any) -> tuple[float, ...]:
    if isinstance(text, (list, tuple)):
        return tuple(float(x) for x in text)
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return tuple()
    return tuple(float(p) for p in parts)
