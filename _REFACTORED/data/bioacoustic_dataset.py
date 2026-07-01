#!/usr/bin/env python3
"""Generic PyTorch dataset for the canonical bioacoustic manifest layout.

This file intentionally knows nothing about WhaleSounds, AnuraSet, BirdSet, or
Mendeley. It reads rows from a manifest produced by the homogeneous downloader.

Supported sample storage:
  audio_path  -> decoded with soundfile
  array_path + array_index -> read from a numpy array/memmap, used for WhaleSounds

No torchaudio. No dataset-specific path guessing.

Sample-rate augmentation:
  Bioacoustic_Dataset(..., aug=True) randomly resamples each item near its native
  sample rate. For example, a 32 kHz sample uses the default range 0.75x..1.125x,
  i.e. roughly 24 kHz..36 kHz. The returned sample_rate is the augmented rate;
  native_sample_rate remains the original rate.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


MANIFEST_FIELDS = [
    "global_id",
    "dataset",
    "split",
    "sample_id",
    "audio_path",
    "array_path",
    "array_index",
    "native_sample_rate",
    "duration_seconds",
    "label_indices",
    "label_names",
    "source",
    "source_id",
]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def parse_int_list(text: Any) -> list[int]:
    if text is None:
        return []
    out = []
    for part in str(text).replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def as_mono_float_tensor(audio: Any) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 2:
        if x.shape[0] <= 8:
            pass
        elif x.shape[1] <= 8:
            x = x.transpose(0, 1)
        else:
            x = x.mean(dim=0, keepdim=True)
    else:
        x = x.reshape(-1, x.shape[-1])
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    return x.contiguous()


def peak_normalize(audio: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = audio.abs().max()
    if torch.isfinite(peak) and peak > eps:
        return audio / peak.clamp_min(eps)
    return audio


def rms_normalize(audio: torch.Tensor, target_rms: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(audio * audio)).clamp_min(eps)
    if torch.isfinite(rms):
        return audio * (target_rms / rms)
    return audio


def normalize_audio(audio: torch.Tensor, mode: Optional[str]) -> torch.Tensor:
    if mode is None or mode == "none":
        return audio
    if mode == "peak":
        return peak_normalize(audio)
    if mode == "rms":
        return rms_normalize(audio)
    raise ValueError(f"Unknown normalization mode: {mode}")


def stable_int(text: str) -> int:
    import hashlib

    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def resample_audio(audio: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Resample FloatTensor[C, T] using torch interpolation.

    This intentionally avoids torchaudio. It is a simple linear interpolator,
    which is enough for sample-rate augmentation and keeps the dependency set
    small. The duration is preserved: T_new = round(T * new_sr / orig_sr).
    """
    orig_sr = int(orig_sr)
    new_sr = int(new_sr)
    if orig_sr <= 0 or new_sr <= 0:
        raise ValueError(f"Sample rates must be positive, got orig_sr={orig_sr}, new_sr={new_sr}")
    if orig_sr == new_sr:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shape [C, T], got {tuple(audio.shape)}")
    current_len = int(audio.shape[-1])
    if current_len <= 1:
        return audio
    target_len = max(1, int(round(current_len * float(new_sr) / float(orig_sr))))
    x = audio.unsqueeze(0)  # [1, C, T]
    y = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return y.squeeze(0).contiguous()


def random_augmented_sample_rate(
    native_sr: int,
    min_factor: float = 0.75,
    max_factor: float = 1.125,
    round_to: Optional[int] = None,
) -> tuple[int, float]:
    """Return a random sample rate near native_sr and its factor.

    Defaults make 32 kHz become approximately 24 kHz..36 kHz.
    If round_to is set, the rate is rounded to that grid, e.g. round_to=1000.
    For low sample rates, leave round_to=None.
    """
    native_sr = int(native_sr)
    if native_sr <= 0:
        raise ValueError(f"native_sr must be positive, got {native_sr}")
    if min_factor <= 0 or max_factor <= 0:
        raise ValueError("Sample-rate augmentation factors must be positive")
    if min_factor > max_factor:
        raise ValueError(f"sr_aug_min_factor={min_factor} > sr_aug_max_factor={max_factor}")

    factor = random.uniform(float(min_factor), float(max_factor))
    new_sr = int(round(native_sr * factor))
    if round_to is not None and int(round_to) > 1:
        q = int(round_to)
        new_sr = int(round(new_sr / q) * q)
    new_sr = max(1, int(new_sr))
    return new_sr, float(new_sr) / float(native_sr)


def crop_or_pad_seconds(
    audio: torch.Tensor,
    sr: int,
    clip_seconds: Optional[float],
    mode: str = "random",
    fill_value: float = 0.0,
    deterministic_key: Optional[str] = None,
) -> tuple[torch.Tensor, int]:
    if clip_seconds is None:
        return audio, int(audio.shape[-1])
    target_len = max(1, int(round(float(clip_seconds) * int(sr))))
    current_len = int(audio.shape[-1])
    if current_len == target_len:
        return audio, current_len
    if current_len < target_len:
        pad_len = target_len - current_len
        pad = torch.full((audio.shape[0], pad_len), float(fill_value), dtype=audio.dtype)
        return torch.cat([audio, pad], dim=-1), current_len
    max_start = current_len - target_len
    if mode == "first":
        start = 0
    elif mode == "center":
        start = max_start // 2
    elif mode == "deterministic":
        start = stable_int(deterministic_key or str(current_len)) % (max_start + 1)
    elif mode == "random":
        start = random.randint(0, max_start)
    else:
        raise ValueError(f"Unknown crop mode: {mode}")
    return audio[:, start : start + target_len], target_len


def make_multihot(label_indices: Sequence[int], num_classes: int) -> torch.Tensor:
    y = torch.zeros(int(num_classes), dtype=torch.float32)
    for idx in label_indices:
        i = int(idx)
        if 0 <= i < num_classes:
            y[i] = 1.0
    return y


def read_classes(classes_csv: str | Path) -> tuple[list[str], dict[tuple[str, int], int]]:
    rows = read_csv(classes_csv)
    class_names: list[str] = []
    global_index: dict[tuple[str, int], int] = {}
    for row in rows:
        dataset = row["dataset"]
        local_index = int(row["class_index"])
        class_name = row["class_name"]
        global_index[(dataset, local_index)] = len(class_names)
        class_names.append(f"{dataset}:{class_name}")
    return class_names, global_index


class Bioacoustic_Dataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        classes_csv: str | Path,
        datasets: Optional[Sequence[str]] = None,
        splits: Optional[Sequence[str]] = None,
        target_sample_rate: Optional[int] = None,
        clip_seconds: Optional[float] = None,
        crop_mode: str = "random",
        normalize: Optional[str] = "none",
        max_samples: Optional[int] = None,
        aug: bool = False,
        sr_aug_prob: float = 1.0,
        sr_aug_min_factor: float = 0.75,
        sr_aug_max_factor: float = 1.125,
        sr_aug_round_to: Optional[int] = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.classes_csv = Path(classes_csv)
        rows = read_csv(self.manifest_csv)
        if datasets is not None:
            allowed = {str(x) for x in datasets}
            rows = [row for row in rows if row["dataset"] in allowed]
        if splits is not None:
            allowed = {str(x) for x in splits}
            rows = [row for row in rows if row.get("split", "") in allowed]
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        if not rows:
            raise ValueError(f"No rows left after filtering manifest: {manifest_csv}")
        self.rows = rows
        self.class_names, self.global_class_index = read_classes(self.classes_csv)
        self.num_classes = len(self.class_names)
        self.target_sample_rate = target_sample_rate
        self.clip_seconds = clip_seconds
        self.crop_mode = crop_mode
        self.normalize = normalize
        self.aug = bool(aug)
        self.sr_aug_prob = float(sr_aug_prob)
        self.sr_aug_min_factor = float(sr_aug_min_factor)
        self.sr_aug_max_factor = float(sr_aug_max_factor)
        self.sr_aug_round_to = sr_aug_round_to
        if not (0.0 <= self.sr_aug_prob <= 1.0):
            raise ValueError(f"sr_aug_prob must be in [0, 1], got {self.sr_aug_prob}")
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_array_sample(self, row: dict[str, str]) -> tuple[torch.Tensor, int]:
        path = row.get("array_path", "")
        if not path:
            raise ValueError("array_path is empty")
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r")
        index = int(row["array_index"])
        audio = as_mono_float_tensor(self._arrays[path][index])
        sr_text = row.get("native_sample_rate", "")
        if not sr_text:
            raise ValueError(f"native_sample_rate is missing for array sample {row.get('global_id')}")
        return audio, int(float(sr_text))

    def _load_audio_file(self, row: dict[str, str]) -> tuple[torch.Tensor, int]:
        path = row.get("audio_path", "")
        if not path:
            raise ValueError("audio_path is empty")
        audio, sr = sf.read(path, always_2d=False)
        return as_mono_float_tensor(audio), int(sr)

    def _maybe_resample(self, audio: torch.Tensor, native_sr: int) -> tuple[torch.Tensor, int, bool, float]:
        """Apply fixed target resampling or random SR augmentation.

        Priority:
          1. If aug=True and the random probability passes, choose a random SR.
          2. Else if target_sample_rate is set, resample to that fixed SR.
          3. Else keep the native SR.
        """
        native_sr = int(native_sr)
        applied = False
        factor = 1.0
        out_sr = native_sr

        if self.aug and random.random() < self.sr_aug_prob:
            out_sr, factor = random_augmented_sample_rate(
                native_sr,
                min_factor=self.sr_aug_min_factor,
                max_factor=self.sr_aug_max_factor,
                round_to=self.sr_aug_round_to,
            )
            if out_sr != native_sr:
                audio = resample_audio(audio, native_sr, out_sr)
                applied = True
        elif self.target_sample_rate is not None and int(self.target_sample_rate) != native_sr:
            out_sr = int(self.target_sample_rate)
            audio = resample_audio(audio, native_sr, out_sr)
            factor = float(out_sr) / float(native_sr)
            applied = True

        return audio, out_sr, applied, factor

    def get_label_indices(self, index: int, global_labels: bool = True) -> list[int]:
        row = self.rows[int(index)]
        local_indices = parse_int_list(row.get("label_indices", ""))
        if not global_labels:
            return local_indices
        dataset = row["dataset"]
        out = []
        for local_index in local_indices:
            key = (dataset, int(local_index))
            if key in self.global_class_index:
                out.append(self.global_class_index[key])
        return sorted(set(out))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        if row.get("array_path"):
            audio, sr = self._load_array_sample(row)
        else:
            audio, sr = self._load_audio_file(row)

        native_sr = int(sr)
        audio, sample_rate, sr_aug_applied, sr_aug_factor = self._maybe_resample(audio, native_sr)
        audio, original_length = crop_or_pad_seconds(
            audio,
            sample_rate,
            self.clip_seconds,
            mode=self.crop_mode,
            deterministic_key=row.get("global_id") or row.get("sample_id"),
        )
        audio = normalize_audio(audio, self.normalize)
        labels = self.get_label_indices(index, global_labels=True)
        return {
            "audio": audio,
            "sample_rate": int(sample_rate),
            "native_sample_rate": native_sr,
            "augmented_sample_rate": int(sample_rate),
            "sr_augmentation_applied": bool(sr_aug_applied),
            "sr_augmentation_factor": float(sr_aug_factor),
            "length": int(audio.shape[-1]),
            "original_length": int(original_length),
            "labels": make_multihot(labels, self.num_classes),
            "label_indices": labels,
            "dataset": row["dataset"],
            "split": row.get("split", ""),
            "sample_id": row.get("sample_id", ""),
            "metadata": row,
        }


def collate_audio_samples(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    audios = [item["audio"] for item in batch]
    sample_rates = [int(item["sample_rate"]) for item in batch]
    same_sr = len(set(sample_rates)) == 1
    same_channels = len({int(x.shape[0]) for x in audios}) == 1
    lengths = torch.tensor([int(x.shape[-1]) for x in audios], dtype=torch.long)
    if same_sr and same_channels:
        max_len = int(lengths.max().item())
        channels = int(audios[0].shape[0])
        audio_batch = torch.zeros(len(audios), channels, max_len, dtype=audios[0].dtype)
        for i, x in enumerate(audios):
            audio_batch[i, :, : x.shape[-1]] = x
    else:
        audio_batch = audios
    return {
        "audio": audio_batch,
        "lengths": lengths,
        "sample_rate": sample_rates[0] if same_sr else sample_rates,
        "native_sample_rate": [int(item["native_sample_rate"]) for item in batch],
        "augmented_sample_rate": [int(item["augmented_sample_rate"]) for item in batch],
        "sr_augmentation_applied": [bool(item["sr_augmentation_applied"]) for item in batch],
        "sr_augmentation_factor": torch.tensor([float(item["sr_augmentation_factor"]) for item in batch], dtype=torch.float32),
        "labels": torch.stack([item["labels"] for item in batch]),
        "label_indices": [item["label_indices"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "split": [item["split"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def compute_class_counts(dataset: Bioacoustic_Dataset, max_scan: Optional[int] = None) -> torch.Tensor:
    n = len(dataset) if max_scan is None else min(len(dataset), int(max_scan))
    counts = torch.zeros(dataset.num_classes, dtype=torch.float64)
    for i in range(n):
        for idx in dataset.get_label_indices(i, global_labels=True):
            counts[int(idx)] += 1.0
    return counts


def compute_pos_weight(dataset: Bioacoustic_Dataset, max_scan: Optional[int] = None) -> torch.Tensor:
    n = len(dataset) if max_scan is None else min(len(dataset), int(max_scan))
    counts = compute_class_counts(dataset, max_scan=max_scan)
    return ((float(n) - counts) / counts.clamp_min(1.0)).float()


def make_weighted_sampler(
    dataset: Bioacoustic_Dataset,
    power: float = 0.5,
    max_scan: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    n = len(dataset) if max_scan is None else min(len(dataset), int(max_scan))
    counts = compute_class_counts(dataset, max_scan=n).clamp_min(1.0)
    inv = (1.0 / counts) ** float(power)
    weights = torch.zeros(n, dtype=torch.float64)
    empty_weight = float(inv.mean().item())
    for i in range(n):
        indices = dataset.get_label_indices(i, global_labels=True)
        if indices:
            weights[i] = inv[torch.tensor(indices, dtype=torch.long)].mean()
        else:
            weights[i] = empty_weight
    weights = weights / weights.mean().clamp_min(1e-12)
    return WeightedRandomSampler(weights=weights, num_samples=n, replacement=replacement)


def make_loader(
    dataset: Bioacoustic_Dataset,
    batch_size: int,
    shuffle: bool = True,
    balanced: bool = False,
    num_workers: int = 4,
    sampler_power: float = 0.5,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    sampler = None
    if balanced:
        sampler = make_weighted_sampler(dataset, power=sampler_power)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) and int(num_workers) > 0,
        collate_fn=collate_audio_samples,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test a canonical bioacoustic manifest dataset.")
    p.add_argument("--root", default="/mnt/volDISI_conci_Datasets/bioacoustics")
    p.add_argument("--manifest", default=None)
    p.add_argument("--classes", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--splits", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--aug", action="store_true", help="Enable random sample-rate augmentation.")
    p.add_argument("--sr-aug-prob", type=float, default=1.0)
    p.add_argument("--sr-aug-min-factor", type=float, default=0.75)
    p.add_argument("--sr-aug-max-factor", type=float, default=1.125)
    p.add_argument("--sr-aug-round-to", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifests" / "all_samples.csv"
    classes = Path(args.classes) if args.classes else root / "manifests" / "all_classes.csv"
    ds = Bioacoustic_Dataset(
        manifest,
        classes,
        datasets=args.datasets,
        splits=args.splits,
        target_sample_rate=None,
        max_samples=args.max_samples,
        aug=args.aug,
        sr_aug_prob=args.sr_aug_prob,
        sr_aug_min_factor=args.sr_aug_min_factor,
        sr_aug_max_factor=args.sr_aug_max_factor,
        sr_aug_round_to=args.sr_aug_round_to,
    )
    print(f"samples={len(ds)} classes={ds.num_classes}")
    for i in range(min(3, len(ds))):
        item = ds[i]
        print(
            i,
            item["dataset"],
            item["sample_id"],
            item["audio"].shape,
            "native_sr=",
            item["native_sample_rate"],
            "sample_rate=",
            item["sample_rate"],
            "factor=",
            round(float(item["sr_augmentation_factor"]), 4),
            item["label_indices"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
