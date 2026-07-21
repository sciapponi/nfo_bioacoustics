from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import wave
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from sfi_bioacoustic_models import (
    build_model,
    count_trainable_parameters,
    fft_resample_1d,
    lowpass_fft_1d,
    make_random_upsampled_view,
    parameter_report,
)

DATASET_NAME = "watkins"
TRAIN_SPLITS = {"train", "training"}
VAL_SPLITS = {"val", "valid", "validation", "dev", "development"}
TEST_SPLITS = {"test", "testing", "eval", "evaluation"}
AUDIO_EXTENSIONS_DEFAULT = ".wav,.flac,.aif,.aiff"

MANIFEST_FIELDS = [
    "global_id",
    "dataset",
    "split",
    "sample_id",
    "audio_path",
    "native_sample_rate",
    "duration_seconds",
    "label",
    "species",
    "source_id",
]
CLASS_FIELDS = ["dataset", "class_index", "class_name", "source_name"]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_float_0_1(text: str) -> float:
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)


def stable_sort_key(text: str, seed: int = 0) -> str:
    return hashlib.sha1(f"{text}:{seed}".encode("utf-8")).hexdigest()


def parse_extensions(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if not part.startswith("."):
            part = "." + part
        out.append(part)
    if not out:
        raise ValueError("At least one audio extension is required")
    return tuple(dict.fromkeys(out))


def parse_name_list(text: str) -> list[str]:
    out = [p.strip().lower() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if not out:
        raise ValueError("Expected at least one name")
    return list(dict.fromkeys(out))


def parse_int_list(text: str) -> list[int]:
    if not str(text).strip():
        return []
    return [int(p.strip()) for p in str(text).replace(";", ",").split(",") if p.strip()]


def parse_float_list(text: str) -> list[float]:
    if not str(text).strip():
        return []
    return [float(p.strip()) for p in str(text).replace(";", ",").split(",") if p.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def progress_iter(iterable: Any, args: argparse.Namespace, desc: str, total: Optional[int] = None, leave: bool = False):
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True, disable=bool(args.no_progress))


def set_progress_postfix(iterator: Any, values: dict[str, float]) -> None:
    if hasattr(iterator, "set_postfix"):
        iterator.set_postfix({k: f"{v:.4f}" for k, v in values.items()})


def amp_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# -----------------------------------------------------------------------------
# Audio I/O. Uses soundfile when available, built-in wave fallback for PCM WAV.
# -----------------------------------------------------------------------------


def audio_info(path: Path) -> tuple[int, float, int]:
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(str(path))
        sr = int(info.samplerate)
        frames = int(info.frames)
        return sr, float(frames) / float(sr) if sr > 0 else 0.0, frames
    except Exception:
        pass
    try:
        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            frames = int(wf.getnframes())
            return sr, float(frames) / float(sr) if sr > 0 else 0.0, frames
    except Exception as exc:
        raise RuntimeError(f"Could not read audio header for {path}. Install soundfile for non-WAV formats.") from exc


def load_audio_mono(path: str | Path) -> tuple[torch.Tensor, int]:
    path = Path(path)
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T).float().mean(dim=0, keepdim=True)
        return wav, int(sr)
    except Exception:
        pass
    try:
        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            channels = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            frames = int(wf.getnframes())
            raw = wf.readframes(frames)
        if sampwidth == 1:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            arr = (arr - 128.0) / 128.0
        elif sampwidth == 2:
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sampwidth == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            signed = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
            arr = signed.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width {sampwidth} bytes in {path}")
        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)
        wav = torch.from_numpy(arr).float().unsqueeze(0)
        return wav, sr
    except Exception as exc:
        raise RuntimeError(f"Could not load audio file {path}. Install soundfile for non-WAV formats.") from exc


# -----------------------------------------------------------------------------
# Watkins-style manifest builder
# -----------------------------------------------------------------------------


def infer_split_from_filename(path: Path) -> str | None:
    import re
    name = path.stem.lower()
    tokens = {tok for tok in re.split(r"[^a-z0-9]+", name) if tok}
    if tokens & TEST_SPLITS:
        return "test"
    if tokens & VAL_SPLITS:
        return "val"
    if tokens & TRAIN_SPLITS:
        return "train"
    if name.startswith("train") or name.endswith("train"):
        return "train"
    if name.startswith("test") or name.endswith("test"):
        return "test"
    if name.startswith("val") or name.endswith("val") or name.startswith("valid") or name.endswith("valid"):
        return "val"
    return None


def discover_audio_files(root: Path, extensions: Sequence[str]) -> list[Path]:
    blocked = {".git", ".venv", "__pycache__", "runs", "manifests", "wandb"}
    root = root.resolve()
    exts = {e.lower() for e in extensions}
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root)
        if len(rel.parts) < 2:
            continue
        if any(part in blocked for part in rel.parts[:-1]):
            continue
        files.append(p.resolve())
    return sorted(files, key=lambda x: str(x.relative_to(root)).lower())


def file_sha1(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def update_split_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    new = dict(row)
    old = str(new.get("split", ""))
    new["split"] = split
    parts = str(new["global_id"]).split(":")
    if len(parts) >= 3:
        parts[1] = split
        new["global_id"] = ":".join(parts)
    elif old:
        new["global_id"] = str(new["global_id"]).replace(f":{old}:", f":{split}:")
    return new


def stratified_move(source: list[dict[str, Any]], fraction: float, split_name: str, seed: int, minimum_total: int = 1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if fraction <= 0 or not source:
        return source, []
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        by_label[int(row["label"])].append(row)
    keep: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    for label, rows in by_label.items():
        rows = sorted(rows, key=lambda r: stable_sort_key(str(r["source_id"]), seed + label * 131))
        if len(rows) <= 1:
            keep.extend(rows)
            continue
        n_move = int(round(len(rows) * float(fraction)))
        if n_move <= 0 and len(rows) >= 3 and fraction > 0:
            n_move = 1
        n_move = min(n_move, len(rows) - 1)
        move_rows = rows[:n_move]
        keep_rows = rows[n_move:]
        keep.extend(keep_rows)
        moved.extend(update_split_row(r, split_name) for r in move_rows)
    if not moved and len(source) > 1 and minimum_total > 0:
        rows = sorted(source, key=lambda r: stable_sort_key(str(r["source_id"]), seed))
        moved = [update_split_row(rows[0], split_name)]
        keep = rows[1:]
    return keep, moved


def build_manifests(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    root = Path(args.root or args.watkins_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Missing root folder: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    use_validation = float(args.val_fraction) > 0.0

    manifest_dir = run_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_csv = manifest_dir / "train_samples.csv"
    val_csv = manifest_dir / "val_samples.csv"
    test_csv = manifest_dir / "test_samples.csv"
    classes_csv = manifest_dir / "classes.csv"

    files = discover_audio_files(root, parse_extensions(args.audio_extensions))
    if not files:
        raise RuntimeError(f"Found zero audio files under {root}")

    if args.remove_duplicates:
        seen: dict[str, Path] = {}
        kept: list[Path] = []
        for p in progress_iter(files, args, desc="hash duplicates", total=len(files), leave=False):
            h = file_sha1(p)
            if h not in seen:
                seen[h] = p
                kept.append(p)
        files = kept

    all_species = [p.relative_to(root).parts[0] for p in files]
    counts_by_species = Counter(all_species)
    allowed_species = sorted([s for s, n in counts_by_species.items() if n >= int(args.min_samples_per_class)])
    if not allowed_species:
        raise RuntimeError("No class remains after --min-samples-per-class filtering")
    species_to_idx = {s: i for i, s in enumerate(allowed_species)}
    allowed = set(allowed_species)
    files = [p for p in files if p.relative_to(root).parts[0] in allowed]

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    unknown_rows: list[dict[str, Any]] = []
    sr_counts: Counter[int] = Counter()
    duration_by_sr: dict[int, list[float]] = defaultdict(list)

    iterator = progress_iter(files, args, desc="scan audio headers", total=len(files), leave=False)
    for p in iterator:
        rel = p.relative_to(root)
        species = rel.parts[0]
        label = species_to_idx[species]
        sr, duration, frames = audio_info(p)
        if sr <= 0 or frames <= 0:
            continue
        split = infer_split_from_filename(p)
        sample_id = str(rel.with_suffix("")).replace("/", "__").replace("\\", "__")
        row = {
            "global_id": f"{DATASET_NAME}:{split or 'unknown'}:{sample_id}",
            "dataset": DATASET_NAME,
            "split": split or "unknown",
            "sample_id": sample_id,
            "audio_path": str(p),
            "native_sample_rate": int(sr),
            "duration_seconds": f"{duration:.9f}",
            "label": int(label),
            "species": species,
            "source_id": str(rel),
        }
        sr_counts[int(sr)] += 1
        duration_by_sr[int(sr)].append(float(duration))
        if split in rows_by_split:
            rows_by_split[split].append(row)
        else:
            unknown_rows.append(row)

    if unknown_rows:
        if args.unknown_split == "error" and not args.paper_protocol:
            example = unknown_rows[0]["source_id"]
            raise RuntimeError(f"Cannot infer split for {example}. Use --unknown-split derive/train or --paper-protocol.")
        if args.unknown_split == "train" and not args.paper_protocol:
            rows_by_split["train"].extend(update_split_row(r, "train") for r in unknown_rows)
        else:
            # Derive missing splits from unknown rows. If validation is disabled
            # (--val-fraction 0.0), only train/test are derived.
            pool = [update_split_row(r, "train") for r in unknown_rows]
            keep, test = stratified_move(pool, args.test_fraction, "test", args.seed + 17)
            if use_validation:
                keep, val = stratified_move(keep, args.val_fraction, "val", args.seed + 31)
            else:
                val = []
            rows_by_split["train"].extend(keep)
            rows_by_split["val"].extend(val)
            rows_by_split["test"].extend(test)

    # If validation is disabled, any explicit val files are folded back into the
    # training set. This lets --val-fraction 0.0 mean a true train/test run.
    if not use_validation and rows_by_split["val"]:
        moved = len(rows_by_split["val"])
        rows_by_split["train"].extend(update_split_row(r, "train") for r in rows_by_split["val"])
        rows_by_split["val"] = []
        print(f"validation disabled: moved {moved} explicit val rows back into train", flush=True)

    # Watkins common case: train/test filenames exist, val does not. Carve val
    # from train only when validation is enabled.
    if use_validation and not rows_by_split["val"] and args.val_fraction > 0:
        rows_by_split["train"], moved_val = stratified_move(rows_by_split["train"], args.val_fraction, "val", args.seed + 101)
        rows_by_split["val"].extend(moved_val)
        print(f"created missing val split from train: moved={len(moved_val)} remaining_train={len(rows_by_split['train'])}", flush=True)

    if not rows_by_split["test"] and args.test_fraction > 0:
        rows_by_split["train"], moved_test = stratified_move(rows_by_split["train"], args.test_fraction, "test", args.seed + 211)
        rows_by_split["test"].extend(moved_test)
        print(f"created missing test split from train: moved={len(moved_test)} remaining_train={len(rows_by_split['train'])}", flush=True)

    required_splits = ("train", "test") if not use_validation else ("train", "val", "test")
    for split in required_splits:
        if not rows_by_split[split]:
            raise RuntimeError(f"Split {split!r} has zero rows. Adjust split files or --unknown-split/--val-fraction/--test-fraction.")
    for split in ("train", "val", "test"):
        rows_by_split[split] = sorted(rows_by_split[split], key=lambda r: (int(r["label"]), str(r["source_id"])))

    class_rows = [
        {"dataset": DATASET_NAME, "class_index": idx, "class_name": species, "source_name": species}
        for species, idx in species_to_idx.items()
    ]
    write_csv(train_csv, rows_by_split["train"], MANIFEST_FIELDS)
    write_csv(val_csv, rows_by_split["val"], MANIFEST_FIELDS)
    write_csv(test_csv, rows_by_split["test"], MANIFEST_FIELDS)
    write_csv(classes_csv, class_rows, CLASS_FIELDS)

    sr_rows = []
    for sr, count in sorted(sr_counts.items()):
        durations = duration_by_sr[sr]
        sr_rows.append({
            "sample_rate": sr,
            "count": int(count),
            "mean_duration_seconds": float(np.mean(durations)) if durations else 0.0,
            "min_duration_seconds": float(np.min(durations)) if durations else 0.0,
            "max_duration_seconds": float(np.max(durations)) if durations else 0.0,
        })
    write_csv(manifest_dir / "sample_rates.csv", sr_rows, ["sample_rate", "count", "mean_duration_seconds", "min_duration_seconds", "max_duration_seconds"])

    class_count_rows = []
    for species in allowed_species:
        idx = species_to_idx[species]
        row = {"class_index": idx, "class_name": species}
        for split in ("train", "val", "test"):
            row[split] = sum(1 for r in rows_by_split[split] if int(r["label"]) == idx)
        class_count_rows.append(row)
    write_csv(manifest_dir / "class_counts_by_split.csv", class_count_rows, ["class_index", "class_name", "train", "val", "test"])

    summary = {
        "root": str(root),
        "use_validation": bool(use_validation),
        "class_count": len(allowed_species),
        "class_names": allowed_species,
        "split_sizes": {k: len(v) for k, v in rows_by_split.items()},
        "sample_rate_counts": {str(k): int(v) for k, v in sorted(sr_counts.items())},
        "num_audio_files_used": sum(len(v) for v in rows_by_split.values()),
        "num_audio_files_discovered": len(files),
        "min_samples_per_class": int(args.min_samples_per_class),
        "remove_duplicates": bool(args.remove_duplicates),
    }
    safe_json_dump(summary, manifest_dir / "manifest_summary.json")

    return {
        "train_manifest": train_csv,
        "val_manifest": val_csv if use_validation else None,
        "test_manifest": test_csv,
        "classes_csv": classes_csv,
        "use_validation": bool(use_validation),
        "class_count": len(allowed_species),
        "class_names": allowed_species,
        "split_sizes": summary["split_sizes"],
        "sample_rate_counts": summary["sample_rate_counts"],
    }


# -----------------------------------------------------------------------------
# Dataset and balanced sampling
# -----------------------------------------------------------------------------


def crop_or_pad(wav: torch.Tensor, sr: int, clip_seconds: float, crop_mode: str, pad_mode: str, rng: random.Random) -> torch.Tensor:
    target = max(1, int(round(float(sr) * float(clip_seconds))))
    n = int(wav.shape[-1])
    if n == target:
        return wav
    if n < target:
        if pad_mode == "zero":
            left = (target - n) // 2
            right = target - n - left
            return F.pad(wav, (left, right))
        if pad_mode == "repeat":
            reps = int(math.ceil(target / max(n, 1)))
            return wav.repeat(1, reps)[:, :target]
        raise ValueError(f"Unsupported pad_mode={pad_mode}")
    if crop_mode == "random":
        start = rng.randint(0, n - target)
    elif crop_mode == "center":
        start = (n - target) // 2
    else:
        raise ValueError(f"Unsupported crop_mode={crop_mode}")
    return wav[:, start : start + target]


def normalize_audio(wav: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return wav
    if mode == "peak":
        return wav / wav.abs().amax().clamp_min(1e-8)
    if mode == "rms":
        rms = wav.square().mean().sqrt().clamp_min(1e-8)
        return wav / rms * 0.1
    if mode == "standardize":
        return (wav - wav.mean()) / wav.std(unbiased=False).clamp_min(1e-5)
    raise ValueError(f"Unsupported normalize mode: {mode}")


class WatkinsAudioDataset(Dataset):
    def __init__(self, manifest_csv: Path, clip_seconds: float, normalize: str, crop_mode: str, pad_mode: str, seed: int) -> None:
        self.rows = read_csv(manifest_csv)
        if not self.rows:
            raise RuntimeError(f"Empty manifest: {manifest_csv}")
        self.clip_seconds = float(clip_seconds)
        self.normalize = str(normalize)
        self.crop_mode = str(crop_mode)
        self.pad_mode = str(pad_mode)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        wav, sr = load_audio_mono(row["audio_path"])
        native_sr = int(row.get("native_sample_rate") or sr)
        if int(sr) != native_sr:
            native_sr = int(sr)
        rng = random.Random(self.seed + 1_000_003 * idx)
        wav = crop_or_pad(wav, native_sr, self.clip_seconds, self.crop_mode, self.pad_mode, rng)
        wav = normalize_audio(wav, self.normalize)
        return {
            "audio": wav,
            "sample_rate": int(native_sr),
            "label": int(row["label"]),
            "species": row["species"],
            "path": row["audio_path"],
            "metadata": row,
        }


def collate_audio(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audio": [item["audio"] for item in batch],
        "sample_rate": [int(item["sample_rate"]) for item in batch],
        "label": torch.tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "species": [item["species"] for item in batch],
        "path": [item["path"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


class ClassBalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: WatkinsAudioDataset, batch_size: int, steps_per_epoch: int, sampler_power: float, seed: int) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.sampler_power = float(sampler_power)
        self.seed = int(seed)
        self.epoch = 0
        counts = Counter(int(row["label"]) for row in dataset.rows)
        weights = [(1.0 / max(counts[int(row["label"])], 1)) ** self.sampler_power for row in dataset.rows]
        w = torch.tensor(weights, dtype=torch.float64)
        self.weights = w / w.sum().clamp_min(1e-12)
        self.indices = list(range(len(dataset)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable[list[int]]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + 1_000_003 * self.epoch)
        for _ in range(self.steps_per_epoch):
            sampled = torch.multinomial(self.weights, self.batch_size, replacement=True, generator=gen).tolist()
            order = torch.randperm(len(sampled), generator=gen).tolist()
            yield [self.indices[sampled[i]] for i in order]


def resolve_steps_per_epoch(train_size: int, batch_size: int, requested_steps: int) -> int:
    if int(requested_steps) > 0:
        return int(requested_steps)
    return max(1, int(math.ceil(float(train_size) / float(max(1, batch_size)))))


def build_dataloaders(manifest_info: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, Optional[DataLoader], DataLoader]:
    train_ds = WatkinsAudioDataset(manifest_info["train_manifest"], args.clip_seconds, args.normalize, "random", args.pad_mode, args.seed)
    val_loader: Optional[DataLoader] = None
    if bool(manifest_info.get("use_validation", True)) and manifest_info.get("val_manifest") is not None:
        val_ds = WatkinsAudioDataset(manifest_info["val_manifest"], args.clip_seconds, args.normalize, "center", args.pad_mode, args.seed)
    else:
        val_ds = None
    test_ds = WatkinsAudioDataset(manifest_info["test_manifest"], args.clip_seconds, args.normalize, "center", args.pad_mode, args.seed)
    steps = resolve_steps_per_epoch(len(train_ds), args.batch_size, args.steps_per_epoch)
    args._resolved_steps_per_epoch = steps
    args._train_samples_per_epoch_basis = len(train_ds)
    train_sampler = ClassBalancedBatchSampler(train_ds, args.batch_size, steps, args.sampler_power, args.seed)
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_audio,
    }
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **kwargs)
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Metrics and losses
# -----------------------------------------------------------------------------


def class_weights_from_manifest(train_manifest: Path, classes: int, cap: float, device: torch.device) -> tuple[torch.Tensor, dict[str, int]]:
    rows = read_csv(train_manifest)
    counts = torch.zeros(classes, dtype=torch.float64)
    for row in rows:
        idx = int(row["label"])
        if 0 <= idx < classes:
            counts[idx] += 1
    total = counts.sum().clamp_min(1.0)
    weights = total / (float(classes) * counts.clamp_min(1.0))
    weights[counts <= 0] = 0.0
    return weights.float().clamp(max=float(cap)).to(device), {str(i): int(counts[i].item()) for i in range(classes)}


def logits_from_output(out: dict[str, Any]) -> torch.Tensor:
    logits = out["logits"]
    if isinstance(logits, dict):
        if "all" in logits:
            return logits["all"]
        if len(logits) == 1:
            return next(iter(logits.values()))
        raise KeyError(f"Cannot infer logits from keys {list(logits)}")
    return logits


def macro_micro_weighted_metrics(logits: torch.Tensor, target: torch.Tensor, classes: int) -> dict[str, float]:
    pred = logits.argmax(dim=-1).detach().cpu()
    target_cpu = target.detach().cpu().long()
    tp = torch.zeros(classes, dtype=torch.float64)
    fp = torch.zeros(classes, dtype=torch.float64)
    fn = torch.zeros(classes, dtype=torch.float64)
    support = torch.zeros(classes, dtype=torch.float64)
    for c in range(classes):
        pc = pred == c
        tc = target_cpu == c
        tp[c] = (pc & tc).sum().double()
        fp[c] = (pc & ~tc).sum().double()
        fn[c] = (~pc & tc).sum().double()
        support[c] = tc.sum().double()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    supported = support > 0
    macro_f1 = float(f1[supported].mean().item()) if bool(supported.any()) else 0.0
    weighted_f1 = float((f1 * support / support.sum().clamp_min(1.0)).sum().item()) if support.sum() > 0 else 0.0
    acc = float((pred == target_cpu).float().mean().item()) if target_cpu.numel() else 0.0
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_p = micro_tp / (micro_tp + micro_fp).clamp_min(1.0)
    micro_r = micro_tp / (micro_tp + micro_fn).clamp_min(1.0)
    micro_f1 = float((2 * micro_p * micro_r / (micro_p + micro_r).clamp_min(1e-12)).item())
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "micro_f1": micro_f1}


def per_class_metrics(logits: torch.Tensor, target: torch.Tensor, class_names: Sequence[str]) -> list[dict[str, Any]]:
    classes = len(class_names)
    pred = logits.argmax(dim=-1).detach().cpu()
    y = target.detach().cpu().long()
    rows: list[dict[str, Any]] = []
    for c, name in enumerate(class_names):
        pc = pred == c
        tc = y == c
        tp = float((pc & tc).sum().item())
        fp = float((pc & ~tc).sum().item())
        fn = float((~pc & tc).sum().item())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append({
            "class_index": c,
            "class_name": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(tc.sum().item()),
        })
    return rows


def compact_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    return (
        f"{prefix}loss={metrics.get('loss', 0.0):.4f} "
        f"{prefix}acc={metrics.get('accuracy', 0.0):.4f} "
        f"{prefix}macro_f1={metrics.get('macro_f1', 0.0):.4f} "
        f"{prefix}weighted_f1={metrics.get('weighted_f1', 0.0):.4f}"
    )


def batch_audio_to_device(batch: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    return [x.to(device, non_blocking=True).float() for x in batch["audio"]]


def symmetric_kl_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits_a, dim=-1)
    logq = F.log_softmax(logits_b, dim=-1)
    p = logp.exp()
    q = logq.exp()
    return 0.5 * (F.kl_div(logp, q.detach(), reduction="batchmean") + F.kl_div(logq, p.detach(), reduction="batchmean"))


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    class_weights: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    classes: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    sampler = getattr(loader, "batch_sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    use_amp = bool(args.amp and device.type == "cuda")
    ce_weight = class_weights if args.use_class_weights else None
    totals = Counter()
    steps = 0
    iterator = progress_iter(loader, args, desc=f"{model_name} train {epoch:03d}", total=len(loader), leave=False)
    for batch in iterator:
        steps += 1
        target = batch["label"].to(device, non_blocking=True)
        audio = batch_audio_to_device(batch, device)
        sr = [int(x) for x in batch["sample_rate"]]
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(use_amp):
            out = model(audio, sr, return_hidden=False)
            logits = logits_from_output(out)
            ce = F.cross_entropy(logits, target, weight=ce_weight)
            loss = ce
            consistency = logits.new_zeros(())
            if model_name in {"sfi_scatnet", "sfi_fno"} and args.sampling_consistency_weight > 0:
                aug_audio, aug_sr = make_random_upsampled_view(
                    audio,
                    sr,
                    min_factor=args.consistency_min_factor,
                    max_factor=args.consistency_max_factor,
                    round_to=args.consistency_round_to,
                    max_hz=args.consistency_max_hz,
                )
                out_aug = model(aug_audio, aug_sr, return_hidden=False)
                logits_aug = logits_from_output(out_aug)
                emb = out["embedding"]
                emb_aug = out_aug["embedding"]
                emb_loss = (1.0 - F.cosine_similarity(emb, emb_aug, dim=-1)).mean()
                pred_loss = symmetric_kl_logits(logits, logits_aug)
                consistency = pred_loss + emb_loss
                loss = loss + args.sampling_consistency_weight * consistency
        if use_amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["classification_loss"] += float(ce.detach().cpu())
        totals["sampling_consistency"] += float(consistency.detach().cpu())
        set_progress_postfix(iterator, {"loss": totals["loss"] / steps, "ce": totals["classification_loss"] / steps})
    return {k: v / max(steps, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    classes: int,
    desc: str,
    transform: Optional[tuple[str, float]] = None,
) -> dict[str, Any]:
    model.eval()
    ce_weight = class_weights if args.use_class_weights else None
    totals = Counter()
    steps = 0
    logits_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []
    sr_list_all: list[int] = []
    path_list: list[str] = []
    iterator = progress_iter(loader, args, desc=desc, total=len(loader), leave=False)
    for batch in iterator:
        steps += 1
        target = batch["label"].to(device, non_blocking=True)
        audio = batch_audio_to_device(batch, device)
        sr = [int(x) for x in batch["sample_rate"]]
        if transform is not None:
            kind, value = transform
            if kind == "sample_rate":
                new_audio = []
                new_sr = []
                for x, s in zip(audio, sr):
                    new_audio.append(fft_resample_1d(x, s, int(value)))
                    new_sr.append(int(value))
                audio, sr = new_audio, new_sr
            elif kind == "cutoff":
                audio = [lowpass_fft_1d(x, s, float(value)) for x, s in zip(audio, sr)]
            else:
                raise ValueError(f"Unknown transform {transform}")
        out = model(audio, sr, return_hidden=False)
        logits = logits_from_output(out)
        loss = F.cross_entropy(logits, target, weight=ce_weight)
        totals["loss"] += float(loss.detach().cpu())
        logits_list.append(logits.detach().cpu())
        target_list.append(target.detach().cpu())
        sr_list_all.extend(sr)
        path_list.extend([str(p) for p in batch["path"]])
        set_progress_postfix(iterator, {"loss": totals["loss"] / steps})
    logits_cat = torch.cat(logits_list, dim=0) if logits_list else torch.empty(0, classes)
    target_cat = torch.cat(target_list, dim=0) if target_list else torch.empty(0, dtype=torch.long)
    metrics = macro_micro_weighted_metrics(logits_cat, target_cat, classes)
    metrics["loss"] = totals["loss"] / max(steps, 1)
    return {"metrics": metrics, "logits": logits_cat, "target": target_cat, "sample_rate": sr_list_all, "path": path_list}


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    max_points: int,
    desc: str,
) -> dict[str, Any]:
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    sample_rates: list[np.ndarray] = []
    paths: list[str] = []
    seen = 0
    unlimited = int(max_points) <= 0
    iterator = progress_iter(loader, args, desc=desc, total=len(loader), leave=False)
    for batch in iterator:
        audio = batch_audio_to_device(batch, device)
        sr = [int(x) for x in batch["sample_rate"]]
        out = model(audio, sr, return_hidden=False)
        logits = logits_from_output(out)
        emb = out["embedding"].detach().cpu().float().numpy()
        pred = logits.argmax(dim=-1).detach().cpu().numpy()
        lab = batch["label"].detach().cpu().numpy()
        sr_np = np.asarray(sr, dtype=np.int64)
        batch_paths = [str(p) for p in batch["path"]]
        if not unlimited:
            remaining = int(max_points) - seen
            if remaining <= 0:
                break
            emb = emb[:remaining]
            pred = pred[:remaining]
            lab = lab[:remaining]
            sr_np = sr_np[:remaining]
            batch_paths = batch_paths[:remaining]
        embeddings.append(emb)
        labels.append(lab)
        preds.append(pred)
        sample_rates.append(sr_np)
        paths.extend(batch_paths)
        seen += emb.shape[0]
        if not unlimited and seen >= int(max_points):
            break
    if not embeddings:
        return {"embedding": np.empty((0, 0)), "label": np.empty((0,), dtype=np.int64), "pred": np.empty((0,), dtype=np.int64), "sample_rate": np.empty((0,), dtype=np.int64), "path": []}
    return {
        "embedding": np.concatenate(embeddings, axis=0),
        "label": np.concatenate(labels, axis=0),
        "pred": np.concatenate(preds, axis=0),
        "sample_rate": np.concatenate(sample_rates, axis=0),
        "path": paths,
    }


# -----------------------------------------------------------------------------
# Plotting and result exports
# -----------------------------------------------------------------------------


def require_matplotlib() -> bool:
    return plt is not None


def categorical_palette(n: int) -> list[str]:
    base = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
        "darkblue", "darkorange", "darkgreen", "darkred", "indigo",
        "saddlebrown", "deeppink", "dimgray", "olive", "teal",
        "royalblue", "goldenrod", "seagreen", "firebrick", "mediumorchid",
        "peru", "hotpink", "slategray", "yellowgreen", "cadetblue",
    ]
    return [base[i % len(base)] for i in range(int(n))]


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    x = x / np.maximum(x.std(axis=0, keepdims=True), 1e-8)
    if x.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
    return coords[:, :2]


def reduce_embeddings_2d(x: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, str]:
    x = np.asarray(x, dtype=np.float64)
    method = str(method).lower()

    if method == "tsne":
        if x.shape[0] < 4:
            # t-SNE is not meaningful for extremely small point clouds.
            return pca_2d(x), "pca_too_few_points_for_tsne"
        try:
            from sklearn.manifold import TSNE  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "--plot-reducer tsne requires scikit-learn. Install it with: pip install scikit-learn"
            ) from exc

        xs = x - x.mean(axis=0, keepdims=True)
        xs = xs / np.maximum(xs.std(axis=0, keepdims=True), 1e-8)
        perplexity = min(30.0, max(2.0, float((x.shape[0] - 1) // 3)))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=int(seed),
        ).fit_transform(xs)
        return coords, "tsne"

    return pca_2d(x), "pca"


def sanitize_label(value: Any) -> str:
    return str(value).replace("_", " ")


def plot_feature_space(coords: np.ndarray, values: np.ndarray, title: str, out_path: Path, kind: str, class_names: Optional[Sequence[str]] = None) -> None:
    if not require_matplotlib() or coords.shape[0] == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=170)
    values = np.asarray(values)
    if kind == "class" and class_names is not None:
        colors = categorical_palette(len(class_names))
        for c, name in enumerate(class_names):
            mask = values.astype(np.int64) == c
            if np.any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.78, color=colors[c], label=sanitize_label(name), linewidths=0)
        ncol = 1 if len(class_names) <= 16 else 2 if len(class_names) <= 32 else 3
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=ncol)
    elif kind == "sample_rate":
        unique_rates = sorted({int(v) for v in values.tolist()})
        colors = categorical_palette(len(unique_rates))
        for i, sr in enumerate(unique_rates):
            mask = values.astype(np.int64) == sr
            if np.any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.78, color=colors[i], label=f"{sr} Hz", linewidths=0)
        ncol = 1 if len(unique_rates) <= 12 else 2
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=ncol)
    elif kind == "correct":
        labels = [(0, "wrong", "tab:red"), (1, "correct", "tab:green")]
        for value, label, color in labels:
            mask = values.astype(np.int64) == value
            if np.any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.78, color=color, label=label, linewidths=0)
        ax.legend(fontsize=7, loc="best", frameon=False)
    else:
        unique_values = sorted({str(v) for v in values.tolist()})
        colors = categorical_palette(len(unique_values))
        for i, val in enumerate(unique_values):
            mask = values.astype(str) == val
            if np.any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.78, color=colors[i], label=val, linewidths=0)
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title(title)
    ax.set_xlabel("projection dim 1")
    ax.set_ylabel("projection dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_embedding_table(features: dict[str, Any], coords: np.ndarray, class_names: Sequence[str], path: Path, reducer_name: str) -> None:
    rows: list[dict[str, Any]] = []
    labels = features["label"]
    preds = features["pred"]
    sample_rates = features["sample_rate"]
    paths = features["path"]
    fields = ["sample_index", "reducer", "projection_x", "projection_y", "label", "class_name", "prediction", "predicted_class_name", "correct", "sample_rate", "path"]
    for i in range(len(labels)):
        lab = int(labels[i])
        pred = int(preds[i])
        rows.append({
            "sample_index": i,
            "reducer": reducer_name,
            "projection_x": float(coords[i, 0]),
            "projection_y": float(coords[i, 1]),
            "label": lab,
            "class_name": class_names[lab] if 0 <= lab < len(class_names) else str(lab),
            "prediction": pred,
            "predicted_class_name": class_names[pred] if 0 <= pred < len(class_names) else str(pred),
            "correct": int(lab == pred),
            "sample_rate": int(sample_rates[i]),
            "path": paths[i] if i < len(paths) else "",
        })
    write_csv(path, rows, fields)


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, class_names: Sequence[str], out_png: Path, out_csv: Path) -> None:
    classes = len(class_names)
    mat = np.zeros((classes, classes), dtype=np.int64)
    for y, p in zip(labels, preds):
        if 0 <= int(y) < classes and 0 <= int(p) < classes:
            mat[int(y), int(p)] += 1
    rows = []
    for i in range(classes):
        row = {"true_class": class_names[i]}
        for j in range(classes):
            row[class_names[j]] = int(mat[i, j])
        rows.append(row)
    write_csv(out_csv, rows, ["true_class", *list(class_names)])
    if not require_matplotlib():
        return
    row_sum = mat.sum(axis=1, keepdims=True)
    norm = mat / np.maximum(row_sum, 1)
    fig_size = max(6, min(20, 0.35 * classes + 4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=160)
    im = ax.imshow(norm, aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized count")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if classes <= 60:
        ax.set_xticks(np.arange(classes))
        ax.set_yticks(np.arange(classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=5)
        ax.set_yticklabels(class_names, fontsize=5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def plot_training_curves(history: list[dict[str, Any]], out_path: Path) -> None:
    if not require_matplotlib() or not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r.get("train", {}).get("loss", 0.0)) for r in history]
    train_ce = [float(r.get("train", {}).get("classification_loss", 0.0)) for r in history]
    has_val = any(bool(r.get("val")) for r in history)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(epochs, train_loss, label="train total loss")
    ax.plot(epochs, train_ce, label="train CE")
    if has_val:
        val_loss = [float(r.get("val", {}).get("loss", np.nan)) for r in history]
        val_acc = [float(r.get("val", {}).get("accuracy", np.nan)) for r in history]
        val_macro = [float(r.get("val", {}).get("macro_f1", np.nan)) for r in history]
        ax.plot(epochs, val_loss, label="val loss")
        ax.plot(epochs, val_acc, label="val accuracy")
        ax.plot(epochs, val_macro, label="val macro F1")
    ax.set_xlabel("epoch")
    ax.set_title("Training curves")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_model_plots(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace, model_name: str, class_names: Sequence[str], run_dir: Path) -> dict[str, str]:
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    features = collect_embeddings(model, loader, device, args, args.plot_max_points, desc=f"{model_name} collect embeddings")
    emb = features["embedding"]
    if emb.shape[0] == 0:
        return {}
    coords, reducer = reduce_embeddings_2d(emb, args.plot_reducer, args.seed)
    np.savez_compressed(
        plot_dir / "feature_dump.npz",
        embedding=emb,
        coord=coords,
        label=features["label"],
        pred=features["pred"],
        sample_rate=features["sample_rate"],
        path=np.asarray(features["path"], dtype=object),
        reducer=np.asarray([reducer], dtype=object),
    )
    save_embedding_table(features, coords, class_names, plot_dir / "feature_projection.csv", reducer)
    plot_feature_space(coords, features["label"], f"{model_name}: {reducer} by class", plot_dir / "latent_by_class.png", "class", class_names)
    plot_feature_space(coords, features["sample_rate"], f"{model_name}: {reducer} by native sample rate", plot_dir / "latent_by_sample_rate.png", "sample_rate", class_names)
    correctness = (features["label"] == features["pred"]).astype(np.int64)
    plot_feature_space(coords, correctness, f"{model_name}: {reducer} by correctness", plot_dir / "latent_by_correctness.png", "correct", None)
    plot_confusion_matrix(features["label"], features["pred"], class_names, plot_dir / "confusion_matrix.png", plot_dir / "confusion_matrix.csv")
    return {
        "feature_dump": str(plot_dir / "feature_dump.npz"),
        "feature_projection_csv": str(plot_dir / "feature_projection.csv"),
        "latent_by_class": str(plot_dir / "latent_by_class.png"),
        "latent_by_sample_rate": str(plot_dir / "latent_by_sample_rate.png"),
        "latent_by_correctness": str(plot_dir / "latent_by_correctness.png"),
        "confusion_matrix_png": str(plot_dir / "confusion_matrix.png"),
        "confusion_matrix_csv": str(plot_dir / "confusion_matrix.csv"),
    }


def plot_robustness(rows: list[dict[str, Any]], x_field: str, metric_field: str, title: str, out_png: Path) -> None:
    if not require_matplotlib() or not rows:
        return
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[str(r["model"])].append(r)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for model, rs in by_model.items():
        rs = sorted(rs, key=lambda r: float(r[x_field]))
        x = [float(r[x_field]) for r in rs]
        y = [float(r[metric_field]) for r in rs]
        ax.plot(x, y, marker="o", label=model)
    ax.set_xscale("log")
    ax.set_xlabel(x_field.replace("_", " "))
    ax.set_ylabel(metric_field.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def plot_summary_accuracy(rows: list[dict[str, Any]], out_png: Path) -> None:
    if not require_matplotlib() or not rows:
        return
    names = [str(r["model"]) for r in rows]
    acc = [float(r["test_accuracy"]) for r in rows]
    macro = [float(r["test_macro_f1"]) for r in rows]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5), dpi=160)
    ax.bar(x - width / 2, acc, width, label="accuracy")
    ax.bar(x + width / 2, macro, width, label="macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Test metrics by model")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Checkpointing and model loop
# -----------------------------------------------------------------------------


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float], args: argparse.Namespace, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": int(epoch), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics, "args": vars(args), "model_name": model_name}, path)


def load_model_weights(path: Path, model: nn.Module, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return int(checkpoint.get("epoch", 0))


def evaluate_robustness_suite(
    model: nn.Module,
    model_name: str,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    classes: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transfer_rows: list[dict[str, Any]] = []
    for sr in parse_int_list(args.transfer_sample_rates):
        result = evaluate_model(model, test_loader, class_weights, device, args, classes, desc=f"{model_name} transfer {sr}", transform=("sample_rate", float(sr)))
        m = result["metrics"]
        transfer_rows.append({"model": model_name, "forced_sample_rate": int(sr), **m})
    cutoff_rows: list[dict[str, Any]] = []
    for cutoff in parse_float_list(args.cutoff_hz):
        result = evaluate_model(model, test_loader, class_weights, device, args, classes, desc=f"{model_name} cutoff {cutoff:g}", transform=("cutoff", float(cutoff)))
        m = result["metrics"]
        cutoff_rows.append({"model": model_name, "cutoff_hz": float(cutoff), **m})
    if transfer_rows:
        write_csv(out_dir / "sample_rate_transfer.csv", transfer_rows, ["model", "forced_sample_rate", "loss", "accuracy", "macro_f1", "weighted_f1", "micro_f1"])
    if cutoff_rows:
        write_csv(out_dir / "cutoff_robustness.csv", cutoff_rows, ["model", "cutoff_hz", "loss", "accuracy", "macro_f1", "weighted_f1", "micro_f1"])
    return transfer_rows, cutoff_rows


def train_one_model(model_name: str, manifest_info: dict[str, Any], args: argparse.Namespace, base_run_dir: Path, device: torch.device) -> dict[str, Any]:
    model_dir = base_run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    classes = int(manifest_info["class_count"])
    class_names = list(manifest_info["class_names"])
    train_loader, val_loader, test_loader = build_dataloaders(manifest_info, args, device)
    use_validation = val_loader is not None
    model = build_model(model_name, classes, args, device)
    class_weights, class_counts = class_weights_from_manifest(manifest_info["train_manifest"], classes, args.class_weight_cap, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(bool(args.amp and device.type == "cuda"))
    params_total = count_trainable_parameters(model)
    try:
        params_by_module = parameter_report(model)
    except Exception:
        params_by_module = {"total": params_total}

    safe_json_dump({
        "model": model_name,
        "validation_enabled": bool(use_validation),
        "checkpoint_selection": "val_macro_f1" if use_validation else "lowest_train_loss",
        "parameters": {"total": params_total, "by_module": params_by_module},
        "classes": classes,
        "class_names": class_names,
        "split_sizes": manifest_info["split_sizes"],
        "sample_rate_counts": manifest_info["sample_rate_counts"],
        "class_counts_train": class_counts,
        "epoch_sampling": {
            "requested_steps_per_epoch": int(args.steps_per_epoch),
            "resolved_steps_per_epoch": int(getattr(args, "_resolved_steps_per_epoch", 0)),
            "train_samples": int(getattr(args, "_train_samples_per_epoch_basis", 0)),
            "batch_size": int(args.batch_size),
            "sampler": "class-balanced replacement",
        },
        "args": vars(args),
    }, model_dir / "config.json")

    (model_dir / "model_summary.txt").write_text(
        "\n".join([
            f"model={model_name}",
            f"validation_enabled={use_validation}",
            f"checkpoint_selection={'val_macro_f1' if use_validation else 'lowest_train_loss'}",
            f"trainable_parameters={params_total}",
            f"parameter_report={json.dumps(params_by_module, indent=2, sort_keys=True)}",
            str(model),
        ]),
        encoding="utf-8",
    )

    print("", flush=True)
    print(f"=== model={model_name} ===", flush=True)
    print(f"run_dir={model_dir}", flush=True)
    print(f"validation_enabled={use_validation}", flush=True)
    print(f"checkpoint_selection={'val_macro_f1' if use_validation else 'lowest_train_loss'}", flush=True)
    print(f"model_trainable_parameters={params_total}", flush=True)
    print(f"parameter_report={params_by_module}", flush=True)
    print(f"steps_per_epoch={getattr(args, '_resolved_steps_per_epoch', 'unknown')} train_samples={getattr(args, '_train_samples_per_epoch_basis', 'unknown')}", flush=True)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, model_name, train_loader, optimizer, scaler, class_weights, device, args, classes, epoch)
        if use_validation:
            assert val_loader is not None
            val_eval = evaluate_model(model, val_loader, class_weights, device, args, classes, desc=f"{model_name} val {epoch:03d}")
            val_metrics = val_eval["metrics"]
            score = float(val_metrics.get("macro_f1", 0.0))
            checkpoint_metrics = val_metrics
            val_text = compact_metrics(val_metrics, prefix="val_")
        else:
            val_metrics = {}
            # In no-validation mode, the best checkpoint is the epoch with the
            # lowest training loss. This avoids evaluating on the test split for
            # model selection.
            score = -float(train_metrics.get("loss", 0.0))
            checkpoint_metrics = {"train_loss": float(train_metrics.get("loss", 0.0)), "selection": "lowest_train_loss"}
            val_text = "no_val=true"
        elapsed = time.time() - t0
        record = {"epoch": epoch, "epoch_seconds": elapsed, "train": train_metrics, "val": val_metrics}
        history.append(record)
        safe_json_dump(history, model_dir / "history.json")
        save_checkpoint(model_dir / "last.pt", model, optimizer, epoch, checkpoint_metrics, args, model_name)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(model_dir / "best.pt", model, optimizer, epoch, checkpoint_metrics, args, model_name)
        print(
            f"model={model_name} epoch={epoch:03d} time={elapsed:.1f}s "
            f"train_loss={train_metrics.get('loss', 0.0):.4f} "
            f"train_ce={train_metrics.get('classification_loss', 0.0):.4f} "
            f"train_sampling_consistency={train_metrics.get('sampling_consistency', 0.0):.4f} "
            f"{val_text} best_epoch={best_epoch:03d}",
            flush=True,
        )
    plot_training_curves(history, model_dir / "plots" / "training_curves.png")
    loaded_epoch = load_model_weights(model_dir / "best.pt", model, device)
    test_eval = evaluate_model(model, test_loader, class_weights, device, args, classes, desc=f"{model_name} test")
    test_metrics = test_eval["metrics"]
    per_class = per_class_metrics(test_eval["logits"], test_eval["target"], class_names)
    write_csv(model_dir / "test_per_class_metrics.csv", per_class, ["class_index", "class_name", "precision", "recall", "f1", "support"])
    if args.plot_split == "val" and use_validation:
        plot_loader = val_loader
    else:
        plot_loader = test_loader
        if args.plot_split == "val" and not use_validation:
            print(f"model={model_name} plot_split=val requested but validation is disabled; using test split for plots", flush=True)
    plot_paths = save_model_plots(model, plot_loader, device, args, model_name, class_names, model_dir)
    if (model_dir / "plots" / "training_curves.png").exists():
        plot_paths["training_curves"] = str(model_dir / "plots" / "training_curves.png")
    transfer_rows, cutoff_rows = evaluate_robustness_suite(model, model_name, test_loader, class_weights, device, args, classes, model_dir)
    if transfer_rows:
        plot_robustness(transfer_rows, "forced_sample_rate", "macro_f1", f"{model_name}: sample-rate transfer", model_dir / "plots" / "sample_rate_transfer.png")
        plot_paths["sample_rate_transfer_png"] = str(model_dir / "plots" / "sample_rate_transfer.png")
    if cutoff_rows:
        plot_robustness(cutoff_rows, "cutoff_hz", "macro_f1", f"{model_name}: cutoff robustness", model_dir / "plots" / "cutoff_robustness.png")
        plot_paths["cutoff_robustness_png"] = str(model_dir / "plots" / "cutoff_robustness.png")

    result = {
        "model": model_name,
        "validation_enabled": bool(use_validation),
        "checkpoint_selection": "val_macro_f1" if use_validation else "lowest_train_loss",
        "parameters": int(params_total),
        "best_epoch": int(best_epoch),
        "loaded_epoch": int(loaded_epoch),
        "checkpoint": str(model_dir / "best.pt"),
        "test": test_metrics,
        "plots": plot_paths,
        "transfer_rows": transfer_rows,
        "cutoff_rows": cutoff_rows,
    }
    safe_json_dump(result, model_dir / "test_metrics.json")
    print(f"model={model_name} test_checkpoint={model_dir / 'best.pt'} epoch={loaded_epoch:03d} {compact_metrics(test_metrics, prefix='test_')}", flush=True)
    return result


def resolve_model_names(args: argparse.Namespace) -> list[str]:
    values = parse_name_list(args.models)
    allowed = {"sfi_fno", "sfi_scatnet", "mel_cnn", "mel_efficientnet", "mel_cnn_fno", "whalenet_mel", "whalenet_wst1", "whalenet_wst2", "whalenet_wst12", "whalenet_full"}
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"Unknown models {bad}. Allowed: {sorted(allowed)}")
    ordered: list[str] = []
    for preferred in ("sfi_fno", "sfi_scatnet"):
        if preferred in values:
            ordered.append(preferred)
    ordered.extend([v for v in values if v not in set(ordered)])
    return ordered


def run_training(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_info = build_manifests(args, run_dir)
    safe_json_dump(vars(args), run_dir / "args.json")
    print(f"root={Path(args.root or args.watkins_root).expanduser().resolve()}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"classes={manifest_info['class_count']}", flush=True)
    print(f"class_names={manifest_info['class_names']}", flush=True)
    print(f"split_sizes={manifest_info['split_sizes']}", flush=True)
    print(f"validation_enabled={manifest_info.get('use_validation', True)}", flush=True)
    print(f"sample_rate_counts={manifest_info['sample_rate_counts']}", flush=True)
    model_names = resolve_model_names(args)
    print(f"models={model_names}", flush=True)

    if args.dry_run:
        train_loader, _val_loader, _test_loader = build_dataloaders(manifest_info, args, device)
        batch = next(iter(train_loader))
        model = build_model(model_names[0], int(manifest_info["class_count"]), args, device)
        audio = batch_audio_to_device(batch, device)
        sr = [int(x) for x in batch["sample_rate"]]
        with torch.no_grad():
            out = model(audio, sr, return_hidden=True)
        print(f"dry_run_model={model_names[0]}", flush=True)
        print(f"dry_run_parameters={count_trainable_parameters(model)}", flush=True)
        print(f"dry_run_logits_shape={tuple(logits_from_output(out).shape)}", flush=True)
        print(f"dry_run_embedding_shape={tuple(out['embedding'].shape)}", flush=True)
        return 0

    all_results: list[dict[str, Any]] = []
    all_transfer_rows: list[dict[str, Any]] = []
    all_cutoff_rows: list[dict[str, Any]] = []
    for i, name in enumerate(model_names):
        result = train_one_model(name, manifest_info, args, run_dir, device)
        all_results.append(result)
        all_transfer_rows.extend(result.get("transfer_rows", []))
        all_cutoff_rows.extend(result.get("cutoff_rows", []))
        result_rows = [
            {
                "model": r["model"],
                "parameters": r["parameters"],
                "best_epoch": r["best_epoch"],
                "test_loss": r["test"].get("loss", 0.0),
                "test_accuracy": r["test"].get("accuracy", 0.0),
                "test_macro_f1": r["test"].get("macro_f1", 0.0),
                "test_weighted_f1": r["test"].get("weighted_f1", 0.0),
                "test_micro_f1": r["test"].get("micro_f1", 0.0),
                "checkpoint": r["checkpoint"],
            }
            for r in all_results
        ]
        write_csv(run_dir / "experiment_results.csv", result_rows, ["model", "parameters", "best_epoch", "test_loss", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_micro_f1", "checkpoint"])
        safe_json_dump(all_results, run_dir / "experiment_results.json")
        plot_summary_accuracy(result_rows, run_dir / "plots" / "model_test_summary.png")
        if all_transfer_rows:
            write_csv(run_dir / "sample_rate_transfer_all_models.csv", all_transfer_rows, ["model", "forced_sample_rate", "loss", "accuracy", "macro_f1", "weighted_f1", "micro_f1"])
            plot_robustness(all_transfer_rows, "forced_sample_rate", "macro_f1", "All models: sample-rate transfer", run_dir / "plots" / "sample_rate_transfer_all_models.png")
        if all_cutoff_rows:
            write_csv(run_dir / "cutoff_robustness_all_models.csv", all_cutoff_rows, ["model", "cutoff_hz", "loss", "accuracy", "macro_f1", "weighted_f1", "micro_f1"])
            plot_robustness(all_cutoff_rows, "cutoff_hz", "macro_f1", "All models: cutoff robustness", run_dir / "plots" / "cutoff_robustness_all_models.png")
        if i == 0 and args.stop_after_first_model:
            print("stopping after first model as requested", flush=True)
            break

    print("", flush=True)
    print("=== experiment summary ===", flush=True)
    for r in all_results:
        print(
            f"{r['model']}: params={r['parameters']} "
            f"acc={r['test'].get('accuracy', 0.0):.4f} "
            f"macro_f1={r['test'].get('macro_f1', 0.0):.4f} "
            f"weighted_f1={r['test'].get('weighted_f1', 0.0):.4f} "
            f"checkpoint={r['checkpoint']}",
            flush=True,
        )
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sampling-frequency-independent scattering classifier and fixed-mel baselines for Watkins-style bioacoustics.")
    p.add_argument("--root", default="/home/mmlab/Desktop/nfo_bioacoustics/watkins_marine", help="Watkins-style folder: <root>/<species>/<train_or_test_file>.wav")
    p.add_argument("--watkins-root", default="", help="Alias for --root. Kept for explicit Watkins runs.")
    p.add_argument("--run-dir", default="./runs/watkins_sfi_scatnet")
    p.add_argument("--models", default="sfi_fno,sfi_scatnet,mel_cnn,mel_cnn_fno,whalenet_mel,whalenet_wst12,whalenet_full", help="Comma-separated. Includes native SFI models, fixed-Mel baselines, and WhaleNet-style WST/Mel baselines.")
    p.add_argument("--stop-after-first-model", action="store_true", help="Train/evaluate only the new model, then stop.")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--audio-extensions", default=AUDIO_EXTENSIONS_DEFAULT)
    p.add_argument("--unknown-split", default="derive", choices=["error", "train", "derive"])
    p.add_argument("--val-fraction", type=float, default=0.0, help="Validation fraction carved from train. Use 0.0 to disable validation entirely and train/select checkpoints without a validation loader.")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--min-samples-per-class", type=int, default=0)
    p.add_argument("--remove-duplicates", action="store_true")
    p.add_argument("--paper-protocol", action="store_true", help="Convenience flag: use derived stratified split behavior even when filenames lack split tokens. Combine with --min-samples-per-class 50 if needed.")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--pad-mode", default="zero", choices=["zero", "repeat"])
    p.add_argument("--normalize", default="standardize", choices=["none", "peak", "rms", "standardize"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler-power", type=float, default=0.5)
    p.add_argument("--class-weight-cap", type=float, default=10.0)
    p.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True)

    # New model.
    p.add_argument("--sfi-freq-min-hz", type=float, default=5.0)
    p.add_argument("--sfi-freq-max-hz", type=float, default=120000.0)
    p.add_argument("--sfi-freq-bins", type=int, default=96)
    p.add_argument("--sfi-time-bins", type=int, default=96)
    p.add_argument("--sfi-q-factor", type=float, default=12.0)
    p.add_argument("--sfi-window-ms", default="16,64,256", help="Physical analysis windows for the new model, in milliseconds.")
    p.add_argument("--sfi-modulation-rates-hz", default="2,4,8,16")
    p.add_argument("--sfi-token-dim", type=int, default=96)
    p.add_argument("--sfi-transformer-layers", type=int, default=2)
    p.add_argument("--sfi-num-heads", type=int, default=4)
    p.add_argument("--sampling-consistency-weight", type=float, default=0.0, help="Only used for sfi_scatnet. Upsampling-only consistency, so no information is removed. Keep 0 for the clean front-end-only experiment.")
    p.add_argument("--consistency-min-factor", type=float, default=1.05)
    p.add_argument("--consistency-max-factor", type=float, default=1.50)
    p.add_argument("--consistency-round-to", type=int, default=1000)
    p.add_argument("--consistency-max-hz", type=int, default=0)

    # Baselines.
    p.add_argument("--baseline-sr", type=int, default=47600, help="Fixed-rate mel baseline sample rate. 47600 matches the median-rate idea used by WhaleNet-style preprocessing.")
    p.add_argument("--mel-bins", type=int, default=64)
    p.add_argument("--mel-time-bins", type=int, default=96)
    p.add_argument("--mel-n-fft", type=int, default=1024)
    p.add_argument("--mel-hop-length", type=int, default=200)
    p.add_argument("--mel-f-min-hz", type=float, default=5.0)
    p.add_argument("--mel-f-max-hz", type=float, default=24000.0)
    p.add_argument("--baseline-channels", type=int, default=16)
    p.add_argument("--efficientnet-width", type=int, default=16)
    p.add_argument("--fno-depth", type=int, default=2)
    p.add_argument("--fno-modes-freq", type=int, default=8)
    p.add_argument("--fno-modes-time", type=int, default=8)

    # WhaleNet-style fixed-rate WST/Mel baseline.
    p.add_argument("--whalenet-sr", type=int, default=0, help="Fixed resampling rate for WhaleNet-style baselines. 0 means use --baseline-sr.")
    p.add_argument("--whalenet-signal-len", type=int, default=8000, help="Paper-style fixed signal length after resampling.")
    p.add_argument("--whalenet-j", type=int, default=6, help="Kymatio WST scale J for WhaleNet-style baselines.")
    p.add_argument("--whalenet-q", type=int, default=16, help="Kymatio WST Q for WhaleNet-style baselines.")
    p.add_argument("--whalenet-mel-bins", type=int, default=64)
    p.add_argument("--whalenet-mel-n-fft", type=int, default=400)
    p.add_argument("--whalenet-mel-hop-length", type=int, default=200)

    # Shared model dimensions and optimization.
    p.add_argument("--embedding-dim", type=int, default=192)
    p.add_argument("--head-hidden-dim", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # End-of-run diagnostics.
    p.add_argument("--plot-split", default="test", choices=["val", "test"])
    p.add_argument("--plot-max-points", type=int, default=0)
    p.add_argument("--plot-reducer", default="tsne", choices=["pca", "tsne"])
    p.add_argument("--transfer-sample-rates", default="2000,4000,8000,16000,32000,47600,96000")
    p.add_argument("--cutoff-hz", default="50,100,250,1000,4000,12000,24000,48000,96000")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.watkins_root:
        args.root = args.watkins_root
    if args.paper_protocol and args.unknown_split == "error":
        args.unknown_split = "derive"
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
