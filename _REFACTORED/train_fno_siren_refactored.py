from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from bioacoustic_visualization import make_all_plots


from data.bioacoustic_dataset import Bioacoustic_Dataset, collate_audio_samples

try:
    from models.model_siren_fno import (
        DATASET_NAMES,
        SRInvariantBioacousticClassifier,
    )
except ModuleNotFoundError:
    from sr_invariant_fno_siren_bioacoustic import (  # type: ignore
        DATASET_NAMES,
        SRInvariantBioacousticClassifier,
    )


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

CLASS_FIELDS = ["dataset", "class_index", "class_name", "source_name"]

TRAIN_SPLITS = {"train", "training"}
VAL_SPLITS = {"val", "valid", "validation", "dev", "development"}
TEST_SPLITS = {"test", "testing", "test_5s", "test5s", "eval", "evaluation"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_int_list(text: Any) -> list[int]:
    if text is None:
        return []
    values: list[int] = []
    for part in str(text).replace(",", ";").split(";"):
        part = part.strip()
        if part:
            values.append(int(part))
    return sorted(set(values))


def parse_float_sequence(text: Any) -> tuple[float, ...]:
    if isinstance(text, (list, tuple)):
        values = [float(x) for x in text]
    else:
        values = [float(part.strip()) for part in str(text).replace(",", ";").split(";") if part.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return tuple(values)


def stable_float_0_1(text: str) -> float:
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)


def split_name(split: Any) -> str:
    return str(split or "").strip().lower()


def is_train_split(row: dict[str, str]) -> bool:
    return split_name(row.get("split", "")) in TRAIN_SPLITS


def is_val_split(row: dict[str, str]) -> bool:
    return split_name(row.get("split", "")) in VAL_SPLITS


def is_test_split(row: dict[str, str]) -> bool:
    return split_name(row.get("split", "")) in TEST_SPLITS


def split_key(row: dict[str, Any]) -> str:
    for key in ("source_id", "sample_id", "global_id", "audio_path", "array_path"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return json.dumps(row, sort_keys=True)


def load_class_names(classes_csv: Path) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for row in read_csv(classes_csv):
        dataset = row.get("dataset", "")
        if not dataset:
            continue
        out[dataset][int(row["class_index"])] = row.get("class_name", str(row["class_index"]))
    return dict(out)


def compact_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    parts = [f"{prefix}loss={metrics.get('loss', 0.0):.4f}"]
    parts.append(f"{prefix}mean_macro_f1={metrics.get('mean_macro_f1', 0.0):.4f}")
    for name in DATASET_NAMES:
        parts.append(f"{prefix}{name}_macro_f1={metrics.get(name + '_macro_f1', 0.0):.4f}")
    return " ".join(parts)


def select_classes(
    rows_by_dataset: dict[str, list[dict[str, str]]],
    class_names: dict[str, dict[int, str]],
    classes_per_dataset: int,
) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    selected: dict[str, list[int]] = {}
    selected_names: dict[str, list[str]] = {}

    for dataset in DATASET_NAMES:
        dataset_rows = rows_by_dataset.get(dataset, [])
        train_rows = [row for row in dataset_rows if is_train_split(row)]
        rows_for_counting = train_rows if train_rows else dataset_rows

        counts = Counter()
        for row in rows_for_counting:
            for idx in parse_int_list(row.get("label_indices", "")):
                counts[idx] += 1

        if not counts:
            raise RuntimeError(f"No labels found for dataset={dataset}")

        chosen = [idx for idx, _ in counts.most_common(classes_per_dataset)]
        names = [class_names.get(dataset, {}).get(idx, str(idx)) for idx in chosen]
        while len(names) < classes_per_dataset:
            names.append(f"unused_{len(names)}")

        selected[dataset] = chosen
        selected_names[dataset] = names

    return selected, selected_names


def remap_row_to_selected_classes(
    row: dict[str, str],
    selected_for_dataset: list[int],
    selected_names_for_dataset: list[str],
) -> dict[str, Any] | None:
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_for_dataset)}
    old_labels = parse_int_list(row.get("label_indices", ""))
    new_labels = sorted(old_to_new[idx] for idx in old_labels if idx in old_to_new)

    if not new_labels:
        return None

    out_row: dict[str, Any] = dict(row)
    out_row["label_indices"] = ";".join(str(idx) for idx in new_labels)
    out_row["label_names"] = ";".join(selected_names_for_dataset[idx] for idx in new_labels)
    return out_row


def assign_derived_split(
    row: dict[str, Any],
    dataset: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    derive_val: bool,
    derive_test: bool,
) -> str:
    r = stable_float_0_1(f"{dataset}:{split_key(row)}:{seed}")

    if derive_test and r < test_fraction:
        return "test"

    val_start = test_fraction if derive_test else 0.0
    if derive_val and val_start <= r < val_start + val_fraction:
        return "val"

    return "train"


def assign_unknown_split(
    row: dict[str, Any],
    dataset: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> str:
    r = stable_float_0_1(f"{dataset}:{split_key(row)}:{seed}")
    if r < test_fraction:
        return "test"
    if r < test_fraction + val_fraction:
        return "val"
    return "train"


def build_reduced_manifests(
    root: Path,
    run_dir: Path,
    classes_per_dataset: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError("test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    source_manifest = root / "manifests" / "all_samples.csv"
    source_classes = root / "manifests" / "all_classes.csv"
    if not source_manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {source_manifest}")
    if not source_classes.exists():
        raise FileNotFoundError(f"Missing classes CSV: {source_classes}")

    rows = read_csv(source_manifest)
    class_names = load_class_names(source_classes)

    rows_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset = row.get("dataset", "")
        if dataset in DATASET_NAMES:
            rows_by_dataset[dataset].append(row)

    selected, selected_names = select_classes(rows_by_dataset, class_names, classes_per_dataset)

    out_dir = run_dir / "manifests_reduced"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_rows: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        for new_idx, name in enumerate(selected_names[dataset]):
            class_rows.append(
                {
                    "dataset": dataset,
                    "class_index": new_idx,
                    "class_name": name,
                    "source_name": name,
                }
            )

    train_rows_out: list[dict[str, Any]] = []
    val_rows_out: list[dict[str, Any]] = []
    test_rows_out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    for dataset in DATASET_NAMES:
        prepared_rows: list[dict[str, Any]] = []
        dropped = 0

        for row in rows_by_dataset.get(dataset, []):
            out_row = remap_row_to_selected_classes(row, selected[dataset], selected_names[dataset])
            if out_row is None:
                dropped += 1
                continue
            prepared_rows.append(out_row)

        has_official_val = any(is_val_split(row) for row in prepared_rows)
        has_official_test = any(is_test_split(row) for row in prepared_rows)
        derive_val = not has_official_val
        derive_test = not has_official_test

        counts = Counter()
        for i, out_row in enumerate(prepared_rows):
            raw_split = split_name(out_row.get("split", ""))

            if is_test_split(out_row):
                assigned = "test"
            elif is_val_split(out_row):
                assigned = "val"
            elif is_train_split(out_row):
                assigned = assign_derived_split(
                    out_row,
                    dataset=dataset,
                    seed=seed,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                    derive_val=derive_val,
                    derive_test=derive_test,
                )
            else:
                assigned = assign_unknown_split(
                    out_row,
                    dataset=dataset,
                    seed=seed,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                )

            out_row = dict(out_row)
            out_row["split"] = assigned
            original_id = out_row.get("global_id") or out_row.get("sample_id") or str(i)
            out_row["global_id"] = f"{dataset}:{assigned}:{original_id}"

            if assigned == "test":
                test_rows_out.append(out_row)
            elif assigned == "val":
                val_rows_out.append(out_row)
            else:
                train_rows_out.append(out_row)

            counts[assigned] += 1
            counts[f"source_{raw_split or 'empty'}"] += 1

        stats[dataset] = {
            "selected_original_class_indices": selected[dataset],
            "selected_class_names": selected_names[dataset],
            "official_validation_present": has_official_val,
            "official_test_present": has_official_test,
            "train_rows": int(counts["train"]),
            "val_rows": int(counts["val"]),
            "test_rows": int(counts["test"]),
            "dropped_rows_without_selected_labels": dropped,
            "source_split_counts_after_label_filter": {
                key.removeprefix("source_"): int(value)
                for key, value in counts.items()
                if key.startswith("source_")
            },
        }

    if not train_rows_out:
        raise RuntimeError("Reduced train manifest has zero rows")
    if not val_rows_out:
        raise RuntimeError("Reduced validation manifest has zero rows")
    if not test_rows_out:
        raise RuntimeError("Reduced test manifest has zero rows")

    train_manifest = out_dir / "train_samples.csv"
    val_manifest = out_dir / "val_samples.csv"
    test_manifest = out_dir / "test_samples.csv"
    classes_csv = out_dir / "classes.csv"

    write_csv(train_manifest, train_rows_out, MANIFEST_FIELDS)
    write_csv(val_manifest, val_rows_out, MANIFEST_FIELDS)
    write_csv(test_manifest, test_rows_out, MANIFEST_FIELDS)
    write_csv(classes_csv, class_rows, CLASS_FIELDS)
    (out_dir / "selected_classes.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    split_sizes = {
        "train": len(train_rows_out),
        "val": len(val_rows_out),
        "test": len(test_rows_out),
    }

    return {
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "test_manifest": test_manifest,
        "classes_csv": classes_csv,
        "selected": selected,
        "selected_names": selected_names,
        "stats": stats,
        "split_sizes": split_sizes,
    }


class DatasetBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Bioacoustic_Dataset,
        batch_size: int,
        steps_per_epoch: int,
        sampler_power: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.sampler_power = float(sampler_power)
        self.seed = int(seed)
        self.epoch = 0
        self.datasets = list(DATASET_NAMES)

        if self.batch_size < len(self.datasets):
            raise ValueError(f"batch_size must be >= {len(self.datasets)}")
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")

        self.indices_by_dataset: dict[str, list[int]] = {name: [] for name in self.datasets}
        for i, row in enumerate(dataset.rows):
            name = row.get("dataset", "")
            if name in self.indices_by_dataset:
                self.indices_by_dataset[name].append(i)

        missing = [name for name, indices in self.indices_by_dataset.items() if not indices]
        if missing:
            raise RuntimeError(f"Training manifest has no rows for dataset(s): {missing}")

        self.weights_by_dataset: dict[str, torch.Tensor] = {}
        for name, indices in self.indices_by_dataset.items():
            class_counts = Counter()
            labels_by_index: dict[int, list[int]] = {}

            for idx in indices:
                labels = parse_int_list(dataset.rows[idx].get("label_indices", ""))
                labels_by_index[idx] = labels
                for label in labels:
                    class_counts[int(label)] += 1

            weights: list[float] = []
            for idx in indices:
                labels = labels_by_index[idx]
                if labels:
                    label_weights = [
                        (1.0 / max(class_counts[int(label)], 1)) ** self.sampler_power
                        for label in labels
                    ]
                    weights.append(float(sum(label_weights) / len(label_weights)))
                else:
                    weights.append(1.0)

            weight_tensor = torch.tensor(weights, dtype=torch.float64)
            self.weights_by_dataset[name] = weight_tensor / weight_tensor.sum().clamp_min(1e-12)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1_000_003 * self.epoch)

        base = self.batch_size // len(self.datasets)
        remainder = self.batch_size % len(self.datasets)
        per_dataset = {
            name: base + (1 if i < remainder else 0)
            for i, name in enumerate(self.datasets)
        }

        for _ in range(self.steps_per_epoch):
            batch: list[int] = []
            for name in self.datasets:
                indices = self.indices_by_dataset[name]
                weights = self.weights_by_dataset[name]
                count = per_dataset[name]
                sampled = torch.multinomial(
                    weights,
                    num_samples=count,
                    replacement=True,
                    generator=generator,
                ).tolist()
                batch.extend(indices[j] for j in sampled)

            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[i] for i in order]


def audio_batch_to_list(audio: Any) -> list[torch.Tensor]:
    if isinstance(audio, list):
        return [x.float() for x in audio]
    if audio.dim() == 3:
        return [audio[i].float() for i in range(audio.shape[0])]
    if audio.dim() == 2:
        return [audio[i : i + 1].float() for i in range(audio.shape[0])]
    raise ValueError(f"Unsupported audio batch shape: {tuple(audio.shape)}")


def sr_batch_to_list(sample_rate: Any, batch_size: int) -> list[int]:
    if isinstance(sample_rate, int):
        return [int(sample_rate)] * batch_size
    if isinstance(sample_rate, float):
        return [int(round(sample_rate))] * batch_size
    if isinstance(sample_rate, torch.Tensor):
        if sample_rate.ndim == 0:
            return [int(sample_rate.item())] * batch_size
        return [int(x) for x in sample_rate.detach().cpu().tolist()]
    return [int(x) for x in sample_rate]


def choose_augmented_sr(
    sr: int,
    min_factor: float,
    max_factor: float,
    round_to: int,
    min_hz: int,
    max_hz: int,
) -> tuple[int, float]:
    factor = random.uniform(float(min_factor), float(max_factor))
    target = max(1, int(round(float(sr) * factor)))
    if min_hz > 0:
        target = max(target, int(min_hz))
    if max_hz > 0:
        target = min(target, int(max_hz))
    if round_to and target >= 4 * int(round_to):
        target = max(1, int(round(target / int(round_to)) * int(round_to)))
    return target, float(target) / float(sr)


def resample_tensor(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if int(orig_sr) == int(target_sr):
        return audio
    length = int(audio.shape[-1])
    new_length = max(1, int(round(length * float(target_sr) / float(orig_sr))))
    return F.interpolate(
        audio.unsqueeze(0),
        size=new_length,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def make_sr_view(
    batch: dict[str, Any],
    device: torch.device,
    min_factor: float,
    max_factor: float,
    round_to: int,
    min_hz: int,
    max_hz: int,
    augment: bool,
) -> tuple[list[torch.Tensor], list[int]]:
    audios = audio_batch_to_list(batch["audio"])
    sample_rates = sr_batch_to_list(batch["sample_rate"], len(audios))

    out_audio: list[torch.Tensor] = []
    out_sr: list[int] = []

    for audio, sr in zip(audios, sample_rates):
        if augment:
            target_sr, _ = choose_augmented_sr(
                sr,
                min_factor=min_factor,
                max_factor=max_factor,
                round_to=round_to,
                min_hz=min_hz,
                max_hz=max_hz,
            )
        else:
            target_sr = int(sr)

        out_audio.append(resample_tensor(audio, sr, target_sr).to(device, non_blocking=True))
        out_sr.append(int(target_sr))

    return out_audio, out_sr


def make_targets(
    batch: dict[str, Any],
    dataset_name: str,
    classes_per_dataset: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    indices = [i for i, name in enumerate(batch["dataset"]) if name == dataset_name]
    target = torch.zeros(len(indices), classes_per_dataset, dtype=torch.float32, device=device)

    for j, batch_idx in enumerate(indices):
        row = batch["metadata"][batch_idx]
        for label in parse_int_list(row.get("label_indices", "")):
            if 0 <= label < classes_per_dataset:
                target[j, label] = 1.0

    return target, indices


def build_pos_weights(
    train_manifest: Path,
    classes_per_dataset: int,
    cap: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    rows = read_csv(train_manifest)
    counts = {name: torch.zeros(classes_per_dataset, dtype=torch.float64) for name in DATASET_NAMES}
    totals = Counter()

    for row in rows:
        name = row.get("dataset", "")
        if name not in counts:
            continue
        totals[name] += 1
        for idx in parse_int_list(row.get("label_indices", "")):
            if 0 <= idx < classes_per_dataset:
                counts[name][idx] += 1

    pos_weights: dict[str, torch.Tensor] = {}
    valid_counts: dict[str, int] = {}

    for name in DATASET_NAMES:
        class_counts = counts[name]
        valid = int((class_counts > 0).sum().item())
        valid_counts[name] = valid

        n = max(float(totals[name]), 1.0)
        weights = ((n - class_counts) / class_counts.clamp_min(1.0)).float().clamp(max=float(cap))
        if valid < classes_per_dataset:
            weights[valid:] = 0.0
        pos_weights[name] = weights.to(device)

    return pos_weights, valid_counts


def classification_loss(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    pos_weights: dict[str, torch.Tensor],
    valid_counts: dict[str, int],
    classes_per_dataset: int,
    device: torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []

    for name in DATASET_NAMES:
        target, indices = make_targets(batch, name, classes_per_dataset, device)
        valid = valid_counts[name]
        if not indices or valid <= 0:
            continue

        idx = torch.tensor(indices, dtype=torch.long, device=device)
        logits = outputs["logits"][name][idx, :valid]
        target = target[:, :valid]
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weights[name][:valid],
            )
        )

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def prediction_consistency_loss(
    out1: dict[str, Any],
    out2: dict[str, Any],
    batch: dict[str, Any],
    valid_counts: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []

    for name in DATASET_NAMES:
        indices = [i for i, dataset_name in enumerate(batch["dataset"]) if dataset_name == name]
        valid = valid_counts[name]
        if not indices or valid <= 0:
            continue

        idx = torch.tensor(indices, dtype=torch.long, device=device)
        p1 = torch.sigmoid(out1["logits"][name][idx, :valid])
        p2 = torch.sigmoid(out2["logits"][name][idx, :valid])
        losses.append(F.mse_loss(p1, p2))

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def embedding_consistency_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(z1, z2, dim=-1)).mean()


def sr_bucket_edges(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    edges = parse_float_sequence(args.sr_bucket_edges)
    return torch.tensor(edges, dtype=torch.float32, device=device)


def sr_bucket_labels(sample_rates: Sequence[int], args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    edges = sr_bucket_edges(args, device)
    sr = torch.tensor([float(x) for x in sample_rates], dtype=torch.float32, device=device)
    return torch.bucketize(sr, edges)


def sr_adversary_loss(outputs: dict[str, Any], sample_rates: Sequence[int], args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    sr_logits = outputs.get("sr_logits")
    if sr_logits is None:
        return torch.zeros((), device=device)
    labels = sr_bucket_labels(sample_rates, args, device)
    if sr_logits.shape[-1] <= int(labels.max().item()):
        raise ValueError(
            f"sr_logits has {sr_logits.shape[-1]} buckets, but labels require at least "
            f"{int(labels.max().item()) + 1}. Check --sr-bucket-edges and model sr_adversary_buckets."
        )
    return F.cross_entropy(sr_logits, labels)


def grl_lambda_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    target = float(args.grl_lambda)
    warmup = int(args.grl_warmup_epochs)
    if warmup <= 0:
        return target
    return target * min(1.0, max(0.0, float(epoch) / float(warmup)))


def amp_autocast(use_amp: bool):
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)



def progress_iter(iterable: Any, args: argparse.Namespace, desc: str, total: int | None = None, leave: bool = False):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        disable=bool(getattr(args, "no_progress", False)),
    )


def set_progress_postfix(iterator: Any, values: dict[str, float]) -> None:
    if hasattr(iterator, "set_postfix"):
        iterator.set_postfix({key: f"{value:.4f}" for key, value in values.items()})


def train_one_epoch(
    model: SRInvariantBioacousticClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    pos_weights: dict[str, torch.Tensor],
    valid_counts: dict[str, int],
    epoch: int,
) -> dict[str, float]:
    model.train()

    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)

    totals = Counter()
    steps = 0
    use_amp = bool(args.amp and device.type == "cuda")
    grl_lambda = grl_lambda_for_epoch(args, epoch)

    iterator = progress_iter(loader, args, desc=f"train {epoch:03d}", total=len(loader), leave=False)
    for batch in iterator:
        steps += 1

        view1, sr1 = make_sr_view(
            batch,
            device=device,
            min_factor=args.sr_aug_min_factor,
            max_factor=args.sr_aug_max_factor,
            round_to=args.sr_aug_round_to,
            min_hz=args.sr_aug_min_hz,
            max_hz=args.sr_aug_max_hz,
            augment=args.aug,
        )
        view2, sr2 = make_sr_view(
            batch,
            device=device,
            min_factor=args.sr_aug_min_factor,
            max_factor=args.sr_aug_max_factor,
            round_to=args.sr_aug_round_to,
            min_hz=args.sr_aug_min_hz,
            max_hz=args.sr_aug_max_hz,
            augment=args.aug,
        )

        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(use_amp):
            out1 = model(view1, sr1, grl_lambda=grl_lambda, return_sr_logits=True)
            out2 = model(view2, sr2, grl_lambda=grl_lambda, return_sr_logits=True)

            cls1 = classification_loss(out1, batch, pos_weights, valid_counts, args.classes_per_dataset, device)
            cls2 = classification_loss(out2, batch, pos_weights, valid_counts, args.classes_per_dataset, device)
            cls_loss = 0.5 * (cls1 + cls2)

            emb_loss = embedding_consistency_loss(out1["embedding"], out2["embedding"])
            pred_loss = prediction_consistency_loss(out1, out2, batch, valid_counts, device)
            adv_loss = 0.5 * (
                sr_adversary_loss(out1, sr1, args, device) + sr_adversary_loss(out2, sr2, args, device)
            )

            loss = (
                cls_loss
                + args.embedding_consistency_weight * emb_loss
                + args.prediction_consistency_weight * pred_loss
                + args.sr_adversary_weight * adv_loss
            )

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
        totals["classification"] += float(cls_loss.detach().cpu())
        totals["embedding_consistency"] += float(emb_loss.detach().cpu())
        totals["prediction_consistency"] += float(pred_loss.detach().cpu())
        totals["sr_adversary"] += float(adv_loss.detach().cpu())
        totals["grl_lambda"] += float(grl_lambda)

        set_progress_postfix(
            iterator,
            {
                "loss": totals["loss"] / max(steps, 1),
                "cls": totals["classification"] / max(steps, 1),
                "grl": totals["grl_lambda"] / max(steps, 1),
            },
        )

    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: SRInvariantBioacousticClassifier,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    pos_weights: dict[str, torch.Tensor],
    valid_counts: dict[str, int],
    desc: str = "eval",
) -> dict[str, float]:
    model.eval()

    totals = Counter()
    steps = 0
    tp = {name: torch.zeros(args.classes_per_dataset, dtype=torch.float64) for name in DATASET_NAMES}
    fp = {name: torch.zeros(args.classes_per_dataset, dtype=torch.float64) for name in DATASET_NAMES}
    fn = {name: torch.zeros(args.classes_per_dataset, dtype=torch.float64) for name in DATASET_NAMES}

    iterator = progress_iter(loader, args, desc=desc, total=len(loader), leave=False)
    for batch in iterator:
        steps += 1
        view, sr = make_sr_view(batch, device, 1.0, 1.0, 0, 0, 0, augment=False)
        out = model(view, sr, grl_lambda=0.0, return_sr_logits=False)

        cls_loss = classification_loss(out, batch, pos_weights, valid_counts, args.classes_per_dataset, device)
        totals["loss"] += float(cls_loss.detach().cpu())
        set_progress_postfix(iterator, {"loss": totals["loss"] / max(steps, 1)})

        for name in DATASET_NAMES:
            target, indices = make_targets(batch, name, args.classes_per_dataset, device)
            valid = valid_counts[name]
            if not indices or valid <= 0:
                continue

            idx = torch.tensor(indices, dtype=torch.long, device=device)
            probs = torch.sigmoid(out["logits"][name][idx, :valid])
            pred = (probs >= args.eval_threshold).float().cpu()
            target_cpu = target[:, :valid].cpu()

            tp[name][:valid] += (pred * target_cpu).sum(dim=0).double()
            fp[name][:valid] += (pred * (1.0 - target_cpu)).sum(dim=0).double()
            fn[name][:valid] += ((1.0 - pred) * target_cpu).sum(dim=0).double()

    metrics: dict[str, float] = {"loss": totals["loss"] / max(steps, 1)}
    macro_values: list[float] = []

    for name in DATASET_NAMES:
        valid = valid_counts[name]
        if valid <= 0:
            continue

        precision = tp[name][:valid] / (tp[name][:valid] + fp[name][:valid]).clamp_min(1.0)
        recall = tp[name][:valid] / (tp[name][:valid] + fn[name][:valid]).clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

        support = tp[name][:valid] + fn[name][:valid]
        supported = support > 0
        macro_f1 = float(f1[supported].mean().item()) if bool(supported.any()) else 0.0

        micro_tp = tp[name][:valid].sum()
        micro_fp = fp[name][:valid].sum()
        micro_fn = fn[name][:valid].sum()
        micro_p = micro_tp / (micro_tp + micro_fp).clamp_min(1.0)
        micro_r = micro_tp / (micro_tp + micro_fn).clamp_min(1.0)
        micro_f1 = float((2 * micro_p * micro_r / (micro_p + micro_r).clamp_min(1e-12)).item())

        metrics[f"{name}_macro_f1"] = macro_f1
        metrics[f"{name}_micro_f1"] = micro_f1
        macro_values.append(macro_f1)

    metrics["mean_macro_f1"] = float(sum(macro_values) / max(len(macro_values), 1))
    return metrics


@torch.no_grad()
def evaluate_sample_rate_invariance(
    model: SRInvariantBioacousticClassifier,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    valid_counts: dict[str, int],
    desc: str = "invariance",
) -> dict[str, float]:
    model.eval()
    if args.invariance_eval_batches <= 0:
        return {}

    totals = Counter()
    steps = 0
    n_seen = 0

    max_batches = min(int(args.invariance_eval_batches), len(loader)) if hasattr(loader, "__len__") else int(args.invariance_eval_batches)
    iterator = progress_iter(loader, args, desc=desc, total=max_batches, leave=False)
    for batch in iterator:
        steps += 1
        native_view, native_sr = make_sr_view(batch, device, 1.0, 1.0, 0, 0, 0, augment=False)
        aug_view, aug_sr = make_sr_view(
            batch,
            device=device,
            min_factor=args.invariance_eval_min_factor,
            max_factor=args.invariance_eval_max_factor,
            round_to=args.sr_aug_round_to,
            min_hz=args.sr_aug_min_hz,
            max_hz=args.sr_aug_max_hz,
            augment=True,
        )

        out_native = model(native_view, native_sr, grl_lambda=0.0, return_sr_logits=True)
        out_aug = model(aug_view, aug_sr, grl_lambda=0.0, return_sr_logits=True)

        cosine = F.cosine_similarity(out_native["embedding"], out_aug["embedding"], dim=-1).mean()
        totals["embedding_cosine"] += float(cosine.detach().cpu())
        totals["embedding_cosine_distance"] += float((1.0 - cosine).detach().cpu())

        pred_losses: list[torch.Tensor] = []
        for name in DATASET_NAMES:
            indices = [i for i, dataset_name in enumerate(batch["dataset"]) if dataset_name == name]
            valid = valid_counts[name]
            if not indices or valid <= 0:
                continue
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            p_native = torch.sigmoid(out_native["logits"][name][idx, :valid])
            p_aug = torch.sigmoid(out_aug["logits"][name][idx, :valid])
            pred_losses.append(F.mse_loss(p_native, p_aug))
        if pred_losses:
            totals["prediction_mse"] += float(torch.stack(pred_losses).mean().detach().cpu())

        sr_logits = out_aug.get("sr_logits")
        if sr_logits is not None:
            labels = sr_bucket_labels(aug_sr, args, device)
            pred = sr_logits.argmax(dim=-1)
            totals["sr_adversary_acc"] += float((pred == labels).float().mean().detach().cpu())

        n_seen += len(aug_sr)
        set_progress_postfix(
            iterator,
            {
                "cos": totals["embedding_cosine"] / max(steps, 1),
                "mse": totals.get("prediction_mse", 0.0) / max(steps, 1),
            },
        )
        if steps >= args.invariance_eval_batches:
            break

    denom = max(steps, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    metrics["num_batches"] = float(steps)
    metrics["num_examples"] = float(n_seen)
    return metrics


def save_checkpoint(
    path: Path,
    model: SRInvariantBioacousticClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def load_model_weights(path: Path, model: SRInvariantBioacousticClassifier, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return int(checkpoint.get("epoch", 0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FNO/SIREN sample-rate-invariant bioacoustic model.")

    p.add_argument("--root", default="/mnt/volDISI_conci_Datasets/bioacoustics")
    p.add_argument("--run-dir", default="/home/mmlab/Desktop/nfo_bioacoustics/_REFACTORED/runs/fno_siren")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--classes-per-dataset", type=int, default=10)
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--normalize", default="peak", choices=["none", "peak", "rms"])
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--freq-bins", type=int, default=160)
    p.add_argument("--time-bins", type=int, default=256)
    p.add_argument("--freq-min-hz", type=float, default=5.0)
    p.add_argument("--freq-max-hz", type=float, default=120000.0)
    p.add_argument("--window-ms", default="32,64,128")
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--fno-modes-freq", type=int, default=24)
    p.add_argument("--fno-modes-time", type=int, default=24)
    p.add_argument("--siren-channels", type=int, default=16)
    p.add_argument("--siren-hidden-features", type=int, default=64)
    p.add_argument("--siren-hidden-layers", type=int, default=2)
    p.add_argument("--siren-first-omega-0", type=float, default=30.0)
    p.add_argument("--siren-hidden-omega-0", type=float, default=30.0)

    p.add_argument(
        "--aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic sample-rate views during training. Use --no-aug to disable.",
    )
    p.add_argument("--sr-aug-min-factor", type=float, default=0.75)
    p.add_argument("--sr-aug-max-factor", type=float, default=1.125)
    p.add_argument("--sr-aug-round-to", type=int, default=1000)
    p.add_argument("--sr-aug-min-hz", type=int, default=0)
    p.add_argument("--sr-aug-max-hz", type=int, default=0)
    p.add_argument("--sampler-power", type=float, default=0.5)
    p.add_argument("--pos-weight-cap", type=float, default=30.0)

    p.add_argument("--embedding-consistency-weight", type=float, default=0.1)
    p.add_argument("--prediction-consistency-weight", type=float, default=0.05)
    p.add_argument("--sr-adversary-weight", type=float, default=0.05)
    p.add_argument("--grl-lambda", type=float, default=1.0)
    p.add_argument("--grl-warmup-epochs", type=int, default=5)
    p.add_argument("--sr-bucket-edges", default="500,4000,12000,20000,28000,44000,88000")

    p.add_argument("--invariance-eval-batches", type=int, default=10)
    p.add_argument("--invariance-eval-min-factor", type=float, default=0.5)
    p.add_argument("--invariance-eval-max-factor", type=float, default=1.5)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-threshold", type=float, default=0.5)

    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--plot-max-points", type=int, default=4000)
    p.add_argument("--plot-max-heatmap-rows", type=int, default=500)
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--no-progress", action="store_true")

    return p.parse_args()


def build_dataloaders(
    manifest_info: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = Bioacoustic_Dataset(
        manifest_info["train_manifest"],
        manifest_info["classes_csv"],
        target_sample_rate=None,
        clip_seconds=args.clip_seconds,
        normalize=args.normalize,
    )
    val_ds = Bioacoustic_Dataset(
        manifest_info["val_manifest"],
        manifest_info["classes_csv"],
        target_sample_rate=None,
        clip_seconds=args.clip_seconds,
        crop_mode="center",
        normalize=args.normalize,
    )
    test_ds = Bioacoustic_Dataset(
        manifest_info["test_manifest"],
        manifest_info["classes_csv"],
        target_sample_rate=None,
        clip_seconds=args.clip_seconds,
        crop_mode="center",
        normalize=args.normalize,
    )

    batch_sampler = DatasetBalancedBatchSampler(
        train_ds,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        sampler_power=args.sampler_power,
        seed=args.seed,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_audio_samples,
    }

    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def build_model(args: argparse.Namespace, device: torch.device) -> SRInvariantBioacousticClassifier:
    window_ms = parse_float_sequence(args.window_ms)
    sr_adversary_buckets = len(parse_float_sequence(args.sr_bucket_edges)) + 1

    return SRInvariantBioacousticClassifier(
        datasets=DATASET_NAMES,
        classes_per_dataset=args.classes_per_dataset,
        freq_bins=args.freq_bins,
        time_bins=args.time_bins,
        freq_min_hz=args.freq_min_hz,
        freq_max_hz=args.freq_max_hz,
        window_ms=window_ms,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        sr_adversary_buckets=sr_adversary_buckets,
        fno_modes_freq=args.fno_modes_freq,
        fno_modes_time=args.fno_modes_time,
        siren_channels=args.siren_channels,
        siren_hidden_features=args.siren_hidden_features,
        siren_hidden_layers=args.siren_hidden_layers,
        siren_first_omega_0=args.siren_first_omega_0,
        siren_hidden_omega_0=args.siren_hidden_omega_0,
    ).to(device)




def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    root = Path(args.root).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else root / "runs" / "fno_siren"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    manifest_info = build_reduced_manifests(
        root=root,
        run_dir=run_dir,
        classes_per_dataset=args.classes_per_dataset,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (run_dir / "split_sizes.json").write_text(
        json.dumps(manifest_info["split_sizes"], indent=2),
        encoding="utf-8",
    )

    print(f"run_dir={run_dir}", flush=True)
    print(
        "splits "
        f"train={manifest_info['split_sizes']['train']} "
        f"val={manifest_info['split_sizes']['val']} "
        f"test={manifest_info['split_sizes']['test']}",
        flush=True,
    )

    train_loader, val_loader, test_loader = build_dataloaders(manifest_info, args, device)
    model = build_model(args, device)
    print(f"model_parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    print(f"dataset_names={DATASET_NAMES}", flush=True)
    print(f"window_ms={parse_float_sequence(args.window_ms)}", flush=True)
    print(f"sr_bucket_edges={parse_float_sequence(args.sr_bucket_edges)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    pos_weights, valid_counts = build_pos_weights(
        manifest_info["train_manifest"],
        args.classes_per_dataset,
        cap=args.pos_weight_cap,
        device=device,
    )
    (run_dir / "valid_counts.json").write_text(json.dumps(valid_counts, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            pos_weights=pos_weights,
            valid_counts=valid_counts,
            epoch=epoch,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            pos_weights=pos_weights,
            valid_counts=valid_counts,
            desc=f"val {epoch:03d}",
        )
        val_invariance = evaluate_sample_rate_invariance(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            valid_counts=valid_counts,
            desc=f"val inv {epoch:03d}",
        )

        epoch_seconds = time.time() - epoch_start
        score = val_metrics["mean_macro_f1"]

        record = {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "train": train_metrics,
            "val": val_metrics,
            "val_invariance": val_invariance,
        }
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, val_metrics, args)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, val_metrics, args)

        inv_text = ""
        if val_invariance:
            inv_text = (
                f" val_inv_cos={val_invariance.get('embedding_cosine', 0.0):.4f}"
                f" val_inv_pred_mse={val_invariance.get('prediction_mse', 0.0):.6f}"
                f" val_sr_adv_acc={val_invariance.get('sr_adversary_acc', 0.0):.4f}"
            )

        print(
            f"epoch={epoch:03d} "
            f"time={epoch_seconds:.1f}s "
            f"train_loss={train_metrics.get('loss', 0.0):.4f} "
            f"train_cls={train_metrics.get('classification', 0.0):.4f} "
            f"train_grl={train_metrics.get('grl_lambda', 0.0):.4f} "
            f"{compact_metrics(val_metrics, prefix='val_')}"
            f"{inv_text} "
            f"best_epoch={best_epoch:03d}",
            flush=True,
        )

    best_path = run_dir / "best.pt"
    loaded_epoch = load_model_weights(best_path, model, device)
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        args=args,
        pos_weights=pos_weights,
        valid_counts=valid_counts,
        desc="test",
    )
    test_invariance = evaluate_sample_rate_invariance(
        model=model,
        loader=test_loader,
        device=device,
        args=args,
        valid_counts=valid_counts,
        desc="test inv",
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": str(best_path),
                "epoch": loaded_epoch,
                "test": test_metrics,
                "test_invariance": test_invariance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.no_plots:
        make_all_plots(
            model=model,
            loader=test_loader,
            device=device,
            args=args,
            valid_counts=valid_counts,
            classes_csv=manifest_info["classes_csv"],
            run_dir=run_dir,
            split_name_for_plot="test",
            split_metrics=test_metrics,
            history=history,
            dataset_names=DATASET_NAMES,
            make_sr_view_fn=make_sr_view,
            show_progress=not bool(args.no_progress),
        )

    inv_text = ""
    if test_invariance:
        inv_text = (
            f" test_inv_cos={test_invariance.get('embedding_cosine', 0.0):.4f}"
            f" test_inv_pred_mse={test_invariance.get('prediction_mse', 0.0):.6f}"
            f" test_sr_adv_acc={test_invariance.get('sr_adversary_acc', 0.0):.4f}"
        )

    print(
        f"test_checkpoint={best_path} epoch={loaded_epoch:03d} "
        f"{compact_metrics(test_metrics, prefix='test_')}"
        f"{inv_text}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
