
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence


DATASETS = ("whales", "frogs", "birds", "bats")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_dataset_list(values: Sequence[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        for part in str(value).split(","):
            name = part.strip().lower()
            if not name:
                continue
            if name == "all":
                names.extend(DATASETS)
            else:
                names.append(name)
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset name(s): {unknown}. Valid: all, {', '.join(DATASETS)}")
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def parse_int_list(text: Any) -> list[int]:
    if text is None:
        return []
    out: list[int] = []
    for part in str(text).replace(",", ";").split(";"):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def gini(values: list[int]) -> float:
    vals = sorted(float(v) for v in values if v >= 0)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((i + 1) * value for i, value in enumerate(vals))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def class_table(classes: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in classes:
        out[row["dataset"]].append(row)
    for dataset in out:
        out[dataset].sort(key=lambda r: int(r["class_index"]))
    return dict(out)


def summarize_dataset(dataset: str, rows: list[dict[str, str]], class_rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    num_classes = len(class_rows)
    counts = [0 for _ in range(num_classes)]
    cardinality = Counter()
    split_counts = Counter()
    unlabeled = 0
    multilabel = 0

    for row in rows:
        labels = [idx for idx in parse_int_list(row.get("label_indices", "")) if 0 <= idx < num_classes]
        cardinality[len(labels)] += 1
        split_counts[row.get("split", "")] += 1
        if not labels:
            unlabeled += 1
        if len(labels) > 1:
            multilabel += 1
        for idx in labels:
            counts[idx] += 1

    nonzero = [count for count in counts if count > 0]
    total_positive = sum(counts)
    summary = {
        "dataset": dataset,
        "samples": len(rows),
        "classes": num_classes,
        "nonzero_classes": sum(1 for count in counts if count > 0),
        "zero_classes": sum(1 for count in counts if count == 0),
        "positive_labels": total_positive,
        "avg_labels_per_sample": safe_div(total_positive, len(rows)),
        "unlabeled_samples": unlabeled,
        "multilabel_samples": multilabel,
        "min_nonzero_class_count": min(nonzero) if nonzero else 0,
        "median_class_count": float(statistics.median(counts)) if counts else 0.0,
        "mean_class_count": float(statistics.mean(counts)) if counts else 0.0,
        "max_class_count": max(counts) if counts else 0,
        "imbalance_ratio_max_to_min_nonzero": safe_div(max(counts) if counts else 0, min(nonzero) if nonzero else 0),
        "gini_class_counts": gini(counts),
        "split_counts": dict(sorted(split_counts.items())),
        "label_cardinality_histogram": dict(sorted(cardinality.items())),
    }

    per_class: list[dict[str, Any]] = []
    class_name_by_index = {int(row["class_index"]): row["class_name"] for row in class_rows}
    for idx in range(num_classes):
        count = counts[idx]
        per_class.append(
            {
                "dataset": dataset,
                "class_index": idx,
                "class_name": class_name_by_index.get(idx, str(idx)),
                "count": count,
                "fraction_of_samples": safe_div(count, len(rows)),
                "fraction_of_positive_labels": safe_div(count, total_positive),
                "pos_weight_for_bce": safe_div(len(rows) - count, max(count, 1)),
            }
        )
    return summary, per_class


def print_report(summaries: list[dict[str, Any]], per_class_by_dataset: dict[str, list[dict[str, Any]]], k: int) -> None:
    print("\nDATASET SUMMARY")
    print("=" * 88)
    for row in summaries:
        print(
            f"{row['dataset']:>7} | "
            f"samples={row['samples']} "
            f"classes={row['classes']} "
            f"nonzero={row['nonzero_classes']} "
            f"avg_labels={row['avg_labels_per_sample']:.3f} "
            f"max/min={row['imbalance_ratio_max_to_min_nonzero']:.1f} "
            f"gini={row['gini_class_counts']:.3f} "
            f"splits={row['split_counts']}"
        )

    print("\nMOST FREQUENT CLASSES")
    print("=" * 88)
    for summary in summaries:
        dataset = summary["dataset"]
        rows = sorted(per_class_by_dataset[dataset], key=lambda r: int(r["count"]), reverse=True)
        print(f"\n{dataset}")
        for row in rows[:k]:
            print(f"  {row['count']:>8}  {row['class_index']:>4}  {row['class_name']}")

    print("\nLEAST FREQUENT NONZERO CLASSES")
    print("=" * 88)
    for summary in summaries:
        dataset = summary["dataset"]
        rows = [row for row in per_class_by_dataset[dataset] if int(row["count"]) > 0]
        rows = sorted(rows, key=lambda r: int(r["count"]))
        print(f"\n{dataset}")
        for row in rows[:k]:
            print(f"  {row['count']:>8}  {row['class_index']:>4}  {row['class_name']}")


def inspect_audio(root: Path, manifest_csv: Path, classes_csv: Path, selected: Sequence[str], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    from bioacoustic_dataset import Bioacoustic_Dataset

    out: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for dataset in selected:
        ds = Bioacoustic_Dataset(manifest_csv, classes_csv, datasets=[dataset], target_sample_rate=None)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        indices = indices[: min(n, len(indices))]
        rates = Counter()
        durations = []
        failures = 0
        for index in indices:
            try:
                item = ds[index]
                sr = int(item["sample_rate"])
                rates[sr] += 1
                durations.append(float(item["audio"].shape[-1]) / float(sr))
            except Exception:
                failures += 1
        out.append(
            {
                "dataset": dataset,
                "samples_inspected": len(indices),
                "failures": failures,
                "sample_rates_seen": dict(sorted(rates.items())),
                "duration_min": min(durations) if durations else 0.0,
                "duration_median": statistics.median(durations) if durations else 0.0,
                "duration_max": max(durations) if durations else 0.0,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize canonical bioacoustic manifests.")
    p.add_argument("--root", default="/mnt/volDISI_conci_Datasets/bioacoustics")
    p.add_argument("--manifest", default=None)
    p.add_argument("--classes", default=None)
    p.add_argument("--datasets", nargs="*", default=["all"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--inspect-audio", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    manifest_csv = Path(args.manifest) if args.manifest else root / "manifests" / "all_samples.csv"
    classes_csv = Path(args.classes) if args.classes else root / "manifests" / "all_classes.csv"
    out_dir = Path(args.out_dir) if args.out_dir else root / "manifests" / "summary"
    selected = parse_dataset_list(args.datasets)

    print("version: BIOACOUSTICS_SUMMARY_FROM_MANIFEST_2026_06_29")
    print(f"manifest: {manifest_csv}")
    print(f"classes:  {classes_csv}")
    if not manifest_csv.exists():
        raise SystemExit(f"Missing manifest: {manifest_csv}. Run prepare_all_datasets.py first.")
    if not classes_csv.exists():
        raise SystemExit(f"Missing classes CSV: {classes_csv}. Run prepare_all_datasets.py first.")

    rows = read_csv(manifest_csv)
    classes = read_csv(classes_csv)
    rows_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[row["dataset"]].append(row)
    classes_by_dataset = class_table(classes)

    missing = [dataset for dataset in selected if dataset not in rows_by_dataset or dataset not in classes_by_dataset]
    if missing:
        raise SystemExit(f"Manifest is missing selected dataset(s): {missing}")

    summaries: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    per_class_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in selected:
        summary, per_class = summarize_dataset(dataset, rows_by_dataset[dataset], classes_by_dataset[dataset])
        summaries.append(summary)
        per_class_by_dataset[dataset] = per_class
        all_class_rows.extend(per_class)

    print_report(summaries, per_class_by_dataset, k=args.top_k)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "dataset_summary.csv", summaries)
    write_csv(out_dir / "class_counts.csv", all_class_rows)

    audio_rows = inspect_audio(root, manifest_csv, classes_csv, selected, args.inspect_audio, args.seed)
    if audio_rows:
        write_csv(out_dir / "audio_inspection.csv", audio_rows)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": summaries,
                "class_counts": all_class_rows,
                "audio_inspection": audio_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nWrote: {out_dir / 'dataset_summary.csv'}")
    print(f"Wrote: {out_dir / 'class_counts.csv'}")
    print(f"Wrote: {out_dir / 'summary.json'}")
    if audio_rows:
        print(f"Wrote: {out_dir / 'audio_inspection.csv'}")
    return 0


if __name__ == "__main__":
    #raise SystemExit(main())
    main()
