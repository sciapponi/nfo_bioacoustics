from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


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


def load_class_names(classes_csv: Path) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for row in read_csv(classes_csv):
        dataset = row.get("dataset", "")
        if not dataset:
            continue
        out.setdefault(dataset, {})[int(row["class_index"])] = row.get("class_name", str(row["class_index"]))
    return out


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def class_names_for_dataset(classes_csv: Path, dataset_names: Sequence[str]) -> dict[str, list[str]]:
    raw = load_class_names(classes_csv)
    out: dict[str, list[str]] = {}

    for dataset in dataset_names:
        if dataset not in raw or not raw[dataset]:
            out[dataset] = []
            continue
        max_idx = max(raw[dataset].keys())
        out[dataset] = [raw[dataset].get(i, str(i)) for i in range(max_idx + 1)]

    return out


def binary_class_metrics_np(
    target: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    pred = (prob >= float(threshold)).astype(np.float32)
    target = target.astype(np.float32)

    tp = (pred * target).sum(axis=0)
    fp = (pred * (1.0 - target)).sum(axis=0)
    fn = ((1.0 - pred) * target).sum(axis=0)
    support = target.sum(axis=0)

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def write_class_metrics_csv(
    path: Path,
    dataset_name: str,
    class_names: list[str],
    metrics: dict[str, np.ndarray],
) -> None:
    rows: list[dict[str, Any]] = []
    n = len(metrics["f1"])

    for i in range(n):
        rows.append(
            {
                "dataset": dataset_name,
                "class_index": i,
                "class_name": class_names[i] if i < len(class_names) else str(i),
                "precision": float(metrics["precision"][i]),
                "recall": float(metrics["recall"][i]),
                "f1": float(metrics["f1"][i]),
                "support": int(metrics["support"][i]),
                "tp": int(metrics["tp"][i]),
                "fp": int(metrics["fp"][i]),
                "fn": int(metrics["fn"][i]),
            }
        )

    write_csv(
        path,
        rows,
        [
            "dataset",
            "class_index",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
            "tp",
            "fp",
            "fn",
        ],
    )


def plot_training_history(
    history: list[dict[str, Any]],
    out_dir: Path,
    dataset_names: Sequence[str],
) -> None:
    if not history:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]

    train_loss = [float(row.get("train", {}).get("loss", np.nan)) for row in history]
    val_loss = [float(row.get("val", {}).get("loss", np.nan)) for row in history]
    val_mean_f1 = [float(row.get("val", {}).get("mean_macro_f1", np.nan)) for row in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "history_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, val_mean_f1, label="val mean macro F1")

    for dataset_name in dataset_names:
        values = [
            float(row.get("val", {}).get(f"{dataset_name}_macro_f1", np.nan))
            for row in history
        ]
        plt.plot(epochs, values, label=f"{dataset_name} macro F1")

    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.0)
    plt.title("Validation F1")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "history_f1.png", dpi=180)
    plt.close()


def plot_dataset_summary(
    metrics: dict[str, float],
    out_dir: Path,
    split_name_for_plot: str,
    dataset_names: Sequence[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    macro: list[float] = []
    micro: list[float] = []

    for dataset_name in dataset_names:
        macro_key = f"{dataset_name}_macro_f1"
        micro_key = f"{dataset_name}_micro_f1"

        if macro_key in metrics:
            names.append(dataset_name)
            macro.append(float(metrics.get(macro_key, 0.0)))
            micro.append(float(metrics.get(micro_key, 0.0)))

    if not names:
        return

    x = np.arange(len(names))
    width = 0.38

    plt.figure(figsize=(max(7, len(names) * 1.2), 4))
    plt.bar(x - width / 2, macro, width, label="macro F1")
    plt.bar(x + width / 2, micro, width, label="micro F1")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.0)
    plt.title(f"{split_name_for_plot} dataset-level classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{split_name_for_plot}_dataset_f1.png", dpi=180)
    plt.close()


def plot_class_metric_heatmap(
    metrics: dict[str, np.ndarray],
    class_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    values = np.stack(
        [
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        ],
        axis=0,
    )

    labels = ["precision", "recall", "F1"]
    class_labels = [
        class_names[i] if i < len(class_names) else str(i)
        for i in range(values.shape[1])
    ]

    plt.figure(figsize=(max(8, values.shape[1] * 0.7), 3.5))
    plt.imshow(values, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="score")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_probability_heatmap(
    prob: np.ndarray,
    target: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
    max_rows: int,
) -> None:
    if prob.size == 0:
        return

    primary = np.argmax(target, axis=1)
    has_label = target.sum(axis=1) > 0
    order = np.lexsort((np.arange(len(primary)), primary))
    order = order[has_label[order]]

    if len(order) == 0:
        return

    if len(order) > max_rows:
        idx = np.linspace(0, len(order) - 1, max_rows).astype(int)
        order = order[idx]

    sorted_prob = prob[order]

    class_labels = [
        class_names[i] if i < len(class_names) else str(i)
        for i in range(prob.shape[1])
    ]

    plt.figure(figsize=(max(8, prob.shape[1] * 0.7), 6))
    plt.imshow(sorted_prob, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="predicted probability")
    plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45, ha="right")
    plt.ylabel("Samples sorted by true class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


@torch.no_grad()
def collect_plot_data(
    model: Any,
    loader: Any,
    device: torch.device,
    args: Any,
    valid_counts: dict[str, int],
    class_names: dict[str, list[str]],
    dataset_names: Sequence[str],
    make_sr_view_fn: Callable[..., tuple[list[torch.Tensor], list[int]]],
    show_progress: bool,
) -> dict[str, Any]:
    model.eval()

    by_dataset: dict[str, dict[str, list[Any]]] = {
        name: {
            "prob": [],
            "target": [],
            "embedding": [],
            "primary_label": [],
            "primary_name": [],
            "sample_id": [],
        }
        for name in dataset_names
    }

    global_embeddings: list[np.ndarray] = []
    global_dataset: list[str] = []
    global_primary_label: list[int] = []
    global_primary_name: list[str] = []

    iterator = tqdm(
        loader,
        total=len(loader) if hasattr(loader, "__len__") else None,
        desc="collect plot data",
        leave=False,
        dynamic_ncols=True,
        disable=not bool(show_progress),
    )

    for batch in iterator:
        view, sr = make_sr_view_fn(batch, device, 1.0, 1.0, 0, 0, 0, augment=False)
        out = model(view, sr, grl_lambda=0.0, return_sr_logits=False)

        embeddings = out["embedding"].detach().float().cpu().numpy()
        batch_datasets = list(batch["dataset"])

        for i, dataset_name in enumerate(batch_datasets):
            if dataset_name not in by_dataset:
                continue

            valid = int(valid_counts.get(dataset_name, args.classes_per_dataset))
            valid = min(valid, args.classes_per_dataset)
            if valid <= 0:
                continue

            logits = out["logits"][dataset_name][i, :valid]
            prob = torch.sigmoid(logits).detach().float().cpu().numpy()

            target = np.zeros(valid, dtype=np.float32)
            labels = parse_int_list(batch["metadata"][i].get("label_indices", ""))

            for label in labels:
                if 0 <= label < valid:
                    target[label] = 1.0

            active = np.flatnonzero(target > 0)
            primary = int(active[0]) if len(active) else -1

            names = class_names.get(dataset_name, [])
            primary_name = names[primary] if 0 <= primary < len(names) else "none"

            sample_id = (
                batch["metadata"][i].get("global_id")
                or batch["metadata"][i].get("sample_id")
                or str(i)
            )

            by_dataset[dataset_name]["prob"].append(prob)
            by_dataset[dataset_name]["target"].append(target)
            by_dataset[dataset_name]["embedding"].append(embeddings[i])
            by_dataset[dataset_name]["primary_label"].append(primary)
            by_dataset[dataset_name]["primary_name"].append(primary_name)
            by_dataset[dataset_name]["sample_id"].append(sample_id)

            global_embeddings.append(embeddings[i])
            global_dataset.append(dataset_name)
            global_primary_label.append(primary)
            global_primary_name.append(primary_name)

    packed: dict[str, Any] = {
        "by_dataset": {},
        "global": {
            "embedding": np.stack(global_embeddings, axis=0) if global_embeddings else np.zeros((0, 1)),
            "dataset": np.array(global_dataset),
            "primary_label": np.array(global_primary_label),
            "primary_name": np.array(global_primary_name),
        },
    }

    for dataset_name, values in by_dataset.items():
        if values["prob"]:
            packed["by_dataset"][dataset_name] = {
                "prob": np.stack(values["prob"], axis=0),
                "target": np.stack(values["target"], axis=0),
                "embedding": np.stack(values["embedding"], axis=0),
                "primary_label": np.array(values["primary_label"]),
                "primary_name": np.array(values["primary_name"]),
                "sample_id": np.array(values["sample_id"]),
            }
        else:
            packed["by_dataset"][dataset_name] = {
                "prob": np.zeros((0, 0), dtype=np.float32),
                "target": np.zeros((0, 0), dtype=np.float32),
                "embedding": np.zeros((0, args.embedding_dim), dtype=np.float32),
                "primary_label": np.array([], dtype=np.int64),
                "primary_name": np.array([]),
                "sample_id": np.array([]),
            }

    return packed


def compute_tsne(
    embedding: np.ndarray,
    max_points: int,
    perplexity: float,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if TSNE is None:
        print("Skipping t-SNE: scikit-learn is not installed.", flush=True)
        return None, None

    if embedding.shape[0] < 5:
        print("Skipping t-SNE: fewer than 5 samples.", flush=True)
        return None, None

    n = embedding.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        idx = np.sort(idx)
        emb = embedding[idx]
    else:
        idx = np.arange(n)
        emb = embedding

    effective_perplexity = min(float(perplexity), max(2.0, float(emb.shape[0] - 1) / 3.0))

    coords = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(emb)

    return coords, idx


def write_tsne_csv(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    extra: np.ndarray | None = None,
    label_field: str = "label",
    extra_field: str = "extra",
) -> None:
    rows: list[dict[str, Any]] = []
    for i in range(coords.shape[0]):
        row: dict[str, Any] = {
            "index": i,
            "tsne_1": float(coords[i, 0]),
            "tsne_2": float(coords[i, 1]),
            label_field: str(labels[i]),
        }
        if extra is not None:
            row[extra_field] = str(extra[i])
        rows.append(row)

    fields = ["index", "tsne_1", "tsne_2", label_field]
    if extra is not None:
        fields.append(extra_field)
    write_csv(path, rows, fields)


def plot_tsne_by_dataset(
    embedding: np.ndarray,
    datasets: np.ndarray,
    primary_names: np.ndarray,
    out_path: Path,
    csv_path: Path,
    max_points: int,
    perplexity: float,
    seed: int,
    dataset_names: Sequence[str],
) -> None:
    coords, idx = compute_tsne(embedding, max_points, perplexity, seed)
    if coords is None or idx is None:
        return

    ds = datasets[idx]
    cls = primary_names[idx]

    write_tsne_csv(
        csv_path,
        coords=coords,
        labels=ds,
        extra=cls,
        label_field="dataset",
        extra_field="primary_class",
    )

    plt.figure(figsize=(7, 6))
    for dataset_name in dataset_names:
        mask = ds == dataset_name
        if mask.any():
            plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.75, label=dataset_name)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Embedding t-SNE by dataset")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_tsne_by_class(
    embedding: np.ndarray,
    primary_names: np.ndarray,
    out_path: Path,
    csv_path: Path,
    title: str,
    max_points: int,
    perplexity: float,
    seed: int,
) -> None:
    if embedding.shape[0] < 5:
        return

    coords, idx = compute_tsne(embedding, max_points, perplexity, seed)
    if coords is None or idx is None:
        return

    labels = primary_names[idx]

    write_tsne_csv(
        csv_path,
        coords=coords,
        labels=labels,
        label_field="primary_class",
    )

    plt.figure(figsize=(7, 6))
    for name in sorted(set(labels.tolist())):
        mask = labels == name
        if mask.any():
            plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.75, label=name)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_prediction_table(
    path: Path,
    dataset_name: str,
    data: dict[str, Any],
    class_names: list[str],
    threshold: float,
) -> None:
    rows: list[dict[str, Any]] = []
    prob = data["prob"]
    target = data["target"]

    for i in range(prob.shape[0]):
        true_indices = np.flatnonzero(target[i] > 0).tolist()
        pred_indices = np.flatnonzero(prob[i] >= float(threshold)).tolist()
        top1 = int(np.argmax(prob[i])) if prob.shape[1] else -1

        rows.append(
            {
                "dataset": dataset_name,
                "sample_id": str(data["sample_id"][i]),
                "true_indices": ";".join(str(x) for x in true_indices),
                "true_names": ";".join(class_names[x] if x < len(class_names) else str(x) for x in true_indices),
                "pred_indices": ";".join(str(x) for x in pred_indices),
                "pred_names": ";".join(class_names[x] if x < len(class_names) else str(x) for x in pred_indices),
                "top1_index": top1,
                "top1_name": class_names[top1] if 0 <= top1 < len(class_names) else "",
                "top1_prob": float(np.max(prob[i])) if prob.shape[1] else 0.0,
            }
        )

    write_csv(
        path,
        rows,
        [
            "dataset",
            "sample_id",
            "true_indices",
            "true_names",
            "pred_indices",
            "pred_names",
            "top1_index",
            "top1_name",
            "top1_prob",
        ],
    )


def make_all_plots(
    model: Any,
    loader: Any,
    device: torch.device,
    args: Any,
    valid_counts: dict[str, int],
    classes_csv: Path,
    run_dir: Path,
    split_name_for_plot: str,
    split_metrics: dict[str, float],
    history: list[dict[str, Any]] | None = None,
    dataset_names: Sequence[str] = (),
    make_sr_view_fn: Callable[..., tuple[list[torch.Tensor], list[int]]] | None = None,
    show_progress: bool = True,
) -> None:
    if not dataset_names:
        raise ValueError("dataset_names must be provided")
    if make_sr_view_fn is None:
        raise ValueError("make_sr_view_fn must be provided")

    dataset_names = tuple(dataset_names)
    plot_dir = run_dir / "plots" / split_name_for_plot
    plot_dir.mkdir(parents=True, exist_ok=True)

    names_by_dataset = class_names_for_dataset(classes_csv, dataset_names)

    if history is not None:
        plot_training_history(history, run_dir / "plots", dataset_names)

    plot_dataset_summary(split_metrics, plot_dir, split_name_for_plot, dataset_names)

    packed = collect_plot_data(
        model=model,
        loader=loader,
        device=device,
        args=args,
        valid_counts=valid_counts,
        class_names=names_by_dataset,
        dataset_names=dataset_names,
        make_sr_view_fn=make_sr_view_fn,
        show_progress=show_progress,
    )

    for dataset_name in dataset_names:
        data = packed["by_dataset"][dataset_name]
        prob = data["prob"]
        target = data["target"]
        class_names = names_by_dataset.get(dataset_name, [])

        if prob.size == 0:
            continue

        metrics = binary_class_metrics_np(
            target=target,
            prob=prob,
            threshold=args.eval_threshold,
        )

        dataset_dir = plot_dir / safe_name(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        write_class_metrics_csv(
            dataset_dir / "class_metrics.csv",
            dataset_name=dataset_name,
            class_names=class_names,
            metrics=metrics,
        )

        save_prediction_table(
            dataset_dir / "predictions.csv",
            dataset_name=dataset_name,
            data=data,
            class_names=class_names,
            threshold=args.eval_threshold,
        )

        plot_class_metric_heatmap(
            metrics=metrics,
            class_names=class_names,
            out_path=dataset_dir / "class_metrics_heatmap.png",
            title=f"{split_name_for_plot} {dataset_name}: per-class classification",
        )

        plot_probability_heatmap(
            prob=prob,
            target=target,
            class_names=class_names,
            out_path=dataset_dir / "probability_heatmap.png",
            title=f"{split_name_for_plot} {dataset_name}: predicted probabilities",
            max_rows=args.plot_max_heatmap_rows,
        )

        plot_tsne_by_class(
            embedding=data["embedding"],
            primary_names=data["primary_name"],
            out_path=dataset_dir / "tsne_by_class.png",
            csv_path=dataset_dir / "tsne_by_class.csv",
            title=f"{split_name_for_plot} {dataset_name}: embedding t-SNE by class",
            max_points=args.plot_max_points,
            perplexity=args.tsne_perplexity,
            seed=args.seed,
        )

    plot_tsne_by_dataset(
        embedding=packed["global"]["embedding"],
        datasets=packed["global"]["dataset"],
        primary_names=packed["global"]["primary_name"],
        out_path=plot_dir / "tsne_by_dataset.png",
        csv_path=plot_dir / "tsne_by_dataset.csv",
        max_points=args.plot_max_points,
        perplexity=args.tsne_perplexity,
        seed=args.seed,
        dataset_names=dataset_names,
    )

    print(f"plots_saved={plot_dir}", flush=True)
