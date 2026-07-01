#!/usr/bin/env python3
"""
Destructive clean rebuild for the bioacoustic datasets.

This script removes the contents of the output root, downloads the selected
sources from scratch, and writes one canonical manifest layout:

  <root>/
    raw/
      whales/
      frogs/
      birds/
      bats/
    processed/
      whales/manifest.csv, classes.csv, audio_float32.npy
      frogs/manifest.csv, classes.csv
      birds/manifest.csv, classes.csv
      bats/manifest.csv, classes.csv
    manifests/
      all_samples.csv
      all_classes.csv
      dataset_summary.csv
    .archives/
    .cache/

Training and summary code should read manifests/all_samples.csv and
manifests/all_classes.csv, not the source-specific raw folders.

Run:
  python download_datasets_clean_rebuild.py --yes

Optional:
  python download_datasets_clean_rebuild.py --yes --datasets whales frogs birds bats
  python download_datasets_clean_rebuild.py --yes --bats-url '<Mendeley Download All URL>'
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import requests

VERSION = "BIOACOUSTICS_CLEAN_REBUILD_2026_06_30"

DATASETS = ("whales", "frogs", "birds", "bats")

WHALES_REPO = "monster-monash/WhaleSounds"
BIRDSET_REPO = "DBD-research-group/BirdSet"
DEFAULT_BIRD_CONFIG = "HSN"

ANURASET_URL = "https://zenodo.org/records/8056090/files/anuraset.zip?download=1"

BAT_MENDELEY_ID = "g7jy5s65v7"
BAT_MENDELEY_VERSION = "1"
BAT_MENDELEY_PAGE = f"https://data.mendeley.com/datasets/{BAT_MENDELEY_ID}/{BAT_MENDELEY_VERSION}"
BAT_DEFAULT_ZIP = "bat_calls_deep_learning.zip"
BAT_PUBLIC_DOWNLOAD_TEMPLATE = "https://data.mendeley.com/public-files/datasets/{dataset_id}/files/{file_id}/file_downloaded"

AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}

BAT_SPECIES = [
    "Eptesicus serotinus",
    "Myotis daubentonii",
    "Nyctalus noctula",
    "Pipistrellus pipistrellus",
    "Pipistrellus pygmaeus",
    "Rhinolophus hipposideros",
]

RESERVE_GB = {
    "whales": 6.0,
    "frogs": 16.0,
    "birds": 14.0,
    "bats": 9.0,
}

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


class RebuildError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def gb_to_bytes(gb: float) -> int:
    return int(float(gb) * 1_000_000_000)


def bytes_to_gb(n: int) -> str:
    return f"{n / 1_000_000_000:.1f} GB"


def parse_dataset_list(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            name = part.strip().lower()
            if not name:
                continue
            if name == "all":
                out.extend(DATASETS)
            else:
                out.append(name)
    if not out:
        out = list(DATASETS)
    invalid = [name for name in out if name not in DATASETS]
    if invalid:
        raise SystemExit(f"Invalid dataset name(s): {invalid}. Valid: {', '.join(DATASETS)}")
    result: list[str] = []
    seen: set[str] = set()
    for name in out:
        if name not in seen:
            result.append(name)
            seen.add(name)
    return result


def ensure_safe_root(root: Path, allow_any_root: bool) -> None:
    root = root.resolve()
    dangerous = {Path("/"), Path.home().resolve(), Path("/mnt").resolve(), Path("/home").resolve()}
    if root in dangerous:
        raise SystemExit(f"Refusing to delete dangerous root: {root}")
    if not allow_any_root and "bioacoustic" not in root.name.lower():
        raise SystemExit(
            f"Refusing destructive rebuild because root name does not contain 'bioacoustic': {root}\n"
            "Use --allow-any-root only if this is intentional."
        )


def clean_root(root: Path, yes: bool, allow_any_root: bool) -> None:
    ensure_safe_root(root, allow_any_root=allow_any_root)
    if not yes:
        log(f"This will DELETE all contents of: {root}")
        answer = input("Type DELETE to continue: ").strip()
        if answer != "DELETE":
            raise SystemExit("Aborted.")

    try:
        if root.exists():
            log(f"delete existing contents: {root}")
            for child in root.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except PermissionError as exc:
        raise SystemExit(
            f"Permission denied while cleaning/creating {root}.\n"
            "Fix ownership once, then rerun without sudo:\n"
            f"  sudo mkdir -p '{root}'\n"
            f"  sudo chown -R $(id -u):$(id -g) '{root}'\n"
        ) from exc


def ensure_layout(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "raw": root / "raw",
        "processed": root / "processed",
        "manifests": root / "manifests",
        "archives": root / ".archives",
        "cache": root / ".cache",
        "hf_cache": root / ".cache" / "huggingface",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        (paths["raw"] / dataset).mkdir(parents=True, exist_ok=True)
        (paths["processed"] / dataset).mkdir(parents=True, exist_ok=True)
    return paths


def require_free_space(root: Path, datasets: Sequence[str], reserve_gb: Optional[float], no_disk_check: bool) -> None:
    if no_disk_check:
        return
    if reserve_gb is None:
        reserve_gb = sum(RESERVE_GB[d] for d in datasets)
    free = shutil.disk_usage(root).free
    required = gb_to_bytes(reserve_gb)
    log(f"disk reserve check: {reserve_gb:.1f} GB")
    if free < required:
        raise SystemExit(
            f"Not enough free disk space under {root}. Free: {bytes_to_gb(free)}. "
            f"Required reserve: {reserve_gb:.1f} GB."
        )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        dialect = csv.excel
    with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            return [], []
        return [dict(row) for row in reader], list(reader.fieldnames)


def read_csv_plain(path: Path) -> list[dict[str, str]]:
    rows, _ = read_csv_rows(path)
    return rows


def configure_hf_cache(root: Path) -> Path:
    hf_home = root / ".cache" / "huggingface"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    hf_home.mkdir(parents=True, exist_ok=True)
    return hf_home


def write_env_file(root: Path) -> None:
    hf_home = root / ".cache" / "huggingface"
    text = f'''#!/usr/bin/env bash
export HF_HOME="{hf_home}"
export HF_DATASETS_CACHE="{hf_home / 'datasets'}"
export HF_HUB_CACHE="{hf_home / 'hub'}"
'''
    out = root / "use_downloaded_data.sh"
    out.write_text(text, encoding="utf-8")
    out.chmod(0o755)


def request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) bioacoustic-clean-rebuild/1.0",
            "Accept": "text/html,application/json,application/zip,application/octet-stream,*/*",
        }
    )
    return session


def file_starts_like_zip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) in {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}
    except OSError:
        return False


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0 or not file_starts_like_zip(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return bool(zf.infolist())
    except zipfile.BadZipFile:
        return False


def response_is_text(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    return "text/html" in ctype or "application/json" in ctype or "text/plain" in ctype


def download_stream(session: requests.Session, url: str, dst: Path, expect_zip: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    dst.unlink(missing_ok=True)
    part.unlink(missing_ok=True)

    log(f"download: {url}")
    log(f"     to: {dst}")
    with session.get(url, stream=True, allow_redirects=True, timeout=60) as resp:
        if resp.status_code >= 400:
            raise RebuildError(f"HTTP {resp.status_code} for {url}")
        if expect_zip and response_is_text(resp):
            first = resp.raw.read(200, decode_content=True)
            raise RebuildError(
                f"URL did not return ZIP bytes. Content-Type={resp.headers.get('Content-Type')!r}. "
                f"First bytes={first[:120]!r}. URL={url}"
            )
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        next_report = 512 * 1024 * 1024
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    if total:
                        log(f"  {bytes_to_gb(done)} / {bytes_to_gb(total)}")
                    else:
                        log(f"  {bytes_to_gb(done)}")
                    next_report += 512 * 1024 * 1024
    part.rename(dst)
    if expect_zip and not is_valid_zip(dst):
        dst.unlink(missing_ok=True)
        raise RebuildError(f"Downloaded file is not a valid ZIP: {dst}")
    return dst


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if not is_valid_zip(zip_path):
        raise RebuildError(f"Archive is not a valid ZIP: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"extract: {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def list_audio_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS and not any(part.startswith(".") for part in p.parts)
    )


def norm_text(value: Any) -> str:
    text = str(value).strip().lower()
    for ch in ["-", "_", ".", "/", "\\", "(", ")", "[", "]", ":", " "]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def compact_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def intish(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        f = float(text)
        if f.is_integer():
            return str(int(f))
        return str(f).replace(".", "_")
    except Exception:
        return text


def split_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    for sep in [";", "|", ","]:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def encode_labels(labels: Iterable[str], label_to_index: dict[str, int]) -> tuple[str, str]:
    indices: list[int] = []
    names: list[str] = []
    for label in labels:
        label = str(label).strip()
        if label in label_to_index and label not in names:
            names.append(label)
            indices.append(label_to_index[label])
    pairs = sorted(zip(indices, names), key=lambda pair: pair[0])
    return ";".join(str(idx) for idx, _ in pairs), ";".join(name for _, name in pairs)


def safe_float(value: Any, default: str = "") -> str:
    try:
        if value is None or value == "":
            return default
        return str(float(value))
    except Exception:
        return default


def write_classes(dataset: str, class_names: Sequence[str], out_dir: Path) -> Path:
    rows = [
        {
            "dataset": dataset,
            "class_index": i,
            "class_name": str(name),
            "source_name": str(name),
        }
        for i, name in enumerate(class_names)
    ]
    path = out_dir / "classes.csv"
    write_csv(path, rows, CLASS_FIELDS)
    return path


def write_manifest(dataset: str, rows: list[dict[str, Any]], out_dir: Path) -> Path:
    for i, row in enumerate(rows):
        row.setdefault("dataset", dataset)
        row.setdefault("global_id", f"{dataset}:{row.get('split', '')}:{row.get('sample_id', i)}")
        for field in MANIFEST_FIELDS:
            row.setdefault(field, "")
    path = out_dir / "manifest.csv"
    write_csv(path, rows, MANIFEST_FIELDS)
    return path


def dataset_summary_from_rows(dataset: str, rows: list[dict[str, Any]], class_names: Sequence[str]) -> dict[str, Any]:
    counts = [0 for _ in class_names]
    cardinality = Counter()
    for row in rows:
        labels = [int(x) for x in str(row.get("label_indices", "")).split(";") if str(x).strip()]
        cardinality[len(labels)] += 1
        for idx in labels:
            if 0 <= idx < len(counts):
                counts[idx] += 1
    nonzero = [c for c in counts if c > 0]
    return {
        "dataset": dataset,
        "samples": len(rows),
        "classes": len(class_names),
        "nonzero_classes": sum(1 for c in counts if c > 0),
        "zero_classes": sum(1 for c in counts if c == 0),
        "positive_labels": sum(counts),
        "min_nonzero_class_count": min(nonzero) if nonzero else 0,
        "max_class_count": max(counts) if counts else 0,
        "imbalance_ratio_max_to_min_nonzero": (max(counts) / min(nonzero)) if nonzero else 0,
        "label_cardinality_histogram": dict(sorted(cardinality.items())),
    }


# ------------------------- Hugging Face datasets -------------------------


def load_hf_dataset(root: Path, repo: str, name: Optional[str]) -> Any:
    configure_hf_cache(root)
    try:
        import datasets
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit("Install Hugging Face datasets first: pip install 'datasets>=3.0,<4.0'") from exc
    if repo == BIRDSET_REPO:
        version = tuple(int(p) for p in datasets.__version__.split(".")[:2])
        if version >= (4, 0):
            raise SystemExit("BirdSet requires datasets<4.0. Run: pip install 'datasets>=3.0,<4.0'")
    kwargs: dict[str, Any] = {
        "path": repo,
        "cache_dir": str(root / ".cache" / "huggingface" / "datasets"),
        "trust_remote_code": True,
    }
    if name is not None:
        kwargs["name"] = name
    label = repo if name is None else f"{repo} / {name}"
    log(f"load Hugging Face dataset: {label}")
    from datasets import load_dataset
    ds = load_dataset(**kwargs)
    log(str(ds))
    return ds


def save_hf_to_disk(ds: Any, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"save_to_disk: {out_dir}")
    ds.save_to_disk(str(out_dir))


def prepare_whales(root: Path) -> dict[str, str]:
    dataset = "whales"
    raw_dir = root / "raw" / dataset / "WhaleSounds"
    out_dir = root / "processed" / dataset
    ds = load_hf_dataset(root, WHALES_REPO, None)
    save_hf_to_disk(ds, raw_dir)

    train = ds["train"]
    n = len(train)
    import numpy as np
    first_audio = np.asarray(train[0]["X"], dtype=np.float32).reshape(-1)
    length = int(first_audio.shape[0])
    audio_npy = out_dir / "audio_float32.npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(audio_npy, mode="w+", dtype="float32", shape=(n, length))
    rows: list[dict[str, Any]] = []
    labels_seen: set[int] = set()
    for i in range(n):
        row = train[i]
        x = np.asarray(row["X"], dtype=np.float32).reshape(-1)
        if x.shape[0] != length:
            raise RebuildError(f"WhaleSounds sample {i} length {x.shape[0]} != {length}")
        arr[i] = x
        label = int(row["y"])
        labels_seen.add(label)
        rows.append(
            {
                "dataset": dataset,
                "split": "train",
                "sample_id": f"whales_train_{i:06d}",
                "audio_path": "",
                "array_path": str(audio_npy),
                "array_index": i,
                "native_sample_rate": 250,
                "duration_seconds": length / 250.0,
                "label_indices": str(label),
                "label_names": str(label),
                "source": WHALES_REPO,
                "source_id": str(i),
            }
        )
    arr.flush()
    class_names = [str(i) for i in range(max(labels_seen) + 1 if labels_seen else 0)]
    classes = write_classes(dataset, class_names, out_dir)
    manifest = write_manifest(dataset, rows, out_dir)
    write_json(out_dir / "dataset_summary.json", dataset_summary_from_rows(dataset, rows, class_names))
    return {"manifest": str(manifest), "classes": str(classes), "raw": str(raw_dir)}


# ------------------------------ AnuraSet --------------------------------


def download_frogs_raw(root: Path, keep_archives: bool) -> Path:
    session = request_session()
    raw_parent = root / "raw" / "frogs"
    archive = root / ".archives" / "frogs" / "anuraset.zip"
    download_stream(session, ANURASET_URL, archive, expect_zip=True)
    extract_zip(archive, raw_parent)
    if not keep_archives:
        archive.unlink(missing_ok=True)
    return raw_parent


def useful_csv(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    if any(part.startswith(".") for part in parts):
        return False
    if "checkpoint" in name or ".ipynb_checkpoints" in parts:
        return False
    if name.startswith("."):
        return False
    return path.suffix.lower() == ".csv"


def find_audio_indexes(audio_paths: Sequence[Path], root: Path) -> dict[str, dict[str, Path]]:
    indexes = {
        "basename": {},
        "stem": {},
        "compact_name": {},
        "compact_stem": {},
        "compact_rel": {},
        "sample_number": {},
    }
    for path in audio_paths:
        indexes["basename"].setdefault(path.name.lower(), path)
        indexes["stem"].setdefault(path.stem.lower(), path)
        indexes["compact_name"].setdefault(compact_text(path.name), path)
        indexes["compact_stem"].setdefault(compact_text(path.stem), path)
        try:
            rel = path.relative_to(root)
            indexes["compact_rel"].setdefault(compact_text(rel.as_posix()), path)
        except Exception:
            pass
        m = re.search(r"sample\D*0*(\d+)$", compact_text(path.stem))
        if m:
            indexes["sample_number"].setdefault(str(int(m.group(1))), path)
    return indexes


def row_values(row: dict[str, str], names: Sequence[str]) -> list[str]:
    lower = {k.lower(): k for k in row.keys()}
    out: list[str] = []
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            value = str(row.get(key, "")).strip()
            if value:
                out.append(value)
    return out


def frog_candidate_names(row: dict[str, str]) -> list[str]:
    values: list[str] = []
    # Direct fields documented for AnuraSet or commonly present in metadata.
    values += row_values(row, ["sample_name", "sample", "filename", "file", "path", "audio_path", "filepath", "fname"])

    site_vals = row_values(row, ["site"])
    sample_vals = row_values(row, ["sample_name", "sample"])
    fname_vals = row_values(row, ["fname", "filename", "file"])
    min_vals = row_values(row, ["min_t", "start", "start_time", "start_second"])
    max_vals = row_values(row, ["max_t", "end", "end_time", "end_second"])

    for sample in sample_vals:
        values.append(sample)
        if not sample.lower().endswith(tuple(AUDIO_EXTENSIONS)):
            values.append(sample + ".wav")
        for site in site_vals:
            values.append(f"{site}/{sample}")
            values.append(f"{site}_{sample}")

    for fname in fname_vals:
        values.append(fname)
        if not fname.lower().endswith(tuple(AUDIO_EXTENSIONS)):
            values.append(fname + ".wav")
        for site in site_vals:
            values.append(f"{site}/{fname}")
            values.append(f"{site}_{fname}")
        for a in min_vals:
            for b in max_vals:
                ia = intish(a)
                ib = intish(b)
                values.append(f"{fname}_{ia}_{ib}.wav")
                values.append(f"{fname}_{ia}-{ib}.wav")
                for site in site_vals:
                    values.append(f"{site}/{fname}_{ia}_{ib}.wav")
                    values.append(f"{site}_{fname}_{ia}_{ib}.wav")

    # SAMPLE_00000 -> sample number fallback.
    for sample in sample_vals:
        m = re.search(r"sample\D*0*(\d+)", compact_text(sample))
        if m:
            values.append(f"__sample_number__:{int(m.group(1))}")

    # Deduplicate preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def resolve_frog_audio(row: dict[str, str], raw_root: Path, indexes: dict[str, dict[str, Path]]) -> Optional[Path]:
    for value in frog_candidate_names(row):
        if value.startswith("__sample_number__:"):
            key = value.split(":", 1)[1]
            path = indexes["sample_number"].get(key)
            if path is not None:
                return path
            continue
        p = Path(value)
        if p.is_absolute() and p.exists():
            return p
        candidate = raw_root / value
        if candidate.exists():
            return candidate
        lower_name = p.name.lower()
        lower_stem = p.stem.lower()
        for index_name, key in [
            ("basename", lower_name),
            ("stem", lower_stem),
            ("compact_name", compact_text(p.name)),
            ("compact_stem", compact_text(p.stem)),
            ("compact_rel", compact_text(value)),
        ]:
            path = indexes[index_name].get(key)
            if path is not None:
                return path
    return None


def is_binary_label_value(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return True
    if text in {"true", "false"}:
        return True
    try:
        return float(text) in (0.0, 1.0)
    except Exception:
        return False


def is_positive_binary(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "yes", "present"}:
        return True
    try:
        return float(text) == 1.0
    except Exception:
        return False


def detect_frog_label_columns(rows: list[dict[str, str]], fieldnames: Sequence[str]) -> list[str]:
    meta_names = {
        "sample_name", "sample", "fname", "filename", "file", "path", "audio_path", "filepath",
        "site", "date", "time", "min_t", "max_t", "start", "end", "start_time", "end_time",
        "species_number", "subset", "split", "fold", "duration", "id", "index",
    }
    label_cols: list[str] = []
    probe = rows[: min(5000, len(rows))]
    for col in fieldnames:
        if col is None:
            continue
        col_clean = str(col).strip()
        if not col_clean or col_clean.lower() in meta_names:
            continue
        values = [row.get(col, "") for row in probe]
        nonempty = [v for v in values if str(v).strip()]
        if not nonempty:
            continue
        if all(is_binary_label_value(v) for v in nonempty) and any(is_positive_binary(v) for v in nonempty):
            label_cols.append(col)
    return label_cols


def score_frog_metadata(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str], raw_root: Path, indexes: dict[str, dict[str, Path]]) -> dict[str, Any]:
    label_cols = detect_frog_label_columns(rows, fieldnames)
    if not rows or not label_cols:
        return {"csv": str(csv_path), "score": -1, "matches": 0, "label_cols": label_cols, "rows": len(rows)}
    probe = rows[: min(5000, len(rows))]
    matches = sum(1 for row in probe if resolve_frog_audio(row, raw_root, indexes) is not None)
    name_bonus = 1000 if csv_path.name.lower() == "metadata.csv" else 0
    checkpoint_penalty = -100000 if "checkpoint" in csv_path.name.lower() else 0
    score = matches * 100 + len(label_cols) * 10 + name_bonus + checkpoint_penalty + min(len(rows), 100000) // 1000
    return {"csv": str(csv_path), "score": score, "matches": matches, "label_cols": label_cols, "rows": len(rows)}


def select_frog_metadata(raw_root: Path, audio_paths: Sequence[Path], indexes: dict[str, dict[str, Path]]) -> tuple[Path, list[dict[str, str]], list[str], list[str], list[dict[str, Any]]]:
    csv_files = sorted(path for path in raw_root.rglob("*.csv") if useful_csv(path))
    diagnostics: list[dict[str, Any]] = []
    candidates: list[tuple[int, Path, list[dict[str, str]], list[str], list[str]]] = []
    for csv_path in csv_files:
        try:
            rows, fieldnames = read_csv_rows(csv_path)
        except Exception as exc:
            diagnostics.append({"csv": str(csv_path), "error": str(exc)})
            continue
        diag = score_frog_metadata(csv_path, rows, fieldnames, raw_root, indexes)
        diagnostics.append(diag)
        if rows and diag["label_cols"]:
            candidates.append((int(diag["score"]), csv_path, rows, fieldnames, list(diag["label_cols"])))
    if not candidates:
        raise RebuildError(f"No useful frog metadata CSV found under {raw_root}. See raw folder contents.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, csv_path, rows, fieldnames, label_cols = candidates[0]
    return csv_path, rows, fieldnames, label_cols, diagnostics


def prepare_frogs(root: Path, keep_archives: bool) -> dict[str, str]:
    dataset = "frogs"
    raw_root = download_frogs_raw(root, keep_archives=keep_archives)
    out_dir = root / "processed" / dataset
    audio_paths = list_audio_files(raw_root)
    if not audio_paths:
        raise RebuildError(f"No frog audio files found under {raw_root}")

    indexes = find_audio_indexes(audio_paths, raw_root)
    metadata_csv, rows_in, fieldnames, label_cols, diagnostics = select_frog_metadata(raw_root, audio_paths, indexes)
    log(f"frog metadata: {metadata_csv}")
    log(f"frog rows: {len(rows_in)}")
    log(f"frog audio files: {len(audio_paths)}")
    log(f"frog label columns: {len(label_cols)}")

    label_to_index = {name: i for i, name in enumerate(label_cols)}
    split_key = next((c for c in fieldnames if c.lower() in {"subset", "split", "set"}), None)
    out_rows: list[dict[str, Any]] = []
    missing_rows = 0
    for i, row in enumerate(rows_in):
        path = resolve_frog_audio(row, raw_root, indexes)
        if path is None:
            missing_rows += 1
            continue
        labels = [col for col in label_cols if is_positive_binary(row.get(col, ""))]
        label_indices, label_names = encode_labels(labels, label_to_index)
        sample_name = row.get("sample_name") or row.get("sample") or row.get("fname") or path.name
        out_rows.append(
            {
                "dataset": dataset,
                "split": str(row.get(split_key, "train")) if split_key else "train",
                "sample_id": str(sample_name),
                "audio_path": str(path),
                "array_path": "",
                "array_index": "",
                "native_sample_rate": 22050,
                "duration_seconds": 3.0,
                "label_indices": label_indices,
                "label_names": label_names,
                "source": "AnuraSet",
                "source_id": str(sample_name),
            }
        )

    # Last-resort fallback: if metadata and audio have the same cardinality but names do not match,
    # pair sorted metadata rows to sorted audio files. This keeps the rebuild usable while writing
    # an explicit warning file.
    used_row_order_fallback = False
    if not out_rows and len(rows_in) == len(audio_paths):
        used_row_order_fallback = True
        log("WARNING: frog filename matching failed; using row-order fallback because rows == audio files")
        sorted_audio = sorted(audio_paths)
        for i, (row, path) in enumerate(zip(rows_in, sorted_audio)):
            labels = [col for col in label_cols if is_positive_binary(row.get(col, ""))]
            label_indices, label_names = encode_labels(labels, label_to_index)
            sample_name = row.get("sample_name") or row.get("sample") or row.get("fname") or f"frogs_{i:06d}"
            out_rows.append(
                {
                    "dataset": dataset,
                    "split": str(row.get(split_key, "train")) if split_key else "train",
                    "sample_id": str(sample_name),
                    "audio_path": str(path),
                    "array_path": "",
                    "array_index": "",
                    "native_sample_rate": 22050,
                    "duration_seconds": 3.0,
                    "label_indices": label_indices,
                    "label_names": label_names,
                    "source": "AnuraSet,row_order_fallback",
                    "source_id": str(sample_name),
                }
            )

    if not out_rows:
        diag_path = raw_root / "frog_metadata_detection_failed.json"
        write_json(
            diag_path,
            {
                "metadata_csv": str(metadata_csv),
                "fieldnames": fieldnames,
                "label_cols": label_cols,
                "audio_count": len(audio_paths),
                "metadata_rows": len(rows_in),
                "candidate_csvs": diagnostics,
                "example_audio_files": [str(p.relative_to(raw_root)) for p in audio_paths[:50]],
                "example_metadata_rows": rows_in[:5],
            },
        )
        raise RebuildError(f"Frog metadata matched zero audio files. Wrote diagnostics to {diag_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    classes = write_classes(dataset, label_cols, out_dir)
    manifest = write_manifest(dataset, out_rows, out_dir)
    write_json(
        out_dir / "metadata_detection.json",
        {
            "metadata_csv": str(metadata_csv),
            "label_cols": label_cols,
            "audio_files": len(audio_paths),
            "metadata_rows": len(rows_in),
            "matched_rows": len(out_rows),
            "unmatched_metadata_rows": missing_rows,
            "used_row_order_fallback": used_row_order_fallback,
            "candidate_csvs": diagnostics,
        },
    )
    write_json(out_dir / "dataset_summary.json", dataset_summary_from_rows(dataset, out_rows, label_cols))
    return {"manifest": str(manifest), "classes": str(classes), "raw": str(raw_root)}


# ------------------------------ BirdSet --------------------------------


def get_hf_audio_path(value: Any) -> str:
    if isinstance(value, dict):
        for key in ["path", "file", "filename"]:
            v = value.get(key)
            if v:
                return str(v)
    if isinstance(value, str):
        return value
    return ""


def bird_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    labels.extend(split_labels(row.get("ebird_code_multilabel")))
    if not labels:
        labels.extend(split_labels(row.get("ebird_code")))
    for label in split_labels(row.get("ebird_code_secondary")):
        if label not in labels:
            labels.append(label)
    return labels


def prepare_birds(root: Path, bird_config: str) -> dict[str, str]:
    dataset = "birds"
    raw_dir = root / "raw" / dataset / f"BirdSet_{bird_config}"
    out_dir = root / "processed" / dataset
    ds = load_hf_dataset(root, BIRDSET_REPO, bird_config)
    try:
        from datasets import Audio
        for split in ds.keys():
            if "audio" in ds[split].column_names:
                ds[split] = ds[split].cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    save_hf_to_disk(ds, raw_dir)

    class_set: set[str] = set()
    for split in ds.keys():
        for row in ds[split]:
            class_set.update(bird_labels(row))
    class_names = sorted(class_set)
    if not class_names:
        raise RebuildError("BirdSet produced zero labels")
    label_to_index = {name: i for i, name in enumerate(class_names)}

    rows: list[dict[str, Any]] = []
    for split in ds.keys():
        for i, row in enumerate(ds[split]):
            labels = bird_labels(row)
            label_indices, label_names = encode_labels(labels, label_to_index)
            audio_path = get_hf_audio_path(row.get("audio")) or str(row.get("filepath", ""))
            duration = safe_float(row.get("length"), "")
            if not duration and row.get("start_time") not in (None, "") and row.get("end_time") not in (None, ""):
                try:
                    duration = str(float(row["end_time"]) - float(row["start_time"]))
                except Exception:
                    duration = ""
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "sample_id": f"birds_{bird_config}_{split}_{i:06d}",
                    "audio_path": audio_path,
                    "array_path": "",
                    "array_index": "",
                    "native_sample_rate": 32000,
                    "duration_seconds": duration,
                    "label_indices": label_indices,
                    "label_names": label_names,
                    "source": f"{BIRDSET_REPO}/{bird_config}",
                    "source_id": str(row.get("filepath", i)),
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = write_classes(dataset, class_names, out_dir)
    manifest = write_manifest(dataset, rows, out_dir)
    write_json(out_dir / "dataset_summary.json", dataset_summary_from_rows(dataset, rows, class_names))
    return {"manifest": str(manifest), "classes": str(classes), "raw": str(raw_dir), "bird_config": bird_config}


# ------------------------------- Bats -----------------------------------


def try_json(session: requests.Session, url: str, token: Optional[str]) -> Optional[Any]:
    headers = {"Accept": "application/vnd.mendeley-public-dataset.1+json, application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = session.get(url, headers=headers, timeout=60, allow_redirects=True)
        if resp.status_code >= 400:
            return None
        text = resp.text.strip()
        if "json" not in resp.headers.get("Content-Type", "").lower() and not text.startswith(("{", "[")):
            return None
        return resp.json()
    except Exception:
        return None


def walk_json(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_json(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_json(value)


def dedupe_urls(urls: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        url = str(url).strip()
        if not url or url in seen:
            continue
        if "manager/item" in url:
            continue
        seen.add(url)
        out.append(url)
    return out


def mendeley_file_candidates_from_api(session: requests.Session, token: Optional[str]) -> list[str]:
    urls = [
        f"https://api.mendeley.com/datasets/{BAT_MENDELEY_ID}?version={BAT_MENDELEY_VERSION}&fields=*&limit=100",
        f"https://api.mendeley.com/datasets/{BAT_MENDELEY_ID}/files?version={BAT_MENDELEY_VERSION}&fields=*&limit=100",
        f"https://data.mendeley.com/api/datasets/{BAT_MENDELEY_ID}/versions/{BAT_MENDELEY_VERSION}/files",
        f"https://data.mendeley.com/api/datasets/{BAT_MENDELEY_ID}/files",
    ]
    candidates: list[str] = []
    ids: list[str] = []
    guid = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    for url in urls:
        data = try_json(session, url, token=token)
        if data is None:
            continue
        for d in walk_json(data):
            for key in ["download_url", "downloadUrl", "download_uri", "downloadUri", "url", "href", "link"]:
                value = d.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    candidates.append(value)
            for key in ["id", "file_id", "fileId"]:
                value = str(d.get(key) or "")
                if re.fullmatch(guid, value):
                    ids.append(value)
            content = d.get("content_details") or d.get("contentDetails") or {}
            if isinstance(content, dict):
                value = str(content.get("id") or "")
                if re.fullmatch(guid, value):
                    ids.append(value)
    for file_id in ids:
        candidates.append(BAT_PUBLIC_DOWNLOAD_TEMPLATE.format(dataset_id=BAT_MENDELEY_ID, file_id=file_id))
    return dedupe_urls(candidates)


def mendeley_file_candidates_from_page(session: requests.Session) -> list[str]:
    try:
        resp = session.get(BAT_MENDELEY_PAGE, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return []
    candidates: list[str] = []
    for match in re.finditer(r"https?://[^\"'<> ]+", html):
        url = match.group(0).replace("\\u002F", "/")
        if "file_downloaded" in url or "download" in url.lower():
            candidates.append(url)
    pattern = rf"/public-files/datasets/{re.escape(BAT_MENDELEY_ID)}/files/([0-9a-fA-F-]{{36}})/file_downloaded"
    for file_id in re.findall(pattern, html):
        candidates.append(BAT_PUBLIC_DOWNLOAD_TEMPLATE.format(dataset_id=BAT_MENDELEY_ID, file_id=file_id))
    candidates.append(f"https://data.mendeley.com/datasets/{BAT_MENDELEY_ID}/{BAT_MENDELEY_VERSION}/files/download")
    return dedupe_urls(candidates)


def download_bat_zip(root: Path, bats_url: Optional[str], token: Optional[str]) -> Path:
    session = request_session()
    archive = root / ".archives" / "bats" / BAT_DEFAULT_ZIP
    candidates: list[str] = []
    if bats_url:
        candidates.append(bats_url)
    candidates += mendeley_file_candidates_from_api(session, token=token)
    candidates += mendeley_file_candidates_from_page(session)
    candidates = dedupe_urls(candidates)
    errors: list[str] = []
    for url in candidates:
        try:
            return download_stream(session, url, archive, expect_zip=True)
        except Exception as exc:
            errors.append(f"{url}\n  {exc}")
            archive.unlink(missing_ok=True)
    raise SystemExit(
        "Could not download the Mendeley bat dataset automatically.\n"
        f"Dataset page: {BAT_MENDELEY_PAGE}\n"
        "Open the page, copy the browser Download All URL, and rerun with --bats-url '<URL>'.\n"
        + "\n".join(errors)
    )


def infer_bat_species(path: Path) -> Optional[str]:
    text = norm_text(path)
    compact = compact_text(path)
    if "pygmaeus" in text or "ppyg" in compact or "pippyg" in compact:
        return "Pipistrellus pygmaeus"
    if "pipistrellus pipistrellus" in text or "ppip" in compact or "pippip" in compact:
        return "Pipistrellus pipistrellus"
    checks = [
        ("Eptesicus serotinus", ["eptesicus serotinus", "eser", "serotinus"]),
        ("Myotis daubentonii", ["myotis daubentonii", "mdau", "daubentonii"]),
        ("Nyctalus noctula", ["nyctalus noctula", "nnoc", "noctula"]),
        ("Rhinolophus hipposideros", ["rhinolophus hipposideros", "rhip", "hipposideros"]),
    ]
    for label, aliases in checks:
        if any(norm_text(alias) in text or compact_text(alias) in compact for alias in aliases):
            return label
    return None


def infer_bat_source(path: Path) -> str:
    text = norm_text(path)
    if "xeno" in text or "xc" in text:
        return "xeno-canto"
    if "chiro" in text:
        return "chirovox"
    return ""


def prepare_bats(root: Path, bats_url: Optional[str], token: Optional[str], keep_archives: bool) -> dict[str, str]:
    dataset = "bats"
    raw_dir = root / "raw" / dataset / "bat_calls_deep_learning"
    out_dir = root / "processed" / dataset
    archive = download_bat_zip(root, bats_url=bats_url, token=token)
    extract_zip(archive, raw_dir)
    if not keep_archives:
        archive.unlink(missing_ok=True)

    audio_paths = list_audio_files(raw_dir)
    if not audio_paths:
        raise RebuildError(f"No bat audio files found under {raw_dir}")
    label_to_index = {label: i for i, label in enumerate(BAT_SPECIES)}
    rows: list[dict[str, Any]] = []
    for i, path in enumerate(audio_paths):
        label = infer_bat_species(path)
        if label is None:
            continue
        label_indices, label_names = encode_labels([label], label_to_index)
        rows.append(
            {
                "dataset": dataset,
                "split": "train",
                "sample_id": f"bats_{i:06d}",
                "audio_path": str(path),
                "array_path": "",
                "array_index": "",
                "native_sample_rate": "",
                "duration_seconds": "",
                "label_indices": label_indices,
                "label_names": label_names,
                "source": f"Mendeley {BAT_MENDELEY_ID}.{BAT_MENDELEY_VERSION} {infer_bat_source(path)}".strip(),
                "source_id": str(path.relative_to(raw_dir)),
            }
        )
    if not rows:
        raise RebuildError(f"Bat audio was extracted but no species labels could be inferred from paths under {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = write_classes(dataset, BAT_SPECIES, out_dir)
    manifest = write_manifest(dataset, rows, out_dir)
    write_json(out_dir / "dataset_summary.json", dataset_summary_from_rows(dataset, rows, BAT_SPECIES))
    return {"manifest": str(manifest), "classes": str(classes), "raw": str(raw_dir)}


# ------------------------------- Combined outputs -------------------------------


def read_manifest(path: Path) -> list[dict[str, str]]:
    return read_csv_plain(path)


def combine_outputs(root: Path, results: dict[str, dict[str, str]]) -> None:
    all_samples: list[dict[str, Any]] = []
    all_classes: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        if dataset not in results:
            continue
        manifest = Path(results[dataset]["manifest"])
        classes = Path(results[dataset]["classes"])
        sample_rows = read_manifest(manifest)
        class_rows = read_manifest(classes)
        if not sample_rows:
            raise RebuildError(f"Manifest has zero rows: {manifest}")
        all_samples.extend(sample_rows)
        all_classes.extend(class_rows)
        summaries.append(dataset_summary_from_rows(dataset, sample_rows, [row["class_name"] for row in class_rows]))
    manifests_dir = root / "manifests"
    write_csv(manifests_dir / "all_samples.csv", all_samples, MANIFEST_FIELDS)
    write_csv(manifests_dir / "all_classes.csv", all_classes, CLASS_FIELDS)
    write_csv(
        manifests_dir / "dataset_summary.csv",
        summaries,
        [
            "dataset",
            "samples",
            "classes",
            "nonzero_classes",
            "zero_classes",
            "positive_labels",
            "min_nonzero_class_count",
            "max_class_count",
            "imbalance_ratio_max_to_min_nonzero",
            "label_cardinality_histogram",
        ],
    )
    write_json(root / "dataset_paths.json", results | {"all_samples_manifest": str(manifests_dir / "all_samples.csv"), "all_classes_csv": str(manifests_dir / "all_classes.csv")})
    log(f"manifest: {manifests_dir / 'all_samples.csv'}")
    log(f"classes:  {manifests_dir / 'all_classes.csv'}")
    log(f"summary:  {manifests_dir / 'dataset_summary.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Destructively rebuild all bioacoustic datasets into one canonical manifest layout.")
    p.add_argument("--root", default="/mnt/volDISI_conci_Datasets/bioacoustics")
    p.add_argument("--datasets", nargs="*", default=list(DATASETS), help="Dataset subset: all, whales, frogs, birds, bats. Commas accepted.")
    p.add_argument("--bird-config", default=DEFAULT_BIRD_CONFIG)
    p.add_argument("--bats-url", default=None, help="Optional direct Mendeley Download All / ZIP URL.")
    p.add_argument("--mendeley-token", default=os.environ.get("MENDELEY_TOKEN"))
    p.add_argument("--yes", action="store_true", help="Required for non-interactive deletion of root contents.")
    p.add_argument("--allow-any-root", action="store_true", help="Allow destructive deletion when root basename does not contain 'bioacoustic'.")
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--no-disk-check", action="store_true")
    p.add_argument("--reserve-gb", type=float, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    selected = parse_dataset_list(args.datasets)

    log(f"version: {VERSION}")
    log(f"root: {root}")
    log(f"datasets: {', '.join(selected)}")

    clean_root(root, yes=args.yes, allow_any_root=args.allow_any_root)
    ensure_layout(root)
    require_free_space(root, selected, args.reserve_gb, args.no_disk_check)
    write_env_file(root)

    results: dict[str, dict[str, str]] = {}
    if "whales" in selected:
        results["whales"] = prepare_whales(root)
    if "frogs" in selected:
        results["frogs"] = prepare_frogs(root, keep_archives=args.keep_archives)
    if "birds" in selected:
        results["birds"] = prepare_birds(root, bird_config=args.bird_config)
    if "bats" in selected:
        results["bats"] = prepare_bats(root, bats_url=args.bats_url, token=args.mendeley_token, keep_archives=args.keep_archives)

    missing = [name for name in selected if name not in results]
    if missing:
        raise RebuildError(f"Selected datasets were not prepared: {missing}")
    combine_outputs(root, results)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
