from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from train_sfi_bioacoustics import audio_info, discover_audio_files, parse_extensions
from whalenet_paper_staged import run_whalenet_paper_staged

# One user-facing script.
# It trains/evaluates all requested models, for fixed-rate Mel baselines,
# and writes experiment-level plus final summaries in one suite folder.

ALL_CLASSIFICATION_MODELS = (
    'sfi_fno,'
    'sfi_scatnet,'
    'mel_cnn,'
    'mel_efficientnet,'
    'mel_cnn_fno'
)

# Fixed-rate sensitivity applies only to models whose pipeline uses a target sample rate.
# Native-rate models are added as horizontal references in the summary plot.
DEFAULT_FIXED_RATE_MODELS = (
    'mel_cnn,'
    'mel_efficientnet,'
    'mel_cnn_fno'
)

NATIVE_REFERENCE_MODELS = ['sfi_fno', 'sfi_scatnet']


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')


def run_cmd(cmd: list[str], cwd: Path, skip_if: Path | None = None, skip_existing: bool = False) -> None:
    if skip_existing and skip_if is not None and skip_if.exists():
        print(f'SKIP existing: {skip_if}', flush=True)
        return
    print('\nRUN:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def dataset_sample_rates(root: Path, audio_extensions: str) -> tuple[int, dict[int, int]]:
    files = discover_audio_files(root, parse_extensions(audio_extensions))
    if not files:
        raise RuntimeError(f'No audio files found under {root}')
    counts: dict[int, int] = {}
    for p in files:
        sr, _, _ = audio_info(p)
        counts[int(sr)] = counts.get(int(sr), 0) + 1
    return max(counts), dict(sorted(counts.items()))


def four_rates_from_dataset(max_sr: int, min_rate: int = 1000) -> list[int]:
    raw = [max_sr / 8.0, max_sr / 4.0, max_sr / 2.0, max_sr]
    out: list[int] = []
    for r in raw:
        value = max(int(min_rate), int(round(r)))
        if value not in out:
            out.append(value)
    return out


def parse_rate_list(text: str) -> list[int]:
    text = str(text).strip()
    if not text:
        return []
    out: list[int] = []
    for part in text.replace(';', ',').split(','):
        part = part.strip()
        if part:
            value = int(float(part))
            if value not in out:
                out.append(value)
    return out


def parse_model_list(text: str) -> list[str]:
    return [p.strip() for p in str(text).replace(';', ',').split(',') if p.strip()]


def train_base_cmd(args: argparse.Namespace) -> list[str]:
    script = Path(__file__).resolve().parent / 'train_sfi_bioacoustics.py'
    cmd = [
        sys.executable, str(script),
        '--root', str(Path(args.root).expanduser().resolve()),
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--num-workers', str(args.num_workers),
        '--val-fraction', str(args.val_fraction),
        '--test-fraction', str(args.test_fraction),
        '--clip-seconds', str(args.clip_seconds),
        '--normalize', args.normalize,
        '--pad-mode', args.pad_mode,
        '--seed', str(args.seed),
        '--plot-reducer', 'tsne',
        '--plot-max-points', str(args.plot_max_points),
        '--transfer-sample-rates', '',
        '--cutoff-hz', '',
        '--baseline-channels', str(args.baseline_channels),
        '--embedding-dim', str(args.embedding_dim),
        '--device', args.device,
        '--whalenet-j', str(args.whalenet_j),
        '--whalenet-q', str(args.whalenet_q),
        '--whalenet-signal-len', str(args.whalenet_signal_len),
    ]
    if args.amp:
        cmd.append('--amp')
    if args.remove_duplicates:
        cmd.append('--remove-duplicates')
    if args.min_samples_per_class > 0:
        cmd += ['--min-samples-per-class', str(args.min_samples_per_class)]
    if args.no_progress:
        cmd.append('--no-progress')
    return cmd


def write_caption(path: Path, title: str, what: str, better: str, notes: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f'# {title}\n\n'
        f'## What this experiment does\n{what}\n\n'
        f'## What is better and why\n{better}\n'
    )
    if notes:
        text += f'\n## Notes\n{notes}\n'
    path.write_text(text, encoding='utf-8')


def plot_classification(rows: list[dict[str, Any]], out_png: Path) -> None:
    if plt is None or not rows:
        return
    rows = sorted(rows, key=lambda r: float(r.get('test_macro_f1', 0.0)), reverse=True)
    names = [str(r['model']) for r in rows]
    acc = [float(r['test_accuracy']) for r in rows]
    macro = [float(r['test_macro_f1']) for r in rows]
    params = [int(float(r['parameters'])) for r in rows]
    x = np.arange(len(names))
    w = 0.37
    fig, ax = plt.subplots(figsize=(max(12, 1.15 * len(names)), 6.8), dpi=170)
    ax.bar(x - w / 2, acc, w, label='test accuracy')
    ax.bar(x + w / 2, macro, w, label='test macro-F1')
    for i, p in enumerate(params):
        ax.text(i, max(acc[i], macro[i]) + 0.015, f'{p/1000:.0f}k', ha='center', va='bottom', fontsize=7)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel('score; higher is better')
    ax.set_title('Experiment 1: classification on the same train/test split')
    ax.text(
        0, -0.35,
        'All bars use the same split. Higher is better. Macro-F1 is included because class imbalance can make accuracy misleading. Numbers above bars are trainable parameters. WhaleNet-style models use fixed-rate Mel/WST branches; SFI models use native-rate physical-unit front-ends.',
        transform=ax.transAxes, fontsize=8, va='top', wrap=True,
    )
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def plot_fixed_rate(rows: list[dict[str, Any]], native_rows: list[dict[str, Any]], out_png: Path) -> None:
    if plt is None or not rows:
        return
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(str(r['model']), []).append(r)
    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=170)
    for model, rs in sorted(by_model.items()):
        rs = sorted(rs, key=lambda r: int(r['target_sample_rate']))
        xs = [int(r['target_sample_rate']) for r in rs]
        ys = [float(r['test_macro_f1']) for r in rs]
        ax.plot(xs, ys, marker='o', linewidth=2.1, label=model)
    for r in native_rows:
        model = str(r['model'])
        if model not in NATIVE_REFERENCE_MODELS:
            continue
        y = float(r.get('test_macro_f1', 0.0))
        ax.axhline(y, linestyle='--', linewidth=1.5, alpha=0.8, label=f'{model} native reference')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('fixed target sample rate used by preprocessing (Hz)')
    ax.set_ylabel('macro-F1; higher is better')
    ax.set_title('Experiment 2: fixed target-rate sensitivity')
    ax.text(
        0, -0.34,
        'Each curve retrains the same fixed-rate model after forcing all recordings to one target sample rate. Strong variation means the method depends on an arbitrary preprocessing choice. Dashed horizontal lines are native-rate SFI models from Experiment 1; they do not need a target sample-rate choice.',
        transform=ax.transAxes, fontsize=8, va='top', wrap=True,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def make_image_grid(image_paths: list[Path], titles: list[str], out_png: Path, title: str, caption: str) -> None:
    if plt is None:
        return
    imgs = []
    kept = []
    for p, t in zip(image_paths, titles):
        if p.exists():
            try:
                imgs.append(plt.imread(str(p)))
                kept.append(t)
            except Exception:
                pass
    if not imgs:
        return
    cols = min(3, len(imgs))
    rows = int(math.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.5 + 1.2), dpi=135)
    ax_arr = np.atleast_1d(axes).reshape(rows, cols)
    for ax in ax_arr.ravel():
        ax.axis('off')
    for ax, img, ttl in zip(ax_arr.ravel(), imgs, kept):
        ax.imshow(img)
        ax.set_title(ttl, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.text(0.02, 0.01, caption, fontsize=8, va='bottom', wrap=True)
    fig.tight_layout(rect=(0, 0.055, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def collect_model_dirs(run_dir: Path) -> list[Path]:
    out = []
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and (d / 'plots').exists():
            out.append(d)
    return out


def make_tsne_grids(run_dir: Path, out_dir: Path, prefix: str) -> None:
    models = collect_model_dirs(run_dir)
    configs = [
        ('class', 'latent_by_class.png', 'Color shows species. Better visual structure means same-species examples tend to form compact regions while different species are separated. This is qualitative; use the numeric metrics for claims.'),
        ('sample_rate', 'latent_by_sample_rate.png', 'Color shows native sample rate. If this dominates the geometry, the representation may be organizing recordings by recording conditions rather than species.'),
        ('correctness', 'latent_by_correctness.png', 'Color shows correct versus wrong predictions. Wrong predictions near class boundaries are expected; large isolated wrong clusters suggest systematic failure modes.'),
    ]
    for kind, fname, cap in configs:
        paths = [m / 'plots' / fname for m in models]
        titles = [m.name for m in models]
        make_image_grid(paths, titles, out_dir / f'{prefix}_tsne_by_{kind}.png', f't-SNE by {kind}', cap)


def best_fixed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        m = str(r['model'])
        if m not in best or float(r.get('test_macro_f1', 0.0)) > float(best[m].get('test_macro_f1', 0.0)):
            best[m] = r
    return [best[k] for k in sorted(best)]


def plot_final_overview(class_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]], out_png: Path) -> None:
    if plt is None or not class_rows:
        return
    rows = []
    for r in class_rows:
        rows.append({
            'name': str(r['model']),
            'macro_f1': float(r.get('test_macro_f1', 0.0)),
            'accuracy': float(r.get('test_accuracy', 0.0)),
            'kind': 'main',
        })
    for r in best_fixed_rows(fixed_rows):
        rows.append({
            'name': f"{r['model']} best fixed SR",
            'macro_f1': float(r.get('test_macro_f1', 0.0)),
            'accuracy': float(r.get('test_accuracy', 0.0)),
            'kind': 'best_fixed',
        })
    rows = sorted(rows, key=lambda r: r['macro_f1'], reverse=True)
    names = [r['name'] for r in rows]
    macro = [r['macro_f1'] for r in rows]
    acc = [r['accuracy'] for r in rows]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(12, 1.1 * len(names)), 7.0), dpi=170)
    ax.bar(x - w / 2, macro, w, label='macro-F1')
    ax.bar(x + w / 2, acc, w, label='accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylim(0, 1.04)
    ax.set_ylabel('score; higher is better')
    ax.set_title('Final overview: main models plus best fixed-rate configurations')
    ax.text(
        0, -0.37,
        'This figure merges the main classification comparison with the best fixed-target-rate result for each fixed-rate family. Higher is better. A fixed-rate method receiving its best target-rate result is a stronger baseline than comparing against only one arbitrary rate.',
        transform=ax.transAxes, fontsize=8, va='top', wrap=True,
    )
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def experiment_classification(args: argparse.Namespace, suite_dir: Path, cwd: Path) -> list[dict[str, Any]]:
    exp_dir = suite_dir / '01_classification_all_models'

    # First train all standard models through the common training script.
    cmd = train_base_cmd(args) + [
        '--run-dir', str(exp_dir),
        '--models', args.classification_models,
        '--baseline-sr', str(args.reference_sr),
        '--whalenet-sr', str(args.reference_sr),
    ]
    run_cmd(cmd, cwd, skip_if=exp_dir / 'experiment_results.csv', skip_existing=args.skip_existing)
    rows = read_csv(exp_dir / 'experiment_results.csv')

    # Add the paper-reported WhaleNet baseline to the SAME experiment folder.
    # This is staged: Mel, WST1 and WST2 branches are trained separately; WST1+WST2
    # are fused by an MLP; the paper configuration then uses the hard Mel+WST
    # convex merge. By default only the final paper-used row is added to the
    # main summary. Intermediate branch rows are internal diagnostics unless
    # --include-whalenet-branch-ablation is passed.
    exact_dir = exp_dir / 'WHALE_NET_PAPER_BASELINE'
    exact_csv = exact_dir / 'whalenet_paper_results_for_main_summary.csv'
    if not (args.skip_existing and exact_csv.exists()):
        exact_rows = run_whalenet_paper_staged(
            train_manifest=exp_dir / 'manifests' / 'train_samples.csv',
            test_manifest=exp_dir / 'manifests' / 'test_samples.csv',
            classes_csv=exp_dir / 'manifests' / 'classes.csv',
            out_dir=exact_dir,
            sample_rate=int(args.whalenet_exact_sr),
            signal_len=int(args.whalenet_signal_len),
            J=int(args.whalenet_exact_j),
            Q=int(args.whalenet_exact_q),
            hard_lambda=float(args.whalenet_hard_lambda),
            branch_epochs=int(args.whalenet_branch_epochs),
            fusion_epochs=int(args.whalenet_fusion_epochs),
            batch_size=int(args.whalenet_batch_size),
            feature_batch_size=int(args.whalenet_feature_batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
            seed=int(args.seed),
            standardize_signal=bool(args.whalenet_standardize_signal),
            include_branch_ablation=bool(args.include_whalenet_branch_ablation),
        )
    else:
        exact_rows = read_csv(exact_csv)

    # Copy reported WhaleNet model folders into the top classification folder so the t-SNE grid sees all main rows together.
    for r in exact_rows:
        name = str(r['model'])
        src = exact_dir / name
        dst = exp_dir / name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
    rows = rows + exact_rows

    fields = ['model','parameters','best_epoch','test_loss','test_accuracy','test_macro_f1','test_weighted_f1','test_micro_f1','checkpoint']
    write_csv(exp_dir / 'summary_classification.csv', rows, fields)
    write_csv(exp_dir / 'experiment_results_with_whalenet_paper.csv', rows, fields)
    plot_classification(rows, exp_dir / 'summary_classification.png')
    make_tsne_grids(exp_dir, exp_dir, 'classification')
    write_caption(
        exp_dir / 'README_EXPERIMENT.md',
        'Experiment 1: classification, all models in the same split, including paper-reported staged WhaleNet baseline',
        'This trains native SFI models and fixed-rate Mel baselines through the shared trainer, then adds an paper-reported staged WhaleNet baseline in the same folder and summary. The WhaleNet baseline uses Kymatio Scattering1D, separate Mel/WST1/WST2 ResNet branches, WST1+WST2 MLP fusion, and the paper-reported hard Mel+WST merge. Only the final paper-used row is shown by default.',
        'Higher accuracy and macro-F1 are better. Macro-F1 is the safer metric when classes are imbalanced. The exact staged WhaleNet rows are directly comparable here because they use the same manifest split and the same fixed target sample rate.',
        'The paper-reported WhaleNet baseline intentionally avoids the original audio backend but uses Kymatio for scattering. Mel is computed with a local torch STFT + Mel filterbank equivalent. t-SNE grids include all standard models plus the final WhaleNet paper baseline row.',
    )
    return rows


def experiment_fixed_rate(args: argparse.Namespace, suite_dir: Path, cwd: Path, rates: list[int], native_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exp_dir = suite_dir / '02_fixed_target_rate_sensitivity'
    all_rows: list[dict[str, Any]] = []
    for sr in rates:
        run_dir = exp_dir / f'target_sr_{sr}'
        cmd = train_base_cmd(args) + [
            '--run-dir', str(run_dir),
            '--models', args.fixed_rate_models,
            '--baseline-sr', str(sr),
            '--whalenet-sr', str(sr),
        ]
        run_cmd(cmd, cwd, skip_if=run_dir / 'experiment_results.csv', skip_existing=args.skip_existing)
        for r in read_csv(run_dir / 'experiment_results.csv'):
            row = dict(r)
            row['target_sample_rate'] = int(sr)
            row['effective_nyquist_hz'] = float(sr) / 2.0
            all_rows.append(row)
        make_tsne_grids(run_dir, run_dir, f'target_sr_{sr}')
    fields = ['model','target_sample_rate','effective_nyquist_hz','parameters','best_epoch','test_loss','test_accuracy','test_macro_f1','test_weighted_f1','test_micro_f1','checkpoint']
    write_csv(exp_dir / 'summary_fixed_target_rate_sensitivity.csv', all_rows, fields)
    write_csv(exp_dir / 'summary_best_fixed_rate_per_model.csv', best_fixed_rows(all_rows), fields)
    plot_fixed_rate(all_rows, native_rows, exp_dir / 'summary_fixed_target_rate_sensitivity.png')
    write_caption(
        exp_dir / 'README_EXPERIMENT.md',
        'Experiment 2: fixed target-rate sensitivity, for fixed-rate Mel baselines',
        'This retrains the fixed-rate Mel baselines at four target sample rates. The four rates are derived from the maximum sample rate in the dataset: max_sr/8, max_sr/4, max_sr/2, and max_sr. WhaleNet paper baseline is kept in Experiment 1 at its reported fixed rate, rather than expanded into many variants.',
        'Higher macro-F1 is better. A model that changes strongly with target sample rate is sensitive to a preprocessing hyperparameter. Dashed horizontal lines show native-rate SFI models from Experiment 1, which do not require this target-rate choice.',
        f'Target sample rates used: {rates}.',
    )
    return all_rows


def make_final_summary(suite_dir: Path, class_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]]) -> None:
    final_dir = suite_dir / '00_FINAL_SUMMARY'
    final_dir.mkdir(parents=True, exist_ok=True)
    class_fields = ['model','parameters','best_epoch','test_loss','test_accuracy','test_macro_f1','test_weighted_f1','test_micro_f1','checkpoint']
    fixed_fields = ['model','target_sample_rate','effective_nyquist_hz','parameters','best_epoch','test_loss','test_accuracy','test_macro_f1','test_weighted_f1','test_micro_f1','checkpoint']
    write_csv(final_dir / 'classification_all_models.csv', class_rows, class_fields)
    write_csv(final_dir / 'best_fixed_rate_per_model.csv', best_fixed_rows(fixed_rows), fixed_fields)
    plot_classification(class_rows, final_dir / 'classification_all_models.png')
    if fixed_rows:
        plot_final_overview(class_rows, fixed_rows, final_dir / 'overall_summary_main_plus_best_fixed_rate.png')
    write_caption(
        final_dir / 'README_FINAL_SUMMARY.md',
        'Final summary',
        'This folder collects the key CSVs and plots from the whole single-script run. The classification summary compares all models on one split. The fixed-rate summary reports the best target-rate run for each fixed-rate family.',
        'Higher accuracy and macro-F1 are better. The best fixed-rate result is the fairer baseline for fixed-rate methods because it avoids comparing SFI against a poorly chosen single target rate.',
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Single-script paper protocol with SFI, Mel baselines, and the paper-reported staged WhaleNet baseline.')
    p.add_argument('--root', default="/home/mmlab/Desktop/nfo_bioacoustics/watkins_marine")
    p.add_argument('--suite-dir', default='./runs/single_script_whalenet_protocol')
    p.add_argument('--classification-models', default=ALL_CLASSIFICATION_MODELS)
    p.add_argument('--fixed-rate-models', default=DEFAULT_FIXED_RATE_MODELS)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--val-fraction', type=float, default=0.0)
    p.add_argument('--test-fraction', type=float, default=0.25)
    p.add_argument('--clip-seconds', type=float, default=5.0)
    p.add_argument('--normalize', default='standardize', choices=['none','peak','rms','standardize'])
    p.add_argument('--pad-mode', default='zero', choices=['zero','repeat'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--plot-max-points', type=int, default=0, help='0 means all test points for t-SNE. Use e.g. 2000 for faster plotting.')
    p.add_argument('--reference-sr', type=int, default=47600, help='Fixed-rate reference used for standard Mel baselines in Experiment 1. Default 47600 to match the WhaleNet paper protocol.')
    p.add_argument('--target-rates', default='', help='Comma-separated fixed target rates. Empty = max_sr/8,max_sr/4,max_sr/2,max_sr.')
    p.add_argument('--audio-extensions', default='.wav,.flac,.aif,.aiff')
    p.add_argument('--min-samples-per-class', type=int, default=0)
    p.add_argument('--remove-duplicates', action='store_true')
    p.add_argument('--baseline-channels', type=int, default=16)
    p.add_argument('--embedding-dim', type=int, default=192)
    p.add_argument('--whalenet-j', type=int, default=6, help='Used only if you manually include the old joint whalenet_* models.')
    p.add_argument('--whalenet-q', type=int, default=16, help='Used only if you manually include the old joint whalenet_* models.')
    p.add_argument('--whalenet-signal-len', type=int, default=8000)
    p.add_argument('--whalenet-exact-j', type=int, default=6, help='Kymatio J for the paper-reported WhaleNet baseline. Default 6 follows the paper-reported optimal setting.')
    p.add_argument('--whalenet-exact-q', type=int, default=16, help='Kymatio Q for the paper-reported WhaleNet baseline. Default 16 follows the paper-reported optimal setting.')
    p.add_argument('--whalenet-branch-epochs', type=int, default=100)
    p.add_argument('--whalenet-fusion-epochs', type=int, default=500)
    p.add_argument('--whalenet-batch-size', type=int, default=128, help='Batch size for WhaleNet branches/fusions. Default 128 follows the paper-reported optimal setting.')
    p.add_argument('--whalenet-feature-batch-size', type=int, default=256)
    p.add_argument('--no-whalenet-standardize-signal', dest='whalenet_standardize_signal', action='store_false', help='Disable paper-text waveform standardization for WhaleNet. Default is standardization ON.')
    p.set_defaults(whalenet_standardize_signal=True)
    p.add_argument('--whalenet-exact-sr', type=int, default=47600, help='Fixed sample rate for the paper-reported WhaleNet baseline. Default 47600.')
    p.add_argument('--whalenet-hard-lambda', type=float, default=1.0/3.0, help='Hard merge Mel weight lambda in lambda*Mel + (1-lambda)*WST12. Default 1/3, matching the paper-reported hard-merge setting.')
    p.add_argument('--include-whalenet-branch-ablation', action='store_true', help='Also put internal WhaleNet branch rows in the main summary. Off by default to show only the final paper-used WhaleNet model.')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve()
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)

    max_sr, sr_counts = dataset_sample_rates(root, args.audio_extensions)
    rates = parse_rate_list(args.target_rates) or four_rates_from_dataset(max_sr)
    if args.reference_sr <= 0:
        args.reference_sr = rates[1] if len(rates) > 1 else rates[0]

    protocol = {
        'run_this_script': 'run_all_experiments_with_whalenet.py',
        'root': str(root),
        'suite_dir': str(suite_dir),
        'dataset_max_sample_rate': int(max_sr),
        'dataset_sample_rate_counts': {str(k): int(v) for k, v in sr_counts.items()},
        'target_rates': rates,
        'reference_sr': int(args.reference_sr),
        'classification_models': parse_model_list(args.classification_models),
        'fixed_rate_models': parse_model_list(args.fixed_rate_models),
        'whalenet_paper_baseline_parameters': {
            'fixed_sample_rate': int(args.whalenet_exact_sr),
            'signal_len': int(args.whalenet_signal_len),
            'J': int(args.whalenet_exact_j),
            'Q': int(args.whalenet_exact_q),
            'branch_epochs': int(args.whalenet_branch_epochs),
            'fusion_epochs': int(args.whalenet_fusion_epochs),
            'batch_size': int(args.whalenet_batch_size),
            'uses_kymatio_scattering1d': True,
            'standardize_signal': bool(args.whalenet_standardize_signal),
            'hard_lambda_mel_weight': float(args.whalenet_hard_lambda),
            'include_branch_ablation': bool(args.include_whalenet_branch_ablation),
        },
    }
    save_json(protocol, suite_dir / 'PROTOCOL_SUMMARY.json')
    (suite_dir / 'README_RUN_ONLY_THIS.md').write_text(
        '# Run only this script\n\n'
        'Use `run_all_experiments_with_whalenet.py`. Do not run `run_whalenet_comparison.py`, `run_full_ablation.py`, or individual training scripts unless debugging.\n\n'
        'Outputs:\n\n'
        '- `01_classification_all_models/`: all models, for fixed-rate Mel baselines, trained on the same split.\n'
        '- `02_fixed_target_rate_sensitivity/`: fixed-rate models retrained at four target sample rates, with SFI native references in the plot.\n'
        '- `00_FINAL_SUMMARY/`: compact CSVs and final overview plots.\n',
        encoding='utf-8',
    )

    class_rows = experiment_classification(args, suite_dir, cwd)
    fixed_rows = experiment_fixed_rate(args, suite_dir, cwd, rates, class_rows)
    make_final_summary(suite_dir, class_rows, fixed_rows)

    print('\nDONE. Inspect these first:', flush=True)
    print(suite_dir / '00_FINAL_SUMMARY' / 'classification_all_models.csv', flush=True)
    print(suite_dir / '00_FINAL_SUMMARY' / 'classification_all_models.png', flush=True)
    print(suite_dir / '00_FINAL_SUMMARY' / 'best_fixed_rate_per_model.csv', flush=True)
    print(suite_dir / '00_FINAL_SUMMARY' / 'overall_summary_main_plus_best_fixed_rate.png', flush=True)
    print(suite_dir / '01_classification_all_models' / 'classification_tsne_by_class.png', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
