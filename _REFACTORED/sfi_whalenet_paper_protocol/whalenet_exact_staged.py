from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from train_sfi_bioacoustics import read_csv, load_audio_mono
from sfi_bioacoustic_models import fft_resample_1d, make_mel_filterbank


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')


def center_crop_or_pad_1d(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    wav = wav.float()
    n = int(wav.numel())
    length = int(length)
    if n == length:
        return wav
    if n < length:
        left = (length - n) // 2
        right = length - n - left
        return F.pad(wav, (left, right))
    start = (n - length) // 2
    return wav[start:start + length]


def standardize_1d(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-5)


class ManifestAudioDataset(Dataset):
    def __init__(self, manifest_csv: Path) -> None:
        self.rows = read_csv(manifest_csv)
        if not self.rows:
            raise RuntimeError(f'Empty manifest: {manifest_csv}')

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        wav, sr = load_audio_mono(row['audio_path'])
        return {
            'audio': wav,
            'sample_rate': int(sr),
            'label': int(row['label']),
            'species': row.get('species', str(row['label'])),
            'path': row['audio_path'],
        }


def collate_raw(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'audio': [b['audio'] for b in batch],
        'sample_rate': [int(b['sample_rate']) for b in batch],
        'label': torch.tensor([int(b['label']) for b in batch], dtype=torch.long),
        'species': [b['species'] for b in batch],
        'path': [b['path'] for b in batch],
    }


class ExactWhaleNetPreprocessor(nn.Module):
    """Fixed-rate resampling + centered 8000-sample crop/pad.

    This mirrors the user-provided WhaleNet code at preprocessing level, but uses
    the local FFT resampler instead of the original audio backend. Set standardize_signal=True to
    follow the paper text; set it False to follow the pasted code literally after
    its `X = standardize(XC); X = XC` overwrite.
    """

    def __init__(self, sample_rate: int, signal_len: int, standardize_signal: bool = False) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.signal_len = int(signal_len)
        self.standardize_signal = bool(standardize_signal)

    def forward(self, audio: list[torch.Tensor], sample_rate: list[int]) -> torch.Tensor:
        out = []
        device = audio[0].device if audio else torch.device('cpu')
        for wav, sr in zip(audio, sample_rate):
            w = wav.to(device).float()
            if w.dim() == 1:
                w = w.unsqueeze(0)
            if w.shape[0] > 1:
                w = w.mean(dim=0, keepdim=True)
            w = fft_resample_1d(w, int(sr), self.sample_rate).squeeze(0)
            w = center_crop_or_pad_1d(w, self.signal_len)
            if self.standardize_signal:
                w = standardize_1d(w)
            out.append(w)
        return torch.stack(out, dim=0)


class ExactMelComputer(nn.Module):
    """External-audio-backend-free approximation of the original MelSpectrogram transform(normalized=True, n_mels=64)."""

    def __init__(self, sample_rate: int, signal_len: int, n_mels: int = 64, n_fft: int = 400, hop_length: int = 200, standardize_signal: bool = False) -> None:
        super().__init__()
        self.pre = ExactWhaleNetPreprocessor(sample_rate, signal_len, standardize_signal=standardize_signal)
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer('window', torch.hann_window(self.n_fft), persistent=False)

    def forward(self, audio: list[torch.Tensor], sample_rate: list[int]) -> torch.Tensor:
        x = self.pre(audio, sample_rate)
        if x.shape[-1] < self.n_fft:
            x = F.pad(x, (0, self.n_fft - x.shape[-1]))
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=x.device, dtype=x.dtype),
            center=True,
            normalized=True,
            return_complex=True,
        ).abs().pow(2.0)
        fb = make_mel_filterbank(self.n_fft, self.sample_rate, self.n_mels, 0.0, self.sample_rate * 0.5, x.device, x.dtype)
        mel = torch.einsum('mf,bft->bmt', fb, spec)
        return mel.unsqueeze(1)


class ExactWSTComputer(nn.Module):
    """Kymatio Scattering1D feature computer matching the pasted WhaleNet code."""

    def __init__(self, sample_rate: int, signal_len: int, J: int, Q: int, order: int, standardize_signal: bool = False, median_norm: bool = True) -> None:
        super().__init__()
        self.pre = ExactWhaleNetPreprocessor(sample_rate, signal_len, standardize_signal=standardize_signal)
        self.sample_rate = int(sample_rate)
        self.signal_len = int(signal_len)
        self.J = int(J)
        self.Q = int(Q)
        self.order = int(order)
        self.median_norm = bool(median_norm)
        try:
            from kymatio.torch import Scattering1D  # type: ignore
        except Exception as exc:
            raise ImportError('Exact WhaleNet WST baseline requires kymatio: pip install kymatio') from exc
        self.scattering = Scattering1D(J=self.J, shape=self.signal_len, Q=self.Q)
        meta = self.scattering.meta()
        idx = np.where(meta['order'] == self.order)[0]
        if len(idx) == 0:
            raise RuntimeError(f'No WST coefficients for order {self.order}')
        self.register_buffer('order_idx', torch.as_tensor(idx, dtype=torch.long), persistent=False)

    def forward(self, audio: list[torch.Tensor], sample_rate: list[int]) -> torch.Tensor:
        x = self.pre(audio, sample_rate)
        sx = self.scattering(x)
        sx = sx.index_select(1, self.order_idx.to(sx.device))
        if self.median_norm:
            flat = sx.flatten(1)
            med = flat.median(dim=1, keepdim=True).values.view(-1, 1, 1)
            std = flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).view(-1, 1, 1)
            sx = (sx - med) / std
        return sx.unsqueeze(1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
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


class ExactResNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, layers: Sequence[int] = (2, 2, 2)) -> None:
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, int(layers[0]), stride=1)
        self.layer2 = self.make_layer(32, int(layers[1]), stride=2)
        self.layer3 = self.make_layer(64, int(layers[2]), stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, int(num_classes))

    def make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers: list[nn.Module] = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.avg_pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class ExactMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_dim), 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, int(num_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear3(self.features(x))


def class_count_from_manifest(classes_csv: Path) -> tuple[int, list[str]]:
    rows = read_csv(classes_csv)
    rows = sorted(rows, key=lambda r: int(r['class_index']))
    return len(rows), [r['class_name'] for r in rows]


@torch.no_grad()
def extract_features(frontend: nn.Module, manifest_csv: Path, batch_size: int, num_workers: int, device: torch.device, desc: str) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str], list[str]]:
    ds = ManifestAudioDataset(manifest_csv)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_raw)
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    srs: list[int] = []
    paths: list[str] = []
    species: list[str] = []
    frontend.eval().to(device)
    for i, batch in enumerate(loader):
        audio = [x.to(device) for x in batch['audio']]
        sr = [int(x) for x in batch['sample_rate']]
        x = frontend(audio, sr)
        feats.append(x.cpu())
        labels.append(batch['label'].cpu())
        srs.extend(sr)
        paths.extend([str(p) for p in batch['path']])
        species.extend([str(s) for s in batch['species']])
        if (i + 1) % 10 == 0:
            print(f'{desc}: batch {i+1}/{len(loader)}', flush=True)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0), srs, paths, species


def train_classifier(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor, epochs: int, batch_size: int, device: torch.device, lr: float, weight_decay: float, adamw: bool = True) -> dict[str, Any]:
    model = model.to(device)
    ds = TensorDataset(train_x, train_y.long())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    if adamw:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    hist: list[dict[str, float]] = []
    for ep in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        correct = 0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * yb.numel()
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            count += int(yb.numel())
        if ep == 1 or ep == epochs or ep % max(1, epochs // 5) == 0:
            print(f'  epoch {ep:04d}/{epochs} loss={total/max(count,1):.4f} acc={correct/max(count,1):.4f}', flush=True)
        hist.append({'epoch': ep, 'train_loss': total / max(count, 1), 'train_accuracy': correct / max(count, 1)})
    eval_out = evaluate_tensor_model(model, test_x, test_y, batch_size, device)
    return {'model': model, 'history': hist, **eval_out}


@torch.no_grad()
def evaluate_tensor_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, Any]:
    model.eval().to(device)
    logits_list: list[torch.Tensor] = []
    feat_list: list[torch.Tensor] = []
    loader = DataLoader(TensorDataset(x, y.long()), batch_size=batch_size, shuffle=False)
    for xb, _ in loader:
        xb = xb.to(device)
        if hasattr(model, 'features'):
            feat = model.features(xb)
        else:
            feat = None
        logits = model(xb)
        logits_list.append(logits.detach().cpu())
        if feat is not None:
            feat_list.append(feat.detach().cpu())
    logits = torch.cat(logits_list, dim=0)
    emb = torch.cat(feat_list, dim=0) if feat_list else logits
    metrics = metrics_from_logits(logits, y.long(), int(logits.shape[-1]))
    return {'logits': logits, 'embedding': emb, 'metrics': metrics}


def metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, classes: int) -> dict[str, float]:
    pred = logits.argmax(dim=-1).cpu()
    y = target.cpu().long()
    acc = float((pred == y).float().mean().item()) if y.numel() else 0.0
    f1s = []
    supports = []
    for c in range(classes):
        pc = pred == c
        tc = y == c
        tp = float((pc & tc).sum().item())
        fp = float((pc & ~tc).sum().item())
        fn = float((~pc & tc).sum().item())
        supp = float(tc.sum().item())
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        if supp > 0:
            f1s.append(f1)
            supports.append(supp)
    macro = float(np.mean(f1s)) if f1s else 0.0
    weighted = float(np.average(f1s, weights=supports)) if f1s else 0.0
    return {'accuracy': acc, 'macro_f1': macro, 'weighted_f1': weighted, 'micro_f1': acc, 'loss': float(F.cross_entropy(logits, y.long()).item())}


def tsne_coords(emb: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    if emb.shape[0] < 4:
        return pca_coords(emb), 'pca'
    try:
        from sklearn.manifold import TSNE  # type: ignore
        x = emb.astype(np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        x = x / np.maximum(x.std(axis=0, keepdims=True), 1e-8)
        perplexity = min(30.0, max(2.0, float((x.shape[0] - 1) // 3)))
        return TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=int(seed)).fit_transform(x), 'tsne'
    except Exception:
        return pca_coords(emb), 'pca'


def pca_coords(emb: np.ndarray) -> np.ndarray:
    x = emb.astype(np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.empty((0, 2))
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] == 1:
        return np.zeros((1, 2))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    c = x @ vt[:2].T
    if c.shape[1] == 1:
        c = np.concatenate([c, np.zeros((c.shape[0], 1))], axis=1)
    return c[:, :2]


def plot_tsne(embedding: torch.Tensor, labels: torch.Tensor, sample_rates: list[int], preds: torch.Tensor, class_names: list[str], model_dir: Path, seed: int) -> None:
    if plt is None:
        return
    plot_dir = model_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    emb = embedding.detach().cpu().float().numpy()
    coords, reducer = tsne_coords(emb, seed)
    y = labels.cpu().numpy().astype(int)
    pred = preds.cpu().numpy().astype(int)
    sr = np.asarray(sample_rates, dtype=int)
    correct = (y == pred).astype(int)
    np.savez_compressed(plot_dir / 'feature_dump.npz', embedding=emb, coord=coords, label=y, pred=pred, sample_rate=sr, reducer=np.asarray([reducer], dtype=object))
    # CSV projection
    rows = []
    for i in range(len(y)):
        rows.append({'sample_index': i, 'reducer': reducer, 'projection_x': float(coords[i,0]), 'projection_y': float(coords[i,1]), 'label': int(y[i]), 'class_name': class_names[int(y[i])] if 0 <= int(y[i]) < len(class_names) else str(y[i]), 'prediction': int(pred[i]), 'predicted_class_name': class_names[int(pred[i])] if 0 <= int(pred[i]) < len(class_names) else str(pred[i]), 'correct': int(correct[i]), 'sample_rate': int(sr[i])})
    write_csv(plot_dir / 'feature_projection.csv', rows, ['sample_index','reducer','projection_x','projection_y','label','class_name','prediction','predicted_class_name','correct','sample_rate'])
    _scatter_class(coords, y, class_names, f'{model_dir.name}: {reducer} by class', plot_dir / 'latent_by_class.png')
    _scatter_values(coords, sr, f'{model_dir.name}: {reducer} by native sample rate', plot_dir / 'latent_by_sample_rate.png')
    _scatter_values(coords, correct, f'{model_dir.name}: {reducer} by correctness', plot_dir / 'latent_by_correctness.png', labels={0:'wrong',1:'correct'})


def _scatter_class(coords: np.ndarray, y: np.ndarray, class_names: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    cmap = plt.get_cmap('tab20')
    for c, name in enumerate(class_names):
        mask = y == c
        if mask.any():
            ax.scatter(coords[mask,0], coords[mask,1], s=14, alpha=0.75, color=cmap(c % 20), label=name.replace('_',' '), linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=5, loc='center left', bbox_to_anchor=(1.02,0.5), frameon=False, ncol=2 if len(class_names)>20 else 1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def _scatter_values(coords: np.ndarray, values: np.ndarray, title: str, path: Path, labels: dict[int,str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    unique = sorted(set(int(v) for v in values.tolist()))
    cmap = plt.get_cmap('tab20')
    for i, v in enumerate(unique):
        mask = values == v
        lab = labels.get(v, str(v)) if labels else (f'{v} Hz' if max(unique) > 100 else str(v))
        ax.scatter(coords[mask,0], coords[mask,1], s=14, alpha=0.75, color=cmap(i % 20), label=lab, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc='best' if len(unique)<=2 else 'center left', bbox_to_anchor=None if len(unique)<=2 else (1.02,0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_exact_whalenet_staged(
    train_manifest: Path,
    test_manifest: Path,
    classes_csv: Path,
    out_dir: Path,
    sample_rate: int,
    signal_len: int,
    J: int,
    Q: int,
    branch_epochs: int,
    fusion_epochs: int,
    batch_size: int,
    feature_batch_size: int,
    num_workers: int,
    device: str,
    seed: int,
    standardize_signal: bool = False,
) -> list[dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes, class_names = class_count_from_manifest(classes_csv)
    config = {
        'baseline': 'whalenet_exact_staged',
        'uses_kymatio_scattering1d': True,
        'sample_rate': int(sample_rate),
        'signal_len': int(signal_len),
        'J': int(J),
        'Q': int(Q),
        'standardize_signal': bool(standardize_signal),
        'branch_epochs': int(branch_epochs),
        'fusion_epochs': int(fusion_epochs),
        'batch_size': int(batch_size),
        'feature_batch_size': int(feature_batch_size),
        'note': 'This is staged like the pasted WhaleNet code: train Mel, WST1, WST2 ResNets; train WST1+WST2 MLP on branch logits; fuse WST and Mel logits by max, hard lambda, and MLP.',
    }
    save_json(config, out_dir / 'exact_whalenet_config.json')

    print('\n=== exact staged WhaleNet baseline: feature extraction ===', flush=True)
    mel_front = ExactMelComputer(sample_rate, signal_len, 64, 400, 200, standardize_signal=standardize_signal).to(dev)
    wst1_front = ExactWSTComputer(sample_rate, signal_len, J, Q, 1, standardize_signal=standardize_signal).to(dev)
    wst2_front = ExactWSTComputer(sample_rate, signal_len, J, Q, 2, standardize_signal=standardize_signal).to(dev)

    x_mel_tr, y_tr, sr_tr, _, _ = extract_features(mel_front, train_manifest, feature_batch_size, num_workers, dev, 'extract mel train')
    x_mel_te, y_te, sr_te, _, _ = extract_features(mel_front, test_manifest, feature_batch_size, num_workers, dev, 'extract mel test')
    x_w1_tr, _, _, _, _ = extract_features(wst1_front, train_manifest, feature_batch_size, num_workers, dev, 'extract wst1 train')
    x_w1_te, _, _, _, _ = extract_features(wst1_front, test_manifest, feature_batch_size, num_workers, dev, 'extract wst1 test')
    x_w2_tr, _, _, _, _ = extract_features(wst2_front, train_manifest, feature_batch_size, num_workers, dev, 'extract wst2 train')
    x_w2_te, _, _, _, _ = extract_features(wst2_front, test_manifest, feature_batch_size, num_workers, dev, 'extract wst2 test')

    rows: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}

    def train_branch(name: str, xtr: torch.Tensor, xte: torch.Tensor) -> nn.Module:
        print(f'\n=== exact staged WhaleNet: train {name} ResNet branch ===', flush=True)
        model = ExactResNet(num_classes)
        res = train_classifier(model, xtr, y_tr, xte, y_te, branch_epochs, batch_size, dev, lr=0.01, weight_decay=0.001, adamw=True)
        trained = res['model']
        eval_train = evaluate_tensor_model(trained, xtr, y_tr, batch_size, dev)
        outputs[name] = {'test': res, 'train_eval': eval_train, 'params': count_params(trained), 'model': trained}
        save_model_result(name, outputs[name], y_te, sr_te, class_names, out_dir, seed)
        rows.append(make_row(name, outputs[name], out_dir / name / 'best.pt'))
        torch.save({'model': trained.state_dict(), 'config': config, 'metrics': res['metrics']}, out_dir / name / 'best.pt')
        return trained

    mel_model = train_branch('whalenet_exact_mel', x_mel_tr, x_mel_te)
    w1_model = train_branch('whalenet_exact_wst1', x_w1_tr, x_w1_te)
    w2_model = train_branch('whalenet_exact_wst2', x_w2_tr, x_w2_te)

    print('\n=== exact staged WhaleNet: train WST1+WST2 MLP ===', flush=True)
    wst_tr = torch.cat([outputs['whalenet_exact_wst1']['train_eval']['logits'], outputs['whalenet_exact_wst2']['train_eval']['logits']], dim=1)
    wst_te = torch.cat([outputs['whalenet_exact_wst1']['test']['logits'], outputs['whalenet_exact_wst2']['test']['logits']], dim=1)
    wst12_mlp = ExactMLP(2 * num_classes, num_classes)
    res_wst12 = train_classifier(wst12_mlp, wst_tr, y_tr, wst_te, y_te, fusion_epochs, batch_size, dev, lr=0.001, weight_decay=0.0, adamw=False)
    eval_wst12_tr = evaluate_tensor_model(res_wst12['model'], wst_tr, y_tr, batch_size, dev)
    outputs['whalenet_exact_wst12_mlp'] = {'test': res_wst12, 'train_eval': eval_wst12_tr, 'params': count_params(res_wst12['model']), 'model': res_wst12['model']}
    save_model_result('whalenet_exact_wst12_mlp', outputs['whalenet_exact_wst12_mlp'], y_te, sr_te, class_names, out_dir, seed)
    rows.append(make_row('whalenet_exact_wst12_mlp', outputs['whalenet_exact_wst12_mlp'], out_dir / 'whalenet_exact_wst12_mlp' / 'best.pt'))
    torch.save({'model': res_wst12['model'].state_dict(), 'config': config, 'metrics': res_wst12['metrics']}, out_dir / 'whalenet_exact_wst12_mlp' / 'best.pt')

    print('\n=== exact staged WhaleNet: final Mel+WST fusion ===', flush=True)
    star_tr = eval_wst12_tr['logits']
    star_te = res_wst12['logits']
    mel_tr = outputs['whalenet_exact_mel']['train_eval']['logits']
    mel_te = outputs['whalenet_exact_mel']['test']['logits']

    # Element-wise max merge, as in pasted code.
    max_logits_te = torch.stack([star_te, mel_te], dim=1).max(dim=1).values
    max_metrics = metrics_from_logits(max_logits_te, y_te, num_classes)
    max_emb = torch.cat([star_te, mel_te], dim=1)
    outputs['whalenet_exact_full_max'] = {'test': {'logits': max_logits_te, 'embedding': max_emb, 'metrics': max_metrics}, 'params': outputs['whalenet_exact_wst12_mlp']['params'] + outputs['whalenet_exact_mel']['params']}
    save_model_result('whalenet_exact_full_max', outputs['whalenet_exact_full_max'], y_te, sr_te, class_names, out_dir, seed)
    rows.append(make_row('whalenet_exact_full_max', outputs['whalenet_exact_full_max'], out_dir / 'whalenet_exact_full_max' / 'best.pt'))
    torch.save({'config': config, 'metrics': max_metrics}, out_dir / 'whalenet_exact_full_max' / 'best.pt')

    # Hard convex combination. Choose lambda by train accuracy, not test, to avoid test tuning.
    best_lam = 0.0
    best_train_acc = -1.0
    for lam in np.linspace(0.0, 1.0, 31):
        train_logits = float(lam) * star_tr + (1.0 - float(lam)) * mel_tr
        acc = float((train_logits.argmax(dim=1) == y_tr).float().mean().item())
        if acc > best_train_acc:
            best_train_acc = acc
            best_lam = float(lam)
    hard_logits_te = best_lam * star_te + (1.0 - best_lam) * mel_te
    hard_metrics = metrics_from_logits(hard_logits_te, y_te, num_classes)
    hard_emb = torch.cat([star_te, mel_te], dim=1)
    outputs['whalenet_exact_full_hard'] = {'test': {'logits': hard_logits_te, 'embedding': hard_emb, 'metrics': hard_metrics}, 'params': outputs['whalenet_exact_wst12_mlp']['params'] + outputs['whalenet_exact_mel']['params'], 'lambda': best_lam}
    save_model_result('whalenet_exact_full_hard', outputs['whalenet_exact_full_hard'], y_te, sr_te, class_names, out_dir, seed)
    rows.append(make_row('whalenet_exact_full_hard', outputs['whalenet_exact_full_hard'], out_dir / 'whalenet_exact_full_hard' / 'best.pt'))
    torch.save({'config': config, 'metrics': hard_metrics, 'lambda': best_lam}, out_dir / 'whalenet_exact_full_hard' / 'best.pt')

    # Final MLP over WST12 logits + Mel logits.
    full_tr = torch.cat([star_tr, mel_tr], dim=1)
    full_te = torch.cat([star_te, mel_te], dim=1)
    full_mlp = ExactMLP(2 * num_classes, num_classes)
    res_full = train_classifier(full_mlp, full_tr, y_tr, full_te, y_te, fusion_epochs, batch_size, dev, lr=0.001, weight_decay=0.0, adamw=False)
    outputs['whalenet_exact_full_mlp'] = {'test': res_full, 'params': count_params(res_full['model']), 'model': res_full['model']}
    save_model_result('whalenet_exact_full_mlp', outputs['whalenet_exact_full_mlp'], y_te, sr_te, class_names, out_dir, seed)
    rows.append(make_row('whalenet_exact_full_mlp', outputs['whalenet_exact_full_mlp'], out_dir / 'whalenet_exact_full_mlp' / 'best.pt'))
    torch.save({'model': res_full['model'].state_dict(), 'config': config, 'metrics': res_full['metrics']}, out_dir / 'whalenet_exact_full_mlp' / 'best.pt')

    fields = ['model','parameters','best_epoch','test_loss','test_accuracy','test_macro_f1','test_weighted_f1','test_micro_f1','checkpoint']
    write_csv(out_dir / 'exact_whalenet_results.csv', rows, fields)
    return rows


def make_row(name: str, data: dict[str, Any], ckpt: Path) -> dict[str, Any]:
    metrics = data['test']['metrics'] if 'metrics' not in data.get('test', {}) else data['test']['metrics']
    return {
        'model': name,
        'parameters': int(data.get('params', 0)),
        'best_epoch': 'final',
        'test_loss': metrics.get('loss', 0.0),
        'test_accuracy': metrics.get('accuracy', 0.0),
        'test_macro_f1': metrics.get('macro_f1', 0.0),
        'test_weighted_f1': metrics.get('weighted_f1', 0.0),
        'test_micro_f1': metrics.get('micro_f1', 0.0),
        'checkpoint': str(ckpt),
    }


def save_model_result(name: str, data: dict[str, Any], labels: torch.Tensor, sample_rates: list[int], class_names: list[str], out_root: Path, seed: int) -> None:
    model_dir = out_root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    test = data['test']
    logits = test['logits']
    emb = test['embedding']
    preds = logits.argmax(dim=1)
    save_json({'model': name, 'parameters': int(data.get('params', 0)), 'test': test['metrics'], 'lambda': data.get('lambda', None)}, model_dir / 'test_metrics.json')
    plot_tsne(emb, labels, sample_rates, preds, class_names, model_dir, seed)
