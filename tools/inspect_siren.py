#!/usr/bin/env python3
"""Inspect SIREN encoder activations for a whale audio sample.

Saves activation heatmap, per-channel PSD, PCA trajectory, and optional saliency.
"""
import argparse
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import get_default_config
from models.siren import SirenConvClassifier, SirenEncoder
from data.whale_dataset import WhaleSoundsDataset, SyntheticWhaleDataset, get_whale_dataloader


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def load_sample(split, index, use_synthetic=False, target_duration=None, target_sr=None):
    if use_synthetic:
        ds = SyntheticWhaleDataset(num_samples=200, target_duration=target_duration or 4.0, target_sr=target_sr or 16000)
    else:
        ds = WhaleSoundsDataset(split=split, target_duration=target_duration or 4.0, target_sr=target_sr or 16000)

    if index < 0 or index >= len(ds):
        raise IndexError(f"Index {index} out of range for dataset of length {len(ds)}")

    sample = ds[index]
    audio, label, sr = sample
    # audio may be (1, T) or (T,)
    if audio.dim() == 2 and audio.size(0) == 1:
        audio = audio.squeeze(0)

    return audio.numpy(), int(label), int(sr), ds


def plot_activations(feats, outdir, basename="sample"):
    # feats: (C, T)
    ensure_dir(outdir)
    C, T = feats.shape

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(feats, aspect='auto', origin='lower', cmap='magma')
    fig.colorbar(im, ax=ax, label='activation')
    ax.set_xlabel('time (frames)')
    ax.set_ylabel('feature channel')
    ax.set_title('SIREN encoder activations')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{basename}_activations.png"), dpi=150)
    plt.close(fig)


def plot_psd(feats, sr, outdir, basename="sample", nch=4):
    ensure_dir(outdir)
    C, T = feats.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / sr)
    power = np.abs(np.fft.rfft(feats, axis=1)) ** 2

    nch = min(nch, C)
    fig, ax = plt.subplots(figsize=(10, 4))
    for ch in range(nch):
        arr = power[ch]
        if arr.max() > 0:
            arr = arr / arr.max()
        ax.semilogx(freqs + 1e-6, arr, label=f'ch{ch}')
    ax.set_xlabel('Hz')
    ax.set_ylabel('Normalized power')
    ax.set_title('Power spectra of selected encoder channels')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{basename}_psd.png"), dpi=150)
    plt.close(fig)


def plot_pca_trajectory(feats, outdir, basename="sample"):
    ensure_dir(outdir)
    X = feats.T  # (T, C)
    pca = PCA(n_components=3)
    traj = pca.fit_transform(X)
    t = np.linspace(0, 1, len(traj))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=t, cmap='viridis', s=2)
    ax.set_title('PCA trajectory of encoder features (time colored)')
    fig.colorbar(sc, label='normalized time')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{basename}_pca.png"), dpi=150)
    plt.close(fig)


def compute_saliency(classifier, audio, target_class, device='cpu'):
    # audio: 1D numpy
    classifier = classifier.to(device).eval()
    x = torch.from_numpy(audio).unsqueeze(0).to(device)
    x_var = x.clone().detach().requires_grad_(True)
    logits = classifier(x_var)
    score = logits[0, target_class]
    score.backward()
    sal = x_var.grad.abs().cpu().numpy()[0]
    return sal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--use-synthetic', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional path to trained SirenConvClassifier checkpoint to load')
    parser.add_argument('--outdir', type=str, default='plots')
    parser.add_argument('--save-audio', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    cfg = get_default_config()
    target_duration = cfg.data.target_duration
    target_sr = cfg.data.target_sr

    audio, label, sr, ds = load_sample(args.split, args.index, use_synthetic=args.use_synthetic, target_duration=target_duration, target_sr=target_sr)

    # instantiate encoder
    encoder = SirenEncoder(feat_dim=64).to(args.device).eval()

    # if checkpoint provided, load encoder weights from classifier
    classifier = None
    if args.checkpoint is not None:
        # torch.load defaults to weights_only=True in newer PyTorch; allow full checkpoint
        try:
            ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        except TypeError:
            # older torch doesn't accept weights_only kwarg
            ckpt = torch.load(args.checkpoint, map_location=args.device)
        except Exception as e:
            print(f'Failed to load checkpoint: {e}')
            ckpt = None

            if ckpt is not None:
                # Determine state_dict inside checkpoint
                state_dict = None
                if isinstance(ckpt, dict):
                    if 'model_state_dict' in ckpt:
                        state_dict = ckpt['model_state_dict']
                    else:
                        # Maybe the checkpoint is directly a state dict
                        if all(isinstance(v, (torch.Tensor,)) for v in ckpt.values()):
                            state_dict = ckpt

                if state_dict is not None:
                    # Attempt to infer encoder feat_dim from checkpoint and re-create encoder
                    encoder_prefixed_keys = [k for k in state_dict.keys() if k.startswith('encoder.net.') and k.endswith('.weight')]
                    if encoder_prefixed_keys:
                        # pick the weight with the largest index (likely the final conv -> out_channels == feat_dim)
                        try:
                            max_idx = max(int(k.split('.')[2]) for k in encoder_prefixed_keys)
                            final_key = f'encoder.net.{max_idx}.weight'
                            feat_dim_ckpt = state_dict[final_key].shape[0]
                            # Recreate encoder if dims mismatch
                            if hasattr(encoder, 'net') and encoder.net[-1] is not None:
                                # current encoder feat dim
                                cur_feat = encoder.net[-1].out_channels if hasattr(encoder.net[-1], 'out_channels') else None
                            else:
                                cur_feat = None
                            if cur_feat != feat_dim_ckpt:
                                print(f'Recreating SirenEncoder with feat_dim={feat_dim_ckpt} to match checkpoint')
                                encoder = SirenEncoder(feat_dim=feat_dim_ckpt).to(args.device)
                        except Exception:
                            pass

                    # Try to load encoder weights by matching keys with prefix 'encoder.'
                    enc_state = encoder.state_dict()
                    loaded_keys = []
                    for k, v in list(state_dict.items()):
                        if k.startswith('encoder.'):
                            targ = k[len('encoder.'):]
                            if targ in enc_state and enc_state[targ].shape == v.shape:
                                enc_state[targ] = v
                                loaded_keys.append(targ)

                    if loaded_keys:
                        try:
                            encoder.load_state_dict(enc_state)
                            print(f'Loaded {len(loaded_keys)} encoder parameters from checkpoint')
                        except Exception as e:
                            print(f'Failed to load encoder state dict: {e}')
                    else:
                        print('No matching encoder parameters found in checkpoint; skipping encoder weight load')

                    # If possible, also try to construct classifier for saliency when shapes match
                    try:
                        classifier = SirenConvClassifier(num_classes=cfg.num_classes)
                        classifier.load_state_dict(state_dict)
                    except Exception:
                        # shapes didn't match; we'll skip classifier saliency
                        classifier = None
                else:
                    print('Checkpoint did not contain a recognized state_dict; skipping weight load')

    # Prepare audio tensor
    x = torch.from_numpy(audio).unsqueeze(0).to(args.device)  # (1, T)
    with torch.no_grad():
        feats = encoder(x)  # (1, C, T)
    feats = feats.squeeze(0).cpu().numpy()  # (C, T)

    outdir = args.outdir
    basename = f'split_{args.split}_idx_{args.index}_label_{label}'

    plot_activations(feats, outdir, basename)
    plot_psd(feats, sr, outdir, basename, nch=6)
    plot_pca_trajectory(feats, outdir, basename)

    if args.checkpoint is not None and classifier is not None:
        sal = compute_saliency(classifier, audio, target_class=label, device=args.device)
        ensure_dir(outdir)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 2))
        t = np.linspace(0, len(sal) / sr, len(sal))
        plt.plot(t, sal)
        plt.xlabel('time (s)')
        plt.title('Input saliency (abs grad)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{basename}_saliency.png"), dpi=150)
        plt.close()

    if args.save_audio:
        import soundfile as sf
        ensure_dir(outdir)
        sf.write(os.path.join(outdir, f"{basename}.wav"), audio, sr)

    print(f'Plots saved to {outdir}')


if __name__ == '__main__':
    main()
