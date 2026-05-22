"""
Inference script for trained model.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt --audio path/to/audio.wav
"""

import argparse
import torch
import torchaudio
from pathlib import Path

from config import get_default_config
from models import TinyConformer


def load_checkpoint(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["cfg"]
    
    # Recreate model
    model = TinyConformer(
        num_classes=cfg.num_classes,
        fno_in_channels=cfg.fno.in_channels,
        fno_out_channels=cfg.fno.out_channels,
        fno_n_modes=cfg.fno.n_modes,
        fno_n_layers=cfg.fno.n_layers,
        cnn_channels=cfg.conformer.channels,
        cnn_num_blocks=cfg.conformer.num_blocks,
        cnn_kernel_size=cfg.conformer.kernel_size,
        dropout=cfg.conformer.dropout,
        sample_rate=cfg.data.target_sr,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, cfg


@torch.no_grad()
def predict(model, audio_path: str, device: str, target_sr: int = 16000):
    """
    Predict class for an audio file.
    
    Args:
        model: Trained model
        audio_path: Path to audio file
        device: Device to use
        target_sr: Target sample rate
    
    Returns:
        Predicted class index, logits
    """
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # Convert to mono if needed
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Normalize
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val
    
    # Pad/truncate to fixed length (4 seconds default)
    target_samples = int(target_sr * 4.0)
    if audio.shape[1] < target_samples:
        audio = torch.nn.functional.pad(
            audio,
            (0, target_samples - audio.shape[1]),
            mode='constant'
        )
    else:
        audio = audio[:, :target_samples]
    
    # Predict
    audio = audio.to(device)
    logits = model(audio.unsqueeze(0), sample_rate=float(target_sr))
    
    pred_idx = logits.argmax(dim=1).item()
    
    return pred_idx, logits


def main():
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_checkpoint(args.checkpoint, args.device)
    
    # Predict
    print(f"Processing {args.audio}...")
    pred_idx, logits = predict(model, args.audio, args.device, cfg.data.target_sr)
    
    # Get class name from dataset
    from data import WhaleSoundsDataset
    dataset = WhaleSoundsDataset()
    class_name = dataset.get_class_name(pred_idx)
    
    print(f"Predicted class: {class_name} (idx: {pred_idx})")
    print(f"Logits: {logits}")


if __name__ == "__main__":
    main()
