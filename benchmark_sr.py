"""
Super-Resolution / Discretization-Invariance Benchmark for FNO Bioacoustic Model.

This script evaluates a pre-trained FNO model across multiple evaluation sampling rates
to demonstrate its zero-shot super-resolution capability.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from config import get_default_config
from models import TinyConformer
from data import get_whale_dataloader


def create_model_from_checkpoint(checkpoint_path: str, device: str) -> tuple:
    """Loads model architecture and weights from the best saved checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    print(f"-> Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    
    # Check if checkpoint used filtered classes
    # If the training skipped class 7, the checkpoint config reflects the adjusted head
    num_classes = cfg.num_classes - 1 if getattr(cfg, "IGNORE_UNDEFINED", True) else cfg.num_classes
    
    # Reconstruct the exact architecture
    model = TinyConformer(
        num_classes=num_classes,
        fno_in_channels=cfg.fno.in_channels,
        fno_out_channels=cfg.fno.out_channels,
        fno_n_modes=cfg.fno.n_modes,
        fno_n_layers=cfg.fno.n_layers,
        cnn_channels=cfg.conformer.channels,
        cnn_num_blocks=cfg.conformer.num_blocks,
        cnn_kernel_size=cfg.conformer.kernel_size,
        dropout=cfg.conformer.dropout,
        sample_rate=cfg.data.target_sr, # Baseline training SR
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, cfg


@torch.no_grad()
def evaluate_at_resolution(model: nn.Module, val_loader, device: str) -> tuple:
    """Evaluates the model on a dataset at a specific sampling rate."""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    for batch in val_loader:
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        sample_rates = batch["sample_rates"].to(device)
        
        # Pass the current batch's sampling rate directly into the FNO frontend
        current_sr = sample_rates[0].item()
        logits = model(audio, sample_rate=current_sr)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * audio.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += audio.size(0)
        
    if total_samples == 0:
        return 0.0, 0.0
        
    return total_loss / total_samples, total_correct / total_samples


def main():
    # --- CONFIGURATION ---
    CHECKPOINT_PATH = "checkpoints/best_model.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IGNORE_CLASS_7 = True  # Must match your training setup
    
    # Define the evaluation sample rates (Super-Resolution Spectrum)
    # Example: If trained at 250 Hz, test up to 2000 Hz scaling
    EVAL_SAMPLE_RATES = [250, 500, 1000, 2000] 
    # ---------------------
    
    # 1. Load Model and baseline configurations
    try:
        model, cfg = create_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)
        print(f"Successfully loaded model. Baseline training SR: {cfg.data.target_sr} Hz")
    except Exception as e:
        print(f"Could not load checkpoint: {e}. Falling back to default untrained initialization.")
        cfg = get_default_config()
        model = TinyConformer(num_classes=7, sample_rate=250.0).to(DEVICE)
        model.eval()
    
    results = []
    
    print("\n" + "="*50)
    print("RUNNING ZERO-SHOT SUPER-RESOLUTION BENCHMARK")
    print("="*50)
    
    # 2. Loop over each target sampling rate
    for sr in EVAL_SAMPLE_RATES:
        print(f"\n[Evaluating SR: {sr} Hz]")
        print(f"-> Generating validation dataloader interpolated to {sr} Hz...")
        
        # Instantiate the dataloader at the specified super-resolution target rate
        # Your data script uses np.interp to smoothly upscale/downscale sequence vectors
        val_loader, _ = get_whale_dataloader(
            split=cfg.data.split_valid,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            target_duration=cfg.data.target_duration,
            target_sr=sr,               # The benchmark magic happens here
            shuffle=False,
            ignore_class_7=IGNORE_CLASS_7
        )
        
        # Run the validation sweep
        val_loss, val_acc = evaluate_at_resolution(model, val_loader, DEVICE)
        
        # Change line 125 to:
        scaling_factor = sr / cfg.data.target_sr
        print(f"   Results @ {sr} Hz (Scale: {scaling_factor:.1f}x) | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # And change the dictionary tracking on line 127/128 to:
        results.append({
            "Evaluation SR (Hz)": sr,
            "Scale Factor": f"{scaling_factor:.1f}x",
            "Val Loss": round(val_loss, 4),
            "Val Accuracy": round(val_acc, 4)
        })
        
    # 3. Print out final comparison report
    print("\n" + "="*50)
    print("FINAL BENCHMARK SUMMARY")
    print("="*50)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Optional: Save results out to disk for plotting later
    df.to_csv("checkpoints/sr_benchmark_results.csv", index=False)
    print("\nResults saved to checkpoints/sr_benchmark_results.csv")


if __name__ == "__main__":
    main()