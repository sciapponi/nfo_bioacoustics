"""
Training script for bird sound classification using 2D FNO.

Step 3: Pre-train on bird data to learn continuous spectrogram representations.

Usage:
    python train_birds.py
    python train_birds.py --num-epochs 50 --batch-size 32 --learning-rate 1e-3
    python train_birds.py --num-samples 500  # Use only 500 samples for quick testing
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime

from config import ExperimentConfig, get_default_config
from models.spectrogram_model import SpectrogramModel, FNOConformer2D, EfficientNetSpectrogram
from data.bird_dataset import get_bird_dataloader


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(
    num_classes: int,
    model_type: str = "spectrogram",
    fno_out_channels: int = 32,
    fno_n_layers: int = 4,
    fno_n_modes_time: int = 32,
    fno_n_modes_freq: int = 32,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """Create bird classification model."""
    if model_type == "spectrogram":
        return SpectrogramModel(
            num_classes=num_classes,
            fno_in_channels=1,
            fno_out_channels=fno_out_channels,
            fno_n_layers=fno_n_layers,
            fno_n_modes_time=fno_n_modes_time,
            fno_n_modes_freq=fno_n_modes_freq,
        )
    elif model_type == "hybrid":
        return FNOConformer2D(
            num_classes=num_classes,
            fno_in_channels=1,
            fno_out_channels=fno_out_channels,
            fno_n_layers=fno_n_layers,
            fno_n_modes_time=fno_n_modes_time,
            fno_n_modes_freq=fno_n_modes_freq,
        )
    elif model_type == "efficientnet":
        return EfficientNetSpectrogram(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    elif model_type == "spectrogram_conv":
        from models.spectrogram_model import SpectrogramModelConv
        return SpectrogramModelConv(
            num_classes=num_classes,
            fno_in_channels=1,
            fno_out_channels=fno_out_channels,
            fno_n_layers=fno_n_layers,
            fno_n_modes_time=fno_n_modes_time,
            fno_n_modes_freq=fno_n_modes_freq,
        )
    elif model_type == "fno_time":
        from models.spectrogram_model import SpectrogramFNOTime
        # Use conv_channels = fno_out_channels // 2 as a sensible default
        conv_ch = max(8, fno_out_channels // 2)
        return SpectrogramFNOTime(
            num_classes=num_classes,
            fno_in_channels=4,
            conv_channels=conv_ch,
            fno_out_channels=fno_out_channels,
            fno_n_layers=fno_n_layers,
            fno_n_modes_time=fno_n_modes_time,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    log_interval: int = 50,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        spectrograms = batch["spectrograms"].to(device)  # (batch, 1, n_mels, time_steps)
        labels = batch["labels"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(spectrograms)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * spectrograms.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += spectrograms.size(0)
        
        # Logging
        if (batch_idx + 1) % log_interval == 0 and total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})
    
    if total_samples > 0:
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
    else:
        train_loss, train_acc = 0.0, 0.0
    
    return train_loss, train_acc


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in pbar:
        spectrograms = batch["spectrograms"].to(device)
        labels = batch["labels"].to(device)
        
        logits = model(spectrograms)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * spectrograms.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += spectrograms.size(0)
    
    if total_samples > 0:
        eval_loss = total_loss / total_samples
        eval_acc = total_correct / total_samples
    else:
        eval_loss, eval_acc = 0.0, 0.0
    
    return eval_loss, eval_acc


def main():
    parser = argparse.ArgumentParser(description="Train 2D FNO on bird sounds")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-type", type=str, default="spectrogram", 
                        choices=["spectrogram", "spectrogram_conv", "fno_time", "hybrid", "efficientnet"],
                        help="Model architecture")
    parser.add_argument("--fno-layers", type=int, default=None, help="Number of FNO layers (overrides config)")
    parser.add_argument("--fno-out-channels", type=int, default=None, help="FNO output channels (overrides config)")
    parser.add_argument("--fno-modes-time", type=int, default=None, help="FNO temporal modes (overrides config)")
    parser.add_argument("--fno-modes-freq", type=int, default=None, help="FNO frequency modes (overrides config)")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained EfficientNet weights when available")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit dataset size (for testing)")
    parser.add_argument("--data-root", type=str, default=None, help="Path to local bird dataset root directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_birds", help="Checkpoint directory")
    parser.add_argument("--tensorboard-dir", type=str, default="./logs_birds", help="TensorBoard directory")
    
    args = parser.parse_args()
    
    # Load config and resolve defaults
    cfg = get_default_config()

    # If CLI FNO params are not provided, use config defaults
    if args.fno_layers is None:
        args.fno_layers = cfg.fno2d.n_layers
    if args.fno_out_channels is None:
        args.fno_out_channels = cfg.fno2d.out_channels
    if args.fno_modes_time is None:
        args.fno_modes_time = cfg.fno2d.n_modes_time
    if args.fno_modes_freq is None:
        args.fno_modes_freq = cfg.fno2d.n_modes_freq

    # Setup
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(args.tensorboard_dir)
    
    print("\n" + "="*80)
    print("STEP 3: Pre-train 2D FNO on Bird Sounds")
    print("="*80)
    
    # Load data
    print("\nLoading bird datasets...")
    if args.data_root:
        from data.bird_dataset import get_local_bird_dataloader
        train_loader, train_dataset = get_local_bird_dataloader(
            root_dir=args.data_root,
            split="Train",
            batch_size=args.batch_size,
            shuffle=True,
            target_duration=get_default_config().bird_data.target_duration,
            target_sr=get_default_config().bird_data.target_sr,
            num_samples=args.num_samples,
        )
        val_loader, val_dataset = get_local_bird_dataloader(
            root_dir=args.data_root,
            split="Validation",
            batch_size=args.batch_size,
            shuffle=False,
            target_duration=get_default_config().bird_data.target_duration,
            target_sr=get_default_config().bird_data.target_sr,
            num_samples=args.num_samples,
        )
    else:
        train_loader = get_bird_dataloader(
            split="train",
            batch_size=args.batch_size,
            shuffle=True,
            target_duration=get_default_config().bird_data.target_duration,
            target_sr=get_default_config().bird_data.target_sr,
            n_mels=get_default_config().bird_data.n_mels,
            n_fft=get_default_config().bird_data.n_fft,
            hop_length=get_default_config().bird_data.hop_length,
            num_samples=args.num_samples,
        )
        val_loader = get_bird_dataloader(
            split="validation",
            batch_size=args.batch_size,
            shuffle=False,
            target_duration=get_default_config().bird_data.target_duration,
            target_sr=get_default_config().bird_data.target_sr,
            n_mels=get_default_config().bird_data.n_mels,
            n_fft=get_default_config().bird_data.n_fft,
            hop_length=get_default_config().bird_data.hop_length,
            num_samples=args.num_samples,
        )
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
    
    # Get number of classes from dataset
    train_dataset = train_loader.dataset
    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = create_model(
        num_classes=num_classes,
        model_type=args.model_type,
        fno_out_channels=args.fno_out_channels,
        fno_n_layers=args.fno_layers,
        fno_n_modes_time=args.fno_modes_time,
        fno_n_modes_freq=args.fno_modes_freq,
        pretrained=args.pretrained,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler with warmup
    def get_lr_schedule(optimizer, warmup_steps, num_epochs, steps_per_epoch):
        """Create learning rate schedule with linear warmup."""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(num_epochs - current_step // steps_per_epoch) / float(max(1, num_epochs)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = get_lr_schedule(optimizer, args.warmup_steps, args.num_epochs, len(train_loader))
    
    # Training loop
    print("\n" + "="*80)
    print(f"Starting training for {args.num_epochs} epochs on {device}")
    print("="*80 + "\n")
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
        
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("accuracy/val", val_acc, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_path = Path(args.checkpoint_dir) / "best_model_birds.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "model_type": args.model_type,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, checkpoint_path)
    
    writer.close()
    
    print("\n" + "="*80)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {Path(args.checkpoint_dir) / 'best_model_birds.pt'}")
    print("="*80)


if __name__ == "__main__":
    main()
