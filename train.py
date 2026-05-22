"""
Training script for FNO + Conformer bioacoustic model.

Usage:
    python train.py
    python train.py --num-epochs 20 --batch-size 32
    python train.py --learning-rate 2e-3
    python train.py --resume checkpoints/best_model.pt
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
from collections import Counter
from pathlib import Path

from config import ExperimentConfig, get_default_config
from models import TinyConformer
from data import get_whale_dataloader


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(cfg: ExperimentConfig, num_classes: int) -> TinyConformer:
    """Create model from config."""
    return TinyConformer(
        num_classes=num_classes,  # Use adjusted class count
        fno_in_channels=cfg.fno.in_channels,
        fno_out_channels=cfg.fno.out_channels,
        fno_n_modes=cfg.fno.n_modes,
        fno_n_layers=cfg.fno.n_layers,
        cnn_channels=cfg.conformer.channels,
        cnn_num_blocks=cfg.conformer.num_blocks,
        cnn_kernel_size=cfg.conformer.kernel_size,
        dropout=cfg.conformer.dropout,
        sample_rate=cfg.data.target_sr,
    )


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    ignore_undefined: bool = False,
    log_interval: int = 50,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        audio = batch["audio"].to(device)  # (batch, seq_len)
        labels = batch["labels"].to(device)
        sample_rates = batch["sample_rates"].to(device)
        
        # Filter out Class 7 (Undefined) dynamically if flag is enabled
        if ignore_undefined:
            mask = (labels != 7)
            if not mask.any():
                continue  # Skip batch if it only contained class 7
            audio = audio[mask]
            labels = labels[mask]
            sample_rates = sample_rates[mask]
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(audio, sample_rate=sample_rates[0].item())  # Use first rate in batch
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * audio.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += audio.size(0)
        
        # Logging
        if (batch_idx + 1) % log_interval == 0 and total_samples > 0:
            avg_loss = total_loss / total_samples
            acc = total_correct / total_samples
            pbar.set_postfix({"loss": avg_loss, "acc": acc})
    
    if total_samples == 0:
        return 0.0, 0.0
        
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str,
    ignore_undefined: bool = False,
) -> tuple:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in pbar:
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        sample_rates = batch["sample_rates"].to(device)
        
        # Filter out Class 7 (Undefined) dynamically if flag is enabled
        if ignore_undefined:
            mask = (labels != 7)
            if not mask.any():
                continue
            audio = audio[mask]
            labels = labels[mask]
            sample_rates = sample_rates[mask]
        
        # Forward pass
        logits = model(audio, sample_rate=sample_rates[0].item())
        loss = criterion(logits, labels)
        
        # Metrics
        total_loss += loss.item() * audio.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += audio.size(0)
    
    if total_samples == 0:
        return 0.0, 0.0
        
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def print_class_distribution(dataset, dataset_name: str, total_possible_classes: int):
    """Extracts and prints the distribution of classes in a dataset."""
    labels = None
    for attr in ['labels', 'targets', 'y', 'audio_labels']:
        if hasattr(dataset, attr):
            labels = getattr(dataset, attr)
            break
            
    if labels is None:
        print(f"Could not automatically compute class distribution for {dataset_name}.")
        return

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    counts = Counter(labels)
    total = len(labels)
    
    print(f"  → {dataset_name} Original Distribution:")
    for cls in range(total_possible_classes):
        count = counts.get(cls, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"    Class {cls}: {count:<6} ({pct:.2f}%)")


def main():
    """Main training loop."""
    
    # Configuration flags
    IGNORE_UNDEFINED = True  # Toggle this to True to ignore Class 7
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train FNO + Conformer on WhaleSounds")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    
    # Load config
    cfg = get_default_config()
    
    # Override with CLI arguments
    if args.num_epochs:
        cfg.training.num_epochs = args.num_epochs
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.learning_rate:
        cfg.training.learning_rate = args.learning_rate
    if args.device:
        cfg.training.device = args.device
    if args.checkpoint_dir:
        cfg.training.checkpoint_dir = args.checkpoint_dir
    if args.seed:
        cfg.training.seed = args.seed
    
    # Set device
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    
    # Set seed
    set_seed(cfg.training.seed)
    
    # Create directories
    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.training.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("Loading data...")
    train_loader, train_dataset = get_whale_dataloader(
        split=cfg.data.split_train,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        target_duration=cfg.data.target_duration,
        target_sr=cfg.data.target_sr,
        shuffle=cfg.data.shuffle_train,
    )
    
    val_loader, val_dataset = get_whale_dataloader(
        split=cfg.data.split_valid,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        target_duration=cfg.data.target_duration,
        target_sr=cfg.data.target_sr,
        shuffle=False,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Total dataset classes: {cfg.num_classes}")
    
    # Print distributions before any dynamic filtering
    print_class_distribution(train_dataset, "Train Set", cfg.num_classes)
    print_class_distribution(val_dataset, "Validation Set", cfg.num_classes)
    
    # Determine actual network output size based on flag
    # If ignoring 7, network only outputs predictions up to class 6 (7 distinct classes: 0 to 6)
    num_active_classes = cfg.num_classes - 1 if IGNORE_UNDEFINED else cfg.num_classes
    if IGNORE_UNDEFINED:
        print(f"\n[INFO] IGNORE_UNDEFINED is active. Class 7 will be stripped from training batches.")
        print(f"       Model will output dimensions for {num_active_classes} classes (0 through 6).")
    
    # Create model
    print("\nCreating model...")
    model = create_model(cfg, num_classes=num_active_classes).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs,
        eta_min=1e-6,
    )
    
    # Optional: Load checkpoint state
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_acc = checkpoint.get("val_acc", 0.0)
            
            for _ in range(start_epoch):
                scheduler.step()
                
            print(f"Loaded checkpoint successfully. Resuming from epoch {start_epoch}")
        else:
            print(f"Error: No checkpoint found at '{args.resume}'")
            sys.exit(1)
            
    # Tensorboard
    writer = SummaryWriter(cfg.training.tensorboard_dir)
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch + 1} to {cfg.training.num_epochs} on {device}")
    print("-" * 80)
    
    best_epoch = start_epoch - 1 if start_epoch > 0 else 0
    
    for epoch in range(start_epoch, cfg.training.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            ignore_undefined=IGNORE_UNDEFINED,
            log_interval=cfg.training.log_interval,
        )
        
        # Evaluate
        if (epoch + 1) % cfg.training.eval_interval == 0:
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
                ignore_undefined=IGNORE_UNDEFINED,
            )
        else:
            val_loss, val_acc = 0.0, 0.0
        
        # Learning rate step
        scheduler.step()
        
        # Logging
        print(
            f"Epoch {epoch + 1:3d}/{cfg.training.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        # Tensorboard logging
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        if val_acc > 0:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_path = os.path.join(
                cfg.training.checkpoint_dir,
                f"best_model.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "cfg": cfg,
            }, checkpoint_path)
            print(f"  → Saved best model (val_acc: {val_acc:.4f})")
    
    print("-" * 80)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    
    writer.close()


if __name__ == "__main__":
    main()