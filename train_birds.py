"""
Train FilmSIREN on BirdSet high_sierra dataset (raw waveforms).

Usage:
    python train_birds.py --subset high_sierra --num-epochs 50 --batch-size 32 --device cuda
    python train_birds.py --subset high_sierra --num-samples 100  # Quick test
    python train_birds.py --checkpoint ./checkpoints_birds/best_model_birds.pt --resume
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models.siren import FilmSirenClassifier
from data.birdset_dataset import get_birdset_dataloader


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train FilmSIREN on BirdSet")
    parser.add_argument("--subset", type=str, default="high_sierra",
                        help="BirdSet subset (high_sierra, nips4b_bg, etc.)")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_birds")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit to N samples for quick testing")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(42)
    
    print("=" * 90)
    print("TRAIN FilmSIREN ON BirdSet")
    print("=" * 90)
    
    # Load datasets
    print(f"\n[1] Loading BirdSet {args.subset}...")
    try:
        train_loader, train_dataset = get_birdset_dataloader(
            subset=args.subset,
            split="train",
            batch_size=args.batch_size,
            target_sr=16000,
            target_duration=5.0,
            num_samples=args.num_samples,
            shuffle=True,
        )
        val_loader, val_dataset = get_birdset_dataloader(
            subset=args.subset,
            split="val",
            batch_size=args.batch_size,
            target_sr=16000,
            target_duration=5.0,
            num_samples=args.num_samples,
            shuffle=False,
        )
        num_classes = train_dataset.num_classes
        print(f"✓ Train: {len(train_dataset)} samples")
        print(f"✓ Val: {len(val_dataset)} samples")
        print(f"✓ Classes: {num_classes}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Create model
    print(f"\n[2] Creating FilmSIREN model ({num_classes} classes)...")
    model = FilmSirenClassifier(
        num_classes=num_classes,
        modulator_out_dim=32,
        siren_hidden_dim=128,
        num_siren_layers=3,
        num_pe_freqs=12,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable:,}")

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    best_val_acc = 0.0
    
    # Resume from checkpoint if requested
    if args.resume and args.checkpoint and Path(args.checkpoint).exists():
        print(f"\n[3] Resuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("val_acc", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    else:
        print(f"\n[3] Training from scratch")

    # Training loop
    print(f"\n[4] Training...")
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", leave=False):
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(audio)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (logits.argmax(dim=1) == labels).float().mean().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]", leave=False):
                audio = batch["audio"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(audio)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_acc += (logits.argmax(dim=1) == labels).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path(args.output_dir) / "best_model_birds.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
            }, checkpoint_path)
            print(f"  → Saved best model: {checkpoint_path}")

    print(f"\n" + "=" * 90)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"=" * 90)


if __name__ == "__main__":
    main()
