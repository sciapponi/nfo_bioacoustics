"""
Fine-tune whale-trained FilmSirenClassifier on BirdSet high_sierra.

Transfers sinusoidal SIREN features learned on whale infrasound to high-frequency bird calls.

Usage:
    python train_birds_transfer.py --checkpoint ./checkpoints/best_model.pt --num-epochs 50 --subset high_sierra
    python train_birds_transfer.py --checkpoint ./checkpoints/best_model.pt --freeze-siren --freeze-modulator
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.siren import FilmSirenClassifier
from data.birdset_dataset import get_birdset_dataloader, BirdSetDataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SIREN on bird data")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt",
                        help="Whale-trained checkpoint to transfer from")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Use lower LR for fine-tuning")
    parser.add_argument("--bird-split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_birds")
    parser.add_argument("--freeze-siren", action="store_true",
                        help="Freeze SIREN core, only train new bird classifier head")
    parser.add_argument("--freeze-modulator", action="store_true",
                        help="Freeze modulator, only fine-tune SIREN + head")
    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 90)
    print("FINE-TUNE FILM-SIREN ON BIRD DATA (Transfer from Whale Model)")
    print("=" * 90)
    
    # Load bird dataset
    print(f"\n[1] Loading bird dataset ({args.bird_split} split)...")
    try:
        from data.bird_dataset import BirdCLEFDataset
        train_dataset = BirdCLEFDataset(
            split=args.bird_split,
            target_duration=5.0,
            target_sr=16000,
        )
        val_dataset = BirdCLEFDataset(
            split="validation" if args.bird_split == "train" else "test",
            target_duration=5.0,
            target_sr=16000,
        )
        num_bird_classes = train_dataset.num_classes
        train_loader = get_bird_dataloader(
            split=args.bird_split,
            batch_size=args.batch_size,
            target_sr=16000,
            shuffle=True,
        )
        val_loader = get_bird_dataloader(
            split="validation" if args.bird_split == "train" else "test",
            batch_size=args.batch_size,
            target_sr=16000,
            shuffle=False,
        )
        print(f"✓ Bird dataset loaded:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Classes: {num_bird_classes}")
    except Exception as e:
        print(f"✗ Failed to load bird dataset: {e}")
        return

    # Load whale-trained model
    print(f"\n[2] Loading whale-trained checkpoint: {args.checkpoint}")
    if not Path(args.checkpoint).exists():
        print(f"✗ Checkpoint not found")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Infer model dimensions from checkpoint
    state_dict = checkpoint["model_state_dict"]
    modulator_out_dim = state_dict["modulator.linear.weight"].shape[0]
    siren_hidden_dim = state_dict["siren_first.linear.weight"].shape[0]
    num_pe_freqs = (state_dict["siren_first.linear.weight"].shape[1] - 1) // 2
    
    # Create model with bird output classes
    model = FilmSirenClassifier(
        num_classes=num_bird_classes,  # NEW: adapt to bird classes
        modulator_out_dim=modulator_out_dim,
        siren_hidden_dim=siren_hidden_dim,
        num_siren_layers=3,
        num_pe_freqs=num_pe_freqs,
    )
    
    # Load whale weights into compatible parts
    whale_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    
    # Load everything except the classifier head (which has different output dim)
    for key, val in whale_state.items():
        if key.startswith("classifier"):
            continue  # Skip whale classifier, use fresh bird classifier
        if key in model_state:
            model_state[key] = val
    
    model.load_state_dict(model_state)
    model = model.to(device)
    print("✓ Whale model loaded and adapted to bird classes")
    print(f"  Detected dimensions: modulator_out={modulator_out_dim}, siren_hidden={siren_hidden_dim}, pe_freqs={num_pe_freqs}")
    print(f"  - Modulator: transferred")
    print(f"  - SIREN core: transferred")
    print(f"  - Classifier head: initialized fresh ({num_bird_classes} outputs)")

    # Freeze layers if requested
    if args.freeze_siren:
        print("\n[3] Freezing SIREN core (training only classifier head)")
        for name, param in model.named_parameters():
            if "siren" in name or "pe" in name:
                param.requires_grad = False
    
    if args.freeze_modulator:
        print("\n[3] Freezing modulator (training SIREN + head)")
        for name, param in model.named_parameters():
            if "modulator" in name:
                param.requires_grad = False

    # Set up optimizer and loss
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n[4] Training configuration:")
    print(f"  Learning rate: {args.learning_rate:.2e}")
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable):,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.num_epochs):
        # Train
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]"):
            # Handle both audio and spectrogram inputs
            if "audio" in batch:
                audio = batch["audio"]
            elif "spectrograms" in batch:
                # Spectrograms are 2D, need to convert to 1D audio representation
                # For now, take first mel-frequency dimension as proxy
                spec = batch["spectrograms"].squeeze(1)  # (B, n_mels, T) -> (B, n_mels, T)
                # Flatten to (B, n_mels*T) as audio
                audio = spec.reshape(spec.shape[0], -1)
            else:
                audio = batch[0]
            
            labels = batch.get("labels") or batch[1]
            
            audio, labels = audio.to(device), labels.to(device)
            
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]"):
                # Handle both audio and spectrogram inputs
                if "audio" in batch:
                    audio = batch["audio"]
                elif "spectrograms" in batch:
                    spec = batch["spectrograms"].squeeze(1)
                    audio = spec.reshape(spec.shape[0], -1)
                else:
                    audio = batch[0]
                
                labels = batch.get("labels") or batch[1]
                
                audio, labels = audio.to(device), labels.to(device)
                logits = model(audio)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_acc += (logits.argmax(dim=1) == labels).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path(args.output_dir) / f"best_model_birds_transfer.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_bird_classes": num_bird_classes,
            }, checkpoint_path)
            print(f"  → Saved best model: {checkpoint_path}")

    print(f"\n" + "=" * 90)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({num_bird_classes} bird classes)")
    print(f"=" * 90)


if __name__ == "__main__":
    main()
