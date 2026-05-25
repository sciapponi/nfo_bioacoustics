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
from types import SimpleNamespace
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
    # Default conformer model
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
    mixup_alpha: float = 0.0,
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
        
        # Forward pass (with optional mixup)
        optimizer.zero_grad()
        if mixup_alpha and mixup_alpha > 0.0 and audio.size(0) > 1:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(audio.size(0))
            audio_mixed = lam * audio + (1 - lam) * audio[perm]
            logits = model(audio_mixed, sample_rate=sample_rates[0].item())
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])
        else:
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
    parser.add_argument("--model-type", type=str, default="conformer", choices=["conformer", "siren_conv", "film_siren"], help="Model architecture to train")
    # Siren / model sizing
    parser.add_argument("--siren-feat-dim", type=int, default=None, help="Feature dim for Siren encoder")
    parser.add_argument("--siren-blocks", type=int, default=None, help="Number of SIREN blocks in encoder")
    parser.add_argument("--conv-blocks", type=int, default=None, help="Number of conv blocks in Siren classifier")
    parser.add_argument("--conv-channels", type=int, default=None, help="Conv channels for Siren classifier (overrides config)")
    # Optimizer / training recipe
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay for optimizer")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup alpha (0 to disable)")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CrossEntropyLoss")
    parser.add_argument("--lr-scheduler", type=str, default=None, choices=["cosine", "none"], help="Learning rate scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Linear warmup epochs before LR schedule")
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
    # CLI overrides for siren/model sizing and training recipe
    if args.siren_feat_dim:
        cfg.siren = getattr(cfg, 'siren', None) or SimpleNamespace()
        cfg.siren.feat_dim = args.siren_feat_dim
    if args.siren_blocks:
        cfg.siren = getattr(cfg, 'siren', None) or SimpleNamespace()
        cfg.siren.num_blocks = args.siren_blocks
    if args.conv_blocks:
        cfg.siren = getattr(cfg, 'siren', None) or SimpleNamespace()
        cfg.siren.conv_blocks = args.conv_blocks
    if args.conv_channels:
        cfg.siren = getattr(cfg, 'siren', None) or SimpleNamespace()
        cfg.siren.conv_channels = args.conv_channels
    if args.optimizer:
        cfg.training.optimizer = args.optimizer
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay
    if args.mixup is not None:
        cfg.training.mixup = args.mixup
    if args.label_smoothing is not None:
        cfg.training.label_smoothing = args.label_smoothing
    if args.lr_scheduler is not None:
        cfg.training.lr_scheduler = args.lr_scheduler
    if args.warmup_epochs is not None:
        cfg.training.warmup_epochs = args.warmup_epochs
    
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
    print(f"Raw WhaleSounds labels: {cfg.num_classes} (7 active classes after filtering class 7)")
    
    # Print distributions before any dynamic filtering
    print_class_distribution(train_dataset, "Train Set", cfg.num_classes)
    print_class_distribution(val_dataset, "Validation Set", cfg.num_classes)
    
    # Determine actual network output size based on flag
    # If ignoring 7, network only outputs predictions up to class 6 (7 distinct classes: 0 to 6)
    num_active_classes = cfg.num_classes - 1 if IGNORE_UNDEFINED else cfg.num_classes
    if IGNORE_UNDEFINED:
        print(f"\n[INFO] IGNORE_UNDEFINED is active. Class 7 ('Unidentified') will be stripped from training batches.")
        print(f"       Model will output dimensions for {num_active_classes} active classes (0 through 6).")
    
    # Create model
    print("\nCreating model...")
    model = create_model(cfg, num_classes=num_active_classes)
    # If user requested a different model type, override here
    if args.model_type == "siren_conv":
        from models.siren import SirenConvClassifier
        siren_cfg = getattr(cfg, 'siren', None)
        feat_dim = getattr(siren_cfg, 'feat_dim', cfg.fno.out_channels)
        conv_blocks = getattr(siren_cfg, 'conv_blocks', 2)
        siren_num_layers = getattr(siren_cfg, 'num_blocks', 3)
        conv_channels = getattr(siren_cfg, 'conv_channels', cfg.conformer.channels)
        model = SirenConvClassifier(
            num_classes=num_active_classes,
            feat_dim=feat_dim,
            conv_channels=conv_channels,
            conv_blocks=conv_blocks,
            siren_num_layers=siren_num_layers,
            kernel_size=5,
        )
    elif args.model_type == "film_siren":
        from models.siren import FilmSirenClassifier
        siren_cfg = getattr(cfg, 'siren', None)
        modulator_out_dim = getattr(siren_cfg, 'feat_dim', 64)
        siren_hidden_dim = getattr(siren_cfg, 'conv_channels', 128)
        num_siren_layers = getattr(siren_cfg, 'num_blocks', 3)
        model = FilmSirenClassifier(
            num_classes=num_active_classes,
            modulator_out_dim=modulator_out_dim,
            siren_hidden_dim=siren_hidden_dim,
            num_siren_layers=num_siren_layers,
            w0_first=30.0,
            w0_hidden=1.0,
        )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer (support Adam/AdamW) and loss (with optional label smoothing)
    opt_name = getattr(cfg.training, 'optimizer', 'adamw')
    weight_decay = getattr(cfg.training, 'weight_decay', 0.0)
    base_lr = cfg.training.learning_rate
    
    # For FiLM-SIREN, use lower modulator LR to stabilize training
    if args.model_type == "film_siren":
        modulator_params = model.modulator.parameters()
        other_params = [p for name, p in model.named_parameters() if not name.startswith('modulator')]
        modulator_lr = base_lr * 0.1  # 10x lower LR for modulator
        param_groups = [
            {'params': modulator_params, 'lr': modulator_lr},
            {'params': other_params, 'lr': base_lr}
        ]
        print(f"Using separate LRs: modulator={modulator_lr:.2e}, siren={base_lr:.2e}")
        if opt_name == 'adam':
            optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        if opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    label_smoothing = getattr(cfg.training, 'label_smoothing', 0.0)
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:
        # Older PyTorch may not support label_smoothing param
        criterion = nn.CrossEntropyLoss()

    # LR scheduling: support cosine with optional linear warmup (we'll compute lrs manually)
    lr_scheduler_type = getattr(cfg.training, 'lr_scheduler', 'cosine')
    warmup_epochs = getattr(cfg.training, 'warmup_epochs', 0)
    T = cfg.training.num_epochs

    def compute_lr_for_epoch(ep: int):
        if warmup_epochs > 0 and ep < warmup_epochs:
            return base_lr * float(ep + 1) / float(max(1, warmup_epochs))
        if lr_scheduler_type == 'cosine':
            # cosine decay after warmup
            e = max(0, ep - warmup_epochs)
            T_cos = max(1, T - warmup_epochs)
            return 1e-6 + 0.5 * (base_lr - 1e-6) * (1 + np.cos(np.pi * e / T_cos))
        return base_lr
    
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
                # advance manual lr schedule pointer (no-op here, we set lr below)
                pass
                
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
        # Set learning rate for this epoch (supports warmup + cosine)
        lr = compute_lr_for_epoch(epoch)
        
        # For film_siren, keep modulator at 10% of SIREN LR
        if args.model_type == "film_siren":
            for i, g in enumerate(optimizer.param_groups):
                if i == 0:  # modulator group
                    g['lr'] = lr * 0.1
                else:  # siren/other groups
                    g['lr'] = lr
        else:
            for g in optimizer.param_groups:
                g['lr'] = lr
        
        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            ignore_undefined=IGNORE_UNDEFINED,
            log_interval=cfg.training.log_interval,
            mixup_alpha=getattr(cfg.training, 'mixup', 0.0),
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
        
        # Learning rate step is handled manually by compute_lr_for_epoch
        
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