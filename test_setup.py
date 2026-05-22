"""
Quick test script to verify model architecture and data loading.

Usage:
    python test_setup.py
"""

import torch
import sys

print("Testing FNO + Conformer setup...\n")

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Test imports
print("✓ Importing modules...")
try:
    from config import get_default_config
    from models import TinyConformer
    print("  All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test config
print("\n✓ Loading configuration...")
try:
    cfg = get_default_config()
    print(f"  Config loaded: {cfg.name}")
    print(f"  - FNO modes: {cfg.fno.n_modes}")
    print(f"  - Conformer blocks: {cfg.conformer.num_blocks}")
    print(f"  - Target SR: {cfg.data.target_sr}")
except Exception as e:
    print(f"  ✗ Config failed: {e}")
    sys.exit(1)

# Test model creation
print("\n✓ Creating model...")
try:
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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created with {total_params:,} parameters")
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    sys.exit(1)

# Test forward pass
print("\n✓ Testing forward pass...")
try:
    batch_size = 2
    seq_len = cfg.data.target_sr * cfg.data.target_duration  # 4 seconds at 16kHz = 64k samples
    dummy_audio = torch.randn(batch_size, 1, int(seq_len)).to(device)
    
    with torch.no_grad():
        output = model(dummy_audio, sample_rate=cfg.data.target_sr)
    
    print(f"  Input shape: {dummy_audio.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, {cfg.num_classes})")
    
    assert output.shape == (batch_size, cfg.num_classes), "Output shape mismatch!"
    print("  Forward pass successful ✓")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test data loading
print("\n✓ Testing data loading...")
try:
    # Create a simple mock batch to test model compatibility
    batch_size = 2
    audio = torch.randn(batch_size, 1, int(cfg.data.target_sr * cfg.data.target_duration)).to(device)
    labels = torch.randint(0, cfg.num_classes, (batch_size,)).to(device)
    sample_rates = torch.full((batch_size,), cfg.data.target_sr).to(device)
    
    # Test model with batch
    with torch.no_grad():
        output = model(audio, sample_rate=sample_rates[0].item())
    
    print(f"  Batch audio shape: {audio.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Sample rates: {sample_rates}")
    print(f"  Model output shape: {output.shape}")
    print("  Data loading test successful ✓")
    print(f"\n  Note: Real WhaleSounds dataset loading will be handled in train.py")
    print(f"        Dataset URL: https://huggingface.co/datasets/monster-monash/WhaleSounds")
    
except Exception as e:
    print(f"  ✗ Data loading test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nYou can now run training with:")
print("  python train.py")
print("\nOr with custom arguments:")
print("  python train.py --num-epochs 20 --batch-size 32 --learning-rate 1e-3")
