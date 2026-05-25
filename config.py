"""
Configuration for FNO bioacoustics experiments.

Minimal, easy-to-modify configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FNOConfig:
    """1D Fourier Neural Operator configuration."""
    
    in_channels: int = 1
    out_channels: int = 16
    n_modes: int = 64
    n_layers: int = 4
    sample_rate: float = 250.0


@dataclass
class FNO2DConfig:
    """2D Fourier Neural Operator configuration (for spectrograms)."""
    
    in_channels: int = 1
    out_channels: int = 32
    n_modes_time: int = 16
    n_modes_freq: int = 16
    n_layers: int = 2
    sample_rate: float = 16000.0


@dataclass
class ConformerConfig:
    """CNN backbone configuration."""
    
    channels: int = 32
    num_blocks: int = 1
    kernel_size: int = 3
    dropout: float = 0.3


@dataclass
class DataConfig:
    """Data loading configuration."""
    
    split_train: str = "train"
    split_valid: str = "validation"
    batch_size: int = 64
    num_workers: int = 0
    target_duration: float = 4.0  # seconds
    target_sr: int = 250
    shuffle_train: bool = True
    dataset_type: str = "whale"  # whale or bird


@dataclass
class BirdDataConfig:
    """Bird sound data configuration (spectrograms)."""
    
    split_train: str = "train"
    split_valid: str = "validation"
    batch_size: int = 32
    num_workers: int = 0
    target_duration: float = 5.0  # seconds
    target_sr: int = 24000
    n_mels: int = 64
    n_fft: int = 2048
    hop_length: int = 512
    shuffle_train: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    log_interval: int = 50
    eval_interval: int = 1  # Evaluate every N epochs
    device: str = "cuda"  # GPU available
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./logs"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    name: str = "fno_whale_experiment"
    description: str = "FNO + Conformer for whale sound classification"
    
    fno: FNOConfig = field(default_factory=FNOConfig)
    fno2d: FNO2DConfig = field(default_factory=FNO2DConfig)
    conformer: ConformerConfig = field(default_factory=ConformerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    bird_data: BirdDataConfig = field(default_factory=BirdDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Derived
    num_classes: int = 8  # WhaleSounds has 8 raw labels total (7 whale classes + unidentified)


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()
