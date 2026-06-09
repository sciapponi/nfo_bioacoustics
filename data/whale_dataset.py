"""
Data loading utilities for bioacoustic datasets.

Supports: WhaleSounds (direct .npy files from HuggingFace) with fallback to synthetic
"""

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import librosa
from pathlib import Path
from huggingface_hub import hf_hub_download
import io


class SyntheticWhaleDataset(Dataset):
    """
    Synthetic whale sound dataset for testing when real data unavailable.
    Generates synthetic audio with different frequency patterns for each class.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        target_duration: float = 4.0,
        target_sr: int = 16000,
        num_classes: int = 8,
        ignore_class_7: bool = True
    ):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
        self.ignore_class_7 = ignore_class_7
        
        # Original class names
        all_class_names = [
            "Blue Whale",
            "Fin Whale", 
            "Humpback Whale",
            "Right Whale",
            "Sperm Whale",
            "Beluga Whale",
            "Narwhal",
            "Dolphin", # Class 7
        ][:num_classes]

        # Filter out class 7 if requested
        if self.ignore_class_7 and len(all_class_names) > 7:
            self.class_names = [name for i, name in enumerate(all_class_names) if i != 7]
            self.num_classes = len(self.class_names)
            # Map valid items
            self.valid_indices = [i for i in range(num_samples) if (i % num_classes) != 7]
            self.num_samples = len(self.valid_indices)
        else:
            self.class_names = all_class_names
            self.num_classes = len(all_class_names)
            self.valid_indices = list(range(num_samples))
            self.num_samples = num_samples
        
        print(f"Using synthetic whale dataset: {self.num_samples} samples, {self.num_classes} classes")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        """
        Returns:
            audio: (seq_len,) tensor with synthetic whale call
            label: class index
            sample_rate: sample rate
        """
        # Map back to the pre-filtered sequence index
        actual_idx = self.valid_indices[idx]
        
        # Generate synthetic whale call based on original class logic
        # Use the dataset's `self.num_classes` which already reflects whether
        # class 7 was removed. This ensures labels are in range [0, self.num_classes-1].
        label = actual_idx % self.num_classes
        t = np.linspace(0, self.target_duration, self.target_samples)
        
        # Different frequency modulation for each class
        freq_base = 50 + label * 100  # 50-750 Hz range
        
        # Create frequency sweep (chirp-like whale call)
        freq_sweep = freq_base + label * 50 * np.sin(2 * np.pi * t / self.target_duration)
        
        # Generate phase and audio
        phase = 2 * np.pi * np.cumsum(freq_sweep) / self.target_sr
        audio = 0.3 * np.sin(phase)
        
        # Add harmonics (simulate whale overtones)
        for harmonic in range(2, 4):
            harmonic_phase = 2 * np.pi * np.cumsum(freq_sweep * harmonic) / self.target_sr
            audio += (0.1 / harmonic) * np.sin(harmonic_phase)
        
        # Add slight noise
        audio += 0.02 * np.random.randn(len(audio))
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        
        return audio_tensor, label, float(self.target_sr)


class WhaleSoundsDataset(Dataset):
    """
    WhaleSounds dataset from HuggingFace - loaded directly from .npy files.
    
    Bypasses the loading script by downloading raw numpy arrays directly.
    Supports optional time masking augmentation to prevent overfitting to
    specific temporal patterns in training data.
    """
    
    def __init__(
        self,
        split: str = "train",
        target_duration: float = 4.0,
        target_sr: int = 16000,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        ignore_class_7: bool = True,
        time_mask_prob: float = 0.3,
        time_mask_width_range: Tuple[int, int] = (50, 200),
    ):
        self.split = split
        self.target_duration = target_duration
        self.target_sr = target_sr
        self.normalize = normalize
        self.target_samples = int(target_sr * target_duration)
        self.time_mask_prob = time_mask_prob if split == "train" else 0.0
        self.time_mask_width_range = time_mask_width_range
        
        print(f"Loading WhaleSounds {split} split directly from HuggingFace...")
        
        # Whale class names (8 classes - 7 baleen whales + unidentified)
        self.idx_to_label = {
            0: "Blue Whale",
            1: "Fin Whale",
            2: "Humpback Whale",
            3: "Right Whale",
            4: "Sperm Whale",
            5: "Sei Whale",
            6: "Antarctic Minke Whale",
            7: "Unidentified",
        }
        
        if ignore_class_7:
            del self.idx_to_label[7]
            
        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        
        # Download and load feature matrix and labels
        try:
            print("  Downloading WhaleSounds_X.npy...")
            X_path = hf_hub_download(
                repo_id="monster-monash/WhaleSounds",
                filename="WhaleSounds_X.npy",
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            X = np.load(X_path)
            
            print("  Downloading WhaleSounds_y.npy...")
            y_path = hf_hub_download(
                repo_id="monster-monash/WhaleSounds",
                filename="WhaleSounds_y.npy",
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            y = np.load(y_path)
            
            print(f"  Loaded {len(X)} samples, {len(self.idx_to_label)} classes")
            
            # Load fold indices to split train/validation
            fold_indices = []
            for fold in range(5):
                fold_path = hf_hub_download(
                    repo_id="monster-monash/WhaleSounds",
                    filename=f"test_indices_fold_{fold}.txt",
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
                with open(fold_path, 'r') as f:
                    indices = [int(line.strip()) for line in f if line.strip()]
                    fold_indices.append(indices)
            
            # Use first fold as validation, rest as training
            val_indices = set(fold_indices[0])
            all_indices = np.arange(len(X))
            
            if split == "validation":
                self.indices = np.array([i for i in all_indices if i in val_indices])
            else:  # train
                self.indices = np.array([i for i in all_indices if i not in val_indices])
            
            # --- FILTER OUT CLASS 7 HERE ---
            if ignore_class_7:
                self.indices = np.array([i for i in self.indices if int(y[i]) != 7])
            
            self.X = X
            self.y = y
            
            print(f"  {split.capitalize()} set: {len(self.indices)} samples (Class 7 filtered out)")
            print(f"  Classes: {list(self.idx_to_label.values())}")
            
        except Exception as e:
            print(f"  ✗ Failed to load from HuggingFace: {e}")
            raise RuntimeError(
                f"Could not download WhaleSounds dataset files from HuggingFace. "
                f"This might be a temporary network issue. "
                f"Error: {e}"
            ) from e
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        sample_idx = self.indices[idx]
        features = self.X[sample_idx]
        
        if features.ndim == 1:
            features = features
        else:
            features = features.flatten()
        
        if len(features) != self.target_samples:
            indices = np.linspace(0, len(features) - 1, self.target_samples)
            features = np.interp(indices, np.arange(len(features)), features)
        
        # CRITICAL: Z-score normalization per file to handle dynamic range variation
        # Different hydrophones record at different absolute levels; this forces each
        # sample to have mean=0, std=1, highlighting relative vocal patterns over absolute scale
        mean_val = np.mean(features)
        std_val = np.std(features)
        if std_val > 1e-8:  # Avoid division by very small values
            features = (features - mean_val) / (std_val + 1e-8)
        
        if self.normalize:
            # Clamp to [-3, 3] range after z-score to prevent extreme outliers
            features = np.clip(features, -3.0, 3.0)
        
        # Apply time masking augmentation (SpecAugment-style for temporal domain)
        if self.time_mask_prob > 0.0 and np.random.random() < self.time_mask_prob:
            mask_width = np.random.randint(self.time_mask_width_range[0], self.time_mask_width_range[1])
            mask_start = np.random.randint(0, max(1, len(features) - mask_width))
            features[mask_start:mask_start + mask_width] = 0.0
        
        features_tensor = torch.from_numpy(features.astype(np.float32))
        if features_tensor.dim() == 1:
            features_tensor = features_tensor.unsqueeze(0)
        
        label = int(self.y[sample_idx])
        
        return features_tensor, label, float(self.target_sr)
    
    def get_class_name(self, idx: int) -> str:
        return self.idx_to_label.get(idx, f"Class {idx}")


class VariableRateAudioBatch:
    def __call__(self, batch):
        audios, labels, sample_rates = zip(*batch)
        audio_batch = torch.stack(audios)
        labels_batch = torch.tensor(labels, dtype=torch.long)
        sample_rates_batch = torch.tensor(sample_rates, dtype=torch.float32)
        
        return {
            "audio": audio_batch,
            "labels": labels_batch,
            "sample_rates": sample_rates_batch,
        }


def get_whale_dataloader(
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 0,
    target_duration: float = 4.0,
    target_sr: int = 16000,
    shuffle: bool = True,
    use_synthetic: bool = False,
    cache_dir: Optional[str] = None,
    ignore_class_7: bool = True,
    time_mask_prob: float = 0.3,
    time_mask_width_range: Tuple[int, int] = (50, 200),
) -> Tuple[DataLoader, Dataset]:
    """
    Create DataLoader for WhaleSounds dataset with fallback to synthetic data.
    
    Args:
        split: "train" or "validation"
        batch_size: Number of samples per batch
        time_mask_prob: Probability of applying time masking augmentation (train split only)
        time_mask_width_range: (min_width, max_width) in samples for random masking window
    """
    if use_synthetic:
        print("Using synthetic whale dataset")
        num_samples = 800 if split == "train" else 200
        dataset = SyntheticWhaleDataset(
            num_samples=num_samples,
            target_duration=target_duration,
            target_sr=target_sr,
            num_classes=8,
            ignore_class_7=ignore_class_7
        )
    else:
        try:
            dataset = WhaleSoundsDataset(
                split=split,
                target_duration=target_duration,
                target_sr=target_sr,
                cache_dir=cache_dir,
                ignore_class_7=ignore_class_7,
                time_mask_prob=time_mask_prob,
                time_mask_width_range=time_mask_width_range,
            )
        except Exception as e:
            print(f"Warning: Could not load real WhaleSounds dataset: {e}")
            print("Falling back to synthetic whale dataset for testing")
            num_samples = 800 if split == "train" else 200
            dataset = SyntheticWhaleDataset(
                num_samples=num_samples,
                target_duration=target_duration,
                target_sr=target_sr,
                num_classes=8,
                ignore_class_7=ignore_class_7
            )
            
    # The external "ClassFilteredDataset" wrapper is no longer needed 
    # since filtering is handled naturally within the dataset classes.
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle and split == "train",
        collate_fn=VariableRateAudioBatch(),
    )
    
    return dataloader, dataset