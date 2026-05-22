"""
Bird sound dataset loaders for transfer learning.

Supports BirdCLEF 2024 and other bird sound datasets.
Converts raw audio to 2D spectrograms.
"""
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
from pathlib import Path
import json
from datasets import load_dataset


class SpectrogramConverter:
    """Convert audio to mel-spectrograms with standardized parameters."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 50.0,
        f_max: float = 10000.0,
        mode: str = "linear",  # 'mel' or 'linear' (STFT)
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.mode = mode
    
    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        """
        Convert audio to mel-spectrogram.
        
        Args:
            audio: (samples,) numpy array
            
        Returns:
            mel_spec: (1, n_mels, time_steps) normalized to [0, 1]
        """
        if self.mode == "mel":
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=self.f_min,
                fmax=self.f_max,
            )

            # Convert to dB scale (power)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            spec_db = mel_spec_db
        else:
            # Linear spectrogram (STFT magnitude)
            stft = librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=True,
            )

            # Use magnitude (amplitude) spectrogram and convert to dB
            mag = np.abs(stft)
            spec_db = librosa.amplitude_to_db(mag, ref=np.max)
        
        # Normalize to [0, 1]
        spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)

        # Add channel dimension: (1, freq_bins, time_steps)
        mel_spec_tensor = torch.from_numpy(spec_norm[np.newaxis, ...]).float()
        
        return mel_spec_tensor


class BirdCLEFDataset(Dataset):
    """
    BirdCLEF 2024 dataset from HuggingFace.
    
    Loads bird audio clips and converts to spectrograms for 2D FNO.
    """
    
    def __init__(
        self,
        split: str = "train",  # train, validation, test
        target_duration: float = 5.0,  # seconds
        target_sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        cache_dir: Optional[str] = None,
        num_samples: Optional[int] = None,  # Limit dataset size for testing
    ):
        self.split = split
        self.target_duration = target_duration
        self.target_sr = target_sr
        self.target_samples = int(target_sr * target_duration)
        self.num_samples_limit = num_samples
        
        # Initialize spectrogram converter
        self.spec_converter = SpectrogramConverter(
            sample_rate=target_sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        
        print(f"Loading BirdCLEF 2024 {split} split...")
        
        # Load dataset from HuggingFace
        try:
            self.dataset = load_dataset(
                "Ben1892/BirdCLEF2024_species_names",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            
            # Limit dataset if requested
            if num_samples is not None:
                self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
            
            print(f"  Loaded {len(self.dataset)} samples")
            
            # Extract unique species (class names)
            species_set = set()
            for sample in self.dataset:
                if "species" in sample:
                    species_set.add(sample["species"])
            
            self.species_list = sorted(list(species_set))
            self.num_classes = len(self.species_list)
            self.species_to_idx = {sp: idx for idx, sp in enumerate(self.species_list)}
            
            print(f"  Found {self.num_classes} species")
            
        except Exception as e:
            print(f"  Failed to load BirdCLEF: {e}")
            print("  Falling back to synthetic bird dataset")
            self.dataset = None
            self._create_synthetic_birds(num_samples or 100)
    
    def _create_synthetic_birds(self, num_samples: int):
        """Create synthetic bird sounds for testing."""
        print(f"  Creating synthetic bird dataset with {num_samples} samples")
        
        self.species_list = [
            "Red-tailed Hawk", "American Crow", "Common Raven",
            "Northern Mockingbird", "Wood Thrush", "Song Sparrow",
            "American Robin", "Tufted Titmouse",
        ]
        self.num_classes = len(self.species_list)
        self.species_to_idx = {sp: idx for idx, sp in enumerate(self.species_list)}
        
        self.synthetic_data = []
        for i in range(num_samples):
            label = i % self.num_classes
            audio = self._generate_synthetic_bird_call(label)
            self.synthetic_data.append((audio, label, self.species_list[label]))
    
    def _generate_synthetic_bird_call(self, species_idx: int) -> np.ndarray:
        """Generate synthetic bird call for testing."""
        t = np.linspace(0, self.target_duration, self.target_samples)
        
        # Different frequency patterns for different species
        base_freq = 500 + species_idx * 300  # 500-2600 Hz range
        freq_variation = 200 * np.sin(2 * np.pi * 4 * t / self.target_duration)  # 4 Hz modulation
        freq = base_freq + freq_variation
        
        # Generate chirp
        phase = 2 * np.pi * np.cumsum(freq) / self.target_sr
        audio = 0.3 * np.sin(phase)
        
        # Add harmonics (overtones in bird songs)
        for harmonic in range(2, 4):
            harmonic_phase = 2 * np.pi * np.cumsum(freq * harmonic) / self.target_sr
            audio += (0.1 / harmonic) * np.sin(harmonic_phase)
        
        # Add envelope (amplitude modulation)
        envelope = np.sin(np.pi * t / self.target_duration) ** 2
        audio = audio * envelope
        
        # Add noise
        audio += 0.02 * np.random.randn(len(audio))
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio.astype(np.float32)
    
    def __len__(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            return len(self.synthetic_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            spectrogram: (1, n_mels, time_steps)
            label: class index
            species: species name
        """
        if self.dataset is not None:
            sample = self.dataset[idx]
            
            # Load audio from dataset
            if "audio" in sample:
                audio_data = sample["audio"]
                if isinstance(audio_data, dict):
                    # HuggingFace audio format
                    audio = np.array(audio_data["array"], dtype=np.float32)
                    sr = audio_data.get("sampling_rate", self.target_sr)
                else:
                    audio = audio_data
                    sr = self.target_sr
            else:
                # Synthetic fallback
                audio = self._generate_synthetic_bird_call(idx % self.num_classes)
                sr = self.target_sr
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Trim or pad to target length
            if len(audio) >= self.target_samples:
                audio = audio[:self.target_samples]
            else:
                audio = np.pad(audio, (0, self.target_samples - len(audio)), mode='constant')
            
            # Get label
            species = sample.get("species", self.species_list[idx % self.num_classes])
            label = self.species_to_idx.get(species, idx % self.num_classes)
            
        else:
            # Use synthetic data
            audio, label, species = self.synthetic_data[idx]
        
        # Convert to spectrogram
        spectrogram = self.spec_converter(audio)
        
        return {
            "spectrogram": spectrogram,  # (1, n_mels, time_steps)
            "label": torch.tensor(label, dtype=torch.long),
            "species": species,
            "audio": torch.from_numpy(audio),
        }


def get_bird_dataloader(
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 0,
    target_duration: float = 5.0,
    target_sr: int = 16000,
    shuffle: bool = True,
    num_samples: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for bird sounds.
    
    Args:
        split: train, validation, or test
        batch_size: batch size
        num_workers: number of workers for data loading
        target_duration: audio duration in seconds
        target_sr: target sample rate
        shuffle: shuffle dataset
        num_samples: limit dataset size (for testing)
    
    Returns:
        DataLoader with spectrograms
    """
    dataset = BirdCLEFDataset(
        split=split,
        target_duration=target_duration,
        target_sr=target_sr,
        num_samples=num_samples,
    )
    
    def collate_fn(batch):
        """Custom collate to handle spectrograms of potentially different sizes."""
        spectrograms = []
        labels = []
        species_list = []
        
        for item in batch:
            spec = item["spectrogram"]  # (1, n_mels, time_steps)
            
            # Pad/trim to consistent size if needed
            # Most should be similar due to target_duration
            spectrograms.append(spec)
            labels.append(item["label"])
            species_list.append(item["species"])
        
        # Stack spectrograms into (batch, 1, n_mels, time_steps)
        spectrograms = torch.stack(spectrograms, dim=0)
        labels = torch.stack(labels)  # (batch,)
        
        return {
            "spectrograms": spectrograms,
            "labels": labels,
            "species": species_list,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


class LocalBirdDataset(Dataset):
    """Load bird audio from a local directory structured as:
    root_dir/Train/<class_name>/*.wav
    root_dir/Validation/<class_name>/*.wav
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        target_duration: float = 5.0,
        target_sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_samples: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_duration = target_duration
        self.target_sr = target_sr
        self.target_samples = int(target_sr * target_duration)

        self.spec_converter = SpectrogramConverter(
            sample_rate=target_sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise RuntimeError(f"Split directory not found: {split_dir}")

        # Discover class subfolders
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        class_dirs = sorted(class_dirs, key=lambda p: p.name)
        self.class_names = [p.name for p in class_dirs]
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Collect files
        self.files = []  # list of (path, label)
        for cls_dir in class_dirs:
            label = self.class_to_idx[cls_dir.name]
            for wav in cls_dir.glob("*.wav"):
                self.files.append((wav, label))

        if num_samples is not None:
            self.files = self.files[:num_samples]

        print(f"Loaded LocalBirdDataset: split={split}, samples={len(self.files)}, classes={len(self.class_names)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label = self.files[idx]

        # Load audio using librosa to get numpy array
        audio, sr = librosa.load(str(path), sr=None, mono=True)
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

        # Trim/pad
        if len(audio) >= self.target_samples:
            audio = audio[: self.target_samples]
        else:
            audio = np.pad(audio, (0, self.target_samples - len(audio)), mode="constant")

        spectrogram = self.spec_converter(audio)

        return {
            "spectrogram": spectrogram,
            "label": torch.tensor(label, dtype=torch.long),
            "species": self.class_names[label],
            "audio": torch.from_numpy(audio.astype(np.float32)),
        }


def get_local_bird_dataloader(
    root_dir: str,
    split: str = "Train",
    batch_size: int = 32,
    num_workers: int = 0,
    target_duration: float = 5.0,
    target_sr: int = 16000,
    shuffle: bool = True,
    num_samples: Optional[int] = None,
):
    dataset = LocalBirdDataset(
        root_dir=root_dir,
        split=split,
        target_duration=target_duration,
        target_sr=target_sr,
        num_samples=num_samples,
    )

    def collate_fn(batch):
        spectrograms = [item["spectrogram"] for item in batch]
        labels = [item["label"] for item in batch]
        species = [item["species"] for item in batch]
        spectrograms = torch.stack(spectrograms, dim=0)
        labels = torch.stack(labels)
        return {"spectrograms": spectrograms, "labels": labels, "species": species}

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn), dataset
