"""Data loading package."""

from .whale_dataset import get_whale_dataloader, WhaleSoundsDataset
from .bird_dataset import get_bird_dataloader, BirdCLEFDataset

__all__ = [
    "get_whale_dataloader",
    "WhaleSoundsDataset",
    "get_bird_dataloader",
    "BirdCLEFDataset",
]
