"""Models package."""

from .fno import FNOStack
from .fno2d import FNO2DLayer, FNO2DStack
from .conformer import TinyCNN
from .spectrogram_model import SpectrogramModel, FNOConformer2D, EfficientNetSpectrogram, SpectrogramModelConv

# Aliases for backward compatibility
TinyConformer = TinyCNN

__all__ = [
    # 1D Models
    "FNOStack",
    "TinyCNN",
    "TinyConformer",  # Alias for backward compatibility
    # 2D Models
    "FNO2DLayer",
    "FNO2DStack",
    "SpectrogramModel",
    "FNOConformer2D",
    "SpectrogramModelConv",
    "SpectrogramFNOTime",
    "EfficientNetSpectrogram",
]
