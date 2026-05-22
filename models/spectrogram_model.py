"""
2D FNO-based model for spectrogram classification.

Pure continuous operator architecture:
- Mel-Spectrogram frontend
- 2D FNO Stack (processes time-frequency as continuous surface)
- Global Average Pooling
- Linear classification head
"""
import torch
import torch.nn as nn
from typing import Optional
from .fno2d import FNO2DStack


try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except Exception:  # pragma: no cover - optional dependency fallback
    efficientnet_b0 = None
    EfficientNet_B0_Weights = None


class SpectrogramModel(nn.Module):
    """
    2D FNO model for spectrogram-based audio classification.
    
    Architecture:
    - Input: 2D mel-spectrograms (1, n_mels, time_steps)
    - FNO2D Stack: processes spectrograms as continuous surfaces
    - Global Average Pooling: (batch, channels, height, width) -> (batch, channels)
    - Linear head: classification
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        fno_in_channels: int = 1,  # mel-spectrogram is 1-channel
        fno_out_channels: int = 32,
        fno_n_layers: int = 2,
        fno_n_modes_time: int = 32,
        fno_n_modes_freq: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.fno_out_channels = fno_out_channels
        
        # 2D FNO Stack: processes (batch, 1, n_mels, time_steps) spectrograms
        self.fno = FNO2DStack(
            in_channels=fno_in_channels,
            out_channels=fno_out_channels,
            n_layers=fno_n_layers,
            n_modes_time=fno_n_modes_time,
            n_modes_freq=fno_n_modes_freq,
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(fno_out_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, spectrogram: torch.Tensor, sample_rate: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            spectrogram: (batch, 1, n_mels, time_steps) mel-spectrogram tensor
            sample_rate: ignored (for API compatibility)
        
        Returns:
            logits: (batch, num_classes)
        """
        # FNO processing
        x = self.fno(spectrogram, sample_rate=sample_rate)
        
        # Classification
        logits = self.head(x)
        
        return logits


class SpectrogramModelConv(nn.Module):
    """
    Conv-based variant of `SpectrogramModel` which uses pointwise 1x1 convolutions
    for channel mixing in the classification head instead of `nn.Linear`.

    This keeps the FNO featurizer but replaces the MLP head with conv layers
    that are applied across spatial positions and then pooled.
    """

    def __init__(
        self,
        num_classes: int = 8,
        fno_in_channels: int = 1,
        fno_out_channels: int = 32,
        fno_n_layers: int = 2,
        fno_n_modes_time: int = 32,
        fno_n_modes_freq: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.fno_out_channels = fno_out_channels

        # 2D FNO Stack
        self.fno = FNO2DStack(
            in_channels=fno_in_channels,
            out_channels=fno_out_channels,
            n_layers=fno_n_layers,
            n_modes_time=fno_n_modes_time,
            n_modes_freq=fno_n_modes_freq,
        )

        # Global Average Pooling right after the final FNO layer -> tiny vector
        # followed by a small MLP head. This drastically reduces parameter count
        # versus keeping spatial conv heads.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # GAP over (freq, time)
            nn.Flatten(),
            nn.Linear(fno_out_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, spectrogram: torch.Tensor, sample_rate: Optional[float] = None) -> torch.Tensor:
        x = self.fno(spectrogram, sample_rate=sample_rate)
        logits = self.head(x)
        return logits


class SpectrogramFNOTime(nn.Module):
    """
    Time-axis FNO variant.

    Pipeline:
      - 2D spectrogram input (B,1,F,T)
      - 2D conv with kernel (k_freq,1) to extract local frequency features per time-step
      - Pool across frequency dimension -> (B, conv_channels, T)
      - 1D FNO stack operating along time axis -> (B, fno_out_channels, T)
      - Global average pooling over time -> (B, fno_out_channels)
      - Small MLP head -> logits
    """

    def __init__(
        self,
        num_classes: int = 8,
        fno_in_channels: int = 16,
        conv_channels: int = 16,
        fno_out_channels: int = 32,
        fno_n_layers: int = 2,
        fno_n_modes_time: int = 32,
        kernel_freq: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv_freq = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=(kernel_freq, 1),
            padding=(kernel_freq // 2, 0),
            bias=False,
        )

        # 1D FNO along time; input channels = conv_channels
        from .fno import FNOStack

        self.fno_time = FNOStack(
            in_channels=conv_channels,
            out_channels=fno_out_channels,
            n_layers=fno_n_layers,
            n_modes=fno_n_modes_time,
        )

        # Head: GAP over time then MLP
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # expects (B, C, T) -> (B, C, 1)
            nn.Flatten(),
            nn.Linear(fno_out_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, spectrogram: torch.Tensor, sample_rate: Optional[float] = None) -> torch.Tensor:
        # spectrogram: (B, 1, F, T)
        x = self.conv_freq(spectrogram)  # (B, conv_channels, F, T)
        # Pool across frequency axis -> (B, conv_channels, T)
        x = x.mean(dim=2)
        # FNO expects (B, channels, seq_len)
        x = self.fno_time(x, sample_rate=sample_rate)  # (B, fno_out_channels, T)
        logits = self.head(x)  # (B, num_classes)
        return logits


class FNOConformer2D(nn.Module):
    """
    Hybrid 2D FNO + CNN model for spectrograms.
    
    Combines FNO for global structure with CNN for local patterns.
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        fno_in_channels: int = 1,
        fno_out_channels: int = 32,
        fno_n_layers: int = 2,
        fno_n_modes_time: int = 32,
        fno_n_modes_freq: int = 32,
        cnn_channels: int = 64,
        cnn_num_blocks: int = 2,
        cnn_kernel_size: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 2D FNO
        self.fno = FNO2DStack(
            in_channels=fno_in_channels,
            out_channels=fno_out_channels,
            n_layers=fno_n_layers,
            n_modes_time=fno_n_modes_time,
            n_modes_freq=fno_n_modes_freq,
        )
        
        # 2D CNN layers for local refinement
        self.cnn_blocks = nn.ModuleList()
        in_ch = fno_out_channels
        for i in range(cnn_num_blocks):
            self.cnn_blocks.append(nn.Sequential(
                nn.Conv2d(
                    in_ch, cnn_channels, kernel_size=cnn_kernel_size,
                    padding=cnn_kernel_size // 2, bias=False
                ),
                nn.BatchNorm2d(cnn_channels),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ))
            in_ch = cnn_channels
        
        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_ch, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, spectrogram: torch.Tensor, sample_rate: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            spectrogram: (batch, 1, n_mels, time_steps)
            sample_rate: ignored
        
        Returns:
            logits: (batch, num_classes)
        """
        # FNO: global structure
        x = self.fno(spectrogram, sample_rate=sample_rate)
        
        # CNN refinement: local patterns
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
        
        # Classification
        logits = self.head(x)
        
        return logits


class EfficientNetSpectrogram(nn.Module):
    """EfficientNet-B0 classifier for 2D mel-spectrograms."""

    def __init__(self, num_classes: int = 8, dropout: float = 0.2, pretrained: bool = False):
        super().__init__()

        if efficientnet_b0 is None:
            raise ImportError(
                "torchvision is required for EfficientNetSpectrogram. Install torchvision to use model_type='efficientnet'."
            )

        # Use the stock EfficientNet-B0 backbone and adapt 1-channel spectrograms to 3 channels.
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # Replace classifier head for the target number of classes.
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, spectrogram: torch.Tensor, sample_rate: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            spectrogram: (batch, 1, n_mels, time_steps)
            sample_rate: ignored

        Returns:
            logits: (batch, num_classes)
        """
        x = self.input_adapter(spectrogram)
        return self.backbone(x)
