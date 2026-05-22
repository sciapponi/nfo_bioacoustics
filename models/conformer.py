"""
Simple CNN for bioacoustic classification - CUDA stable version.

Architecture: Simple 1D CNN Backbone -> Classification Head
"""

import torch
import torch.nn as nn
from .fno import FNOStack
from typing import Optional


class Conv1DBlock(nn.Module):
    """Simple 1D convolutional block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (batch, in_channels, seq_len)"""
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class TinyCNN(nn.Module):
    """
    Minimal 1D CNN for bioacoustic classification.
    
    Architecture:
    - FNO Featurizer (variable-rate audio handling)
    - 1D CNN Backbone (no striding)
    - Unidirectional LSTM for temporal aggregation
    - Classification head
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        fno_in_channels: int = 1,
        fno_out_channels: int = 16,
        fno_n_modes: int = 64,
        fno_n_layers: int = 3,
        cnn_channels: int = 32,
        cnn_num_blocks: int = 1,
        cnn_kernel_size: int = 3,
        dropout: float = 0.1,
        sample_rate: float = 16000.0,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
    ):
        super().__init__()
        
        # FNO featurizer for variable-rate audio
        self.fno = FNOStack(
            in_channels=fno_in_channels,
            out_channels=fno_out_channels,
            n_layers=fno_n_layers,
            n_modes=fno_n_modes,
            sample_rate=sample_rate,
        )
        
        # Project FNO output to CNN input
        self.fno_to_cnn = nn.Conv1d(fno_out_channels, cnn_channels, kernel_size=1)
        
        # 1D CNN backbone — no striding, stride=1 throughout
        self.cnn_blocks = nn.ModuleList([
            Conv1DBlock(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=cnn_kernel_size,
                stride=1,
                dropout=dropout,
            )
            for _ in range(cnn_num_blocks)
        ])
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(cnn_channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        
        self.sample_rate = sample_rate
    
    def forward(self, x: torch.Tensor, sample_rate: Optional[float] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.fno(x, sample_rate=sample_rate)
        x = self.fno_to_cnn(x)
        
        for block in self.cnn_blocks:
            x = block(x)
        
        # CRITICAL FIX: Global Average Pooling over time dimension (dim=-1)
        # Squeezes (batch, channels, seq_len) -> (batch, channels)
        # This completely un-links the sequence length from the classification head!
        x = torch.mean(x, dim=-1)
        
        logits = self.head(x)
        return logits

class PureFNOClassifier(nn.Module):
    """
    Pure 1D Fourier Neural Operator for discretization-invariant bioacoustic classification.
    
    Architecture:
    - FNO Stack: Extract resolution-invariant continuous spectral features.
    - Global Average Pooling: Squeezes the variable time dimension cleanly.
    - Linear Head: Projects invariant features to class logits.
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        fno_in_channels: int = 1,
        fno_out_channels: int = 16,
        fno_n_modes: int = 64,
        fno_n_layers: int = 3,
        dropout: float = 0.1,
        sample_rate: float = 250.0,
        # Kept in signature for backward compatibility with train.py setup
        cnn_channels: int = 32,
        cnn_num_blocks: int = 1,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
    ):
        super().__init__()
        
        # FNO featurizer for variable-rate audio
        self.fno = FNOStack(
            in_channels=fno_in_channels,
            out_channels=fno_out_channels,
            n_layers=fno_n_layers,
            n_modes=fno_n_modes,
            sample_rate=sample_rate,
        )
        
        # Classification head - takes fno_out_channels directly
        self.head = nn.Sequential(
            nn.Linear(fno_out_channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        
        self.sample_rate = sample_rate
    
    def forward(
        self,
        x: torch.Tensor,
        sample_rate: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Audio input of shape (batch, 1, seq_len) or (batch, seq_len)
            sample_rate: Current evaluation/batch sample rate.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure 3D input: (batch, channels, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 1. Extract resolution-invariant features
        x = self.fno(x, sample_rate=sample_rate)
        
        # 2. Global Average Pooling over the time/spatial dimension (dim=-1)
        # Squeezes (batch, fno_out_channels, seq_len) -> (batch, fno_out_channels)
        x = torch.mean(x, dim=-1)
        
        # 3. Classification
        logits = self.head(x)
        
        return logits
    
# Alias for backward compatibility
TinyConformer = TinyCNN
TinyConformer = PureFNOClassifier