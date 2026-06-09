"""
SIREN-based frontend and Conv classifier.

Maps raw waveform samples to a continuous feature representation using
sinusoidal-activated 1x1 convolutions (SIREN-like), then applies a small
1D convolutional classifier on top.

Also includes FiLM-modulated SIREN architecture for cross-taxa transfer.
"""
import torch
import torch.nn as nn
import numpy as np


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SirenEncoder(nn.Module):
    """Simple SIREN-style per-sample encoder using 1x1 Conv1d blocks.

    Input: (B, seq_len) or (B,1,seq_len)
    Output: (B, feat_dim, seq_len)
    """

    def __init__(self, feat_dim: int = 64, hidden: int = 64, num_layers: int = 3, w0: float = 30.0):
        super().__init__()
        layers = []
        # first layer takes waveform + normalized time coordinate -> 2 channels
        layers.append(nn.Conv1d(2, hidden, kernel_size=1))
        layers.append(Sine(w0))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hidden, hidden, kernel_size=1))
            layers.append(Sine(w0))

        layers.append(nn.Conv1d(hidden, feat_dim, kernel_size=1))

        self.net = nn.Sequential(*layers)
        self._init_weights(w0=w0)

    def _init_weights(self, w0: float = 30.0) -> None:
        """SIREN-style initialization for 1x1 Conv1d layers.

        This keeps sine activations in a usable range as depth/width increase.
        """
        conv_layers = [module for module in self.net if isinstance(module, nn.Conv1d)]
        for idx, layer in enumerate(conv_layers):
            fan_in = layer.in_channels * layer.kernel_size[0]
            if idx == 0:
                bound = 1.0 / fan_in
            elif idx == len(conv_layers) - 1:
                bound = (6.0 / fan_in) ** 0.5 / w0
            else:
                bound = (6.0 / fan_in) ** 0.5 / w0
            nn.init.uniform_(layer.weight, -bound, bound)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize input shape to (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, _, T = x.shape

        # time coordinates normalized to [-1, 1]
        t = torch.linspace(-1.0, 1.0, steps=T, device=x.device, dtype=x.dtype)
        t = t.view(1, 1, T).expand(B, 1, T)

        # concatenate waveform and time: (B,2,T)
        inp = torch.cat([x, t], dim=1)

        return self.net(inp)


class SirenConvClassifier(nn.Module):
    """Classifier that consumes SIREN features.

    - SirenEncoder -> Conv1d blocks -> GAP -> MLP head
    """

    def __init__(
        self,
        num_classes: int = 8,
        feat_dim: int = 64,
        conv_channels: int = 64,
        conv_blocks: int = 3,
        siren_num_layers: int = 3,
        kernel_size: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = SirenEncoder(feat_dim=feat_dim, num_layers=siren_num_layers)

        blocks = []
        in_ch = feat_dim
        for _ in range(conv_blocks):
            blocks.append(nn.Conv1d(in_ch, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            blocks.append(nn.BatchNorm1d(conv_channels))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))
            in_ch = conv_channels

        self.conv_net = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(conv_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, sample_rate: float = None) -> torch.Tensor:
        # x: (B, T) or (B,1,T)
        feats = self.encoder(x)  # (B, feat_dim, T)
        x = self.conv_net(feats)  # (B, conv_channels, T)
        logits = self.head(x)     # (B, num_classes)
        return logits


class Modulator(nn.Module):
    """Lightweight 1D-CNN encoder that extracts condition vectors from raw audio.
    
    CRITICAL: First layer uses massive kernel (65 or 129 samples) to natively 
    contextualize ultra-low frequency baleen whale calls (10-40 Hz).
    At 250 Hz target_sr, a 65-sample kernel spans 260ms, capturing full cycles
    of infrasonic sweeps. Subsequent layers remain small to capture combinations.
    
    Output is fixed-size via Adaptive Average Pooling, making it sample-rate invariant.
    """
    
    def __init__(self, out_dim: int = 32, hidden_channels: int = 32):
        super().__init__()
        layers = []
        
        # MASSIVE first kernel to capture low-frequency whale sweeps
        # kernel_size=65 at 250Hz = 260ms window → sees full cycles of 15Hz calls
        # padding=32 preserves sequence length for subsequent layers
        layers.append(nn.Conv1d(1, hidden_channels, kernel_size=65, padding=32))
        layers.append(nn.GELU())
        
        # Subsequent layers remain small to efficiently combine features
        layers.append(nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1))
        layers.append(nn.GELU())
        
        layers.append(nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1))
        layers.append(nn.GELU())
        
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(hidden_channels * 4, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) or (B, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.net(x)  # (B, hidden_channels*4, T)
        x = self.pool(x)  # (B, hidden_channels*4, 1)
        x = x.squeeze(-1)  # (B, hidden_channels*4)
        x = self.linear(x)  # (B, out_dim)
        return x


class PositionalEncoding1D(nn.Module):
    """Maps 1D time coordinate to multi-frequency sinusoidal space for positional anchoring."""
    
    def __init__(self, num_freqs: int = 6):
        super().__init__()
        # Geometrically spaced frequencies: 2^0, 2^1, ..., 2^(L-1)
        self.register_buffer(
            'freqs',
            torch.pow(2.0, torch.arange(num_freqs).float())
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, T, 1) or (T,) time coordinate in [-1, 1]
        
        Returns:
            (B, T, 2*num_freqs) or (T, 2*num_freqs) with sin and cos encodings
        """
        # Expand time by frequency: t * freqs * pi
        args = t * self.freqs * np.pi  # (B, T, num_freqs)
        # Concatenate sin and cos
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, T, 2*num_freqs)


class FilmSirenLayer(nn.Module):
    """SIREN layer with FiLM (Feature-wise Linear Modulation) from the modulator."""
    
    def __init__(self, in_features: int, out_features: int, mod_dim: int, w0: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mod_scale = nn.Linear(mod_dim, out_features, bias=True)
        self.mod_shift = nn.Linear(mod_dim, out_features, bias=True)
        self.w0 = w0
        
        # SIREN-style initialization
        with torch.no_grad():
            bound = np.sqrt(6 / in_features) / w0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)
        
        # Initialize modulation layers neutrally (scale→1, shift→0)
        with torch.no_grad():
            nn.init.zeros_(self.mod_scale.weight)
            nn.init.ones_(self.mod_scale.bias)  # Scale starts at 1 (identity)
            nn.init.zeros_(self.mod_shift.weight)
            nn.init.zeros_(self.mod_shift.bias)  # Shift starts at 0 (identity)
    
    def forward(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_features) or (B, in_features)
        # modulation: (B, mod_dim)
        x = self.linear(x)  # (B, T, out_features) or (B, out_features)
        
        scale = self.mod_scale(modulation)  # (B, out_features)
        shift = self.mod_shift(modulation)  # (B, out_features)
        
        # Apply FiLM modulation
        if x.dim() == 3:
            # x is (B, T, out_features)
            scale = scale.unsqueeze(1)  # (B, 1, out_features)
            shift = shift.unsqueeze(1)  # (B, 1, out_features)
        
        x = scale * x + shift
        x = torch.sin(self.w0 * x)
        return x


class FilmSirenClassifier(nn.Module):
    """FiLM-modulated SIREN classifier with positional encoding and dual pooling.
    
    Architecture:
    1. Modulator (lightweight 1D-CNN) extracts condition vector from audio
    2. PositionalEncoding maps time coordinates to Fourier features
    3. Concatenate [raw_audio, positional_encoded_time] → feed to SIREN
    4. SIREN core processes combined signal, modulated by condition
    5. Dual pooling (mean + max) captures both structure and transients
    6. Classifier head (linear + GELU + dropout + linear) → species labels
    
    Key insight: The SIREN sees BOTH the raw audio stream AND the modulated
    time-positional context, allowing it to act as a dynamic feature processor
    rather than just a reconstruction generator.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        modulator_out_dim: int = 32,
        siren_hidden_dim: int = 256,
        num_siren_layers: int = 3,
        w0_first: float = 30.0,
        w0_hidden: float = 1.0,
        num_pe_freqs: int = 12,
    ):
        super().__init__()
        self.modulator = Modulator(out_dim=modulator_out_dim)
        self.pe = PositionalEncoding1D(num_freqs=num_pe_freqs)
        
        # Input dimension: 1 (raw waveform) + 2*num_pe_freqs (positional encoding)
        in_features = 1 + (2 * num_pe_freqs)
        
        # First SIREN layer (high w0 to catch high frequencies)
        self.siren_first = FilmSirenLayer(in_features, siren_hidden_dim, modulator_out_dim, w0=w0_first)
        
        # Hidden SIREN layers (w0=1.0)
        self.siren_hidden = nn.ModuleList([
            FilmSirenLayer(siren_hidden_dim, siren_hidden_dim, modulator_out_dim, w0=w0_hidden)
            for _ in range(num_siren_layers - 1)
        ])
        
        # Time-compressing block: Gradually stride down temporal dimension while preserving
        # local acoustic structure. Instead of global pooling all T timesteps, we compress
        # the timeline via strided convolutions, allowing selective temporal focus.
        self.time_compressor = nn.Sequential(
            # Input: (B, siren_hidden_dim, T) where T=1000 samples
            # First stride: compress by 8x (1000 → 125 timesteps)
            nn.Conv1d(siren_hidden_dim, siren_hidden_dim, kernel_size=17, stride=8, padding=8),
            nn.GELU(),
            nn.Dropout(0.1),
            # Second stride: compress by 4x (125 → 31 timesteps)
            nn.Conv1d(siren_hidden_dim, siren_hidden_dim, kernel_size=9, stride=4, padding=4),
            nn.GELU()
        )
        
        # Classifier head: Receives dual-pooled compressed features
        self.classifier = nn.Sequential(
            nn.Linear(siren_hidden_dim * 2, siren_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(siren_hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, sample_rate: float = None) -> torch.Tensor:
        """
        Args:
            x: (B, T) or (B, 1, T) raw waveform
            sample_rate: Unused, for API compatibility
        
        Returns:
            logits: (B, num_classes)
        """
        # Normalize input shape to (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        B, _, T = x.shape
        
        # 1. Extract global modulation context from raw audio (massive kernel captures low freqs)
        mod = self.modulator(x)  # (B, modulator_out_dim)
        
        # 2. Prepare Continuous Coordinate Input with Positional Encoding
        t = torch.linspace(-1.0, 1.0, steps=T, device=x.device, dtype=x.dtype)
        t = t.view(1, T, 1).expand(B, T, 1)  # (B, T, 1)
        
        # Pass time through positional encoding
        t_pe = self.pe(t)  # (B, T, 2*num_pe_freqs)
        
        # Transpose raw audio from (B, 1, T) to (B, T, 1)
        x_trans = x.transpose(1, 2)  # (B, T, 1)
        
        # Concatenate waveform + positional encoded coordinates
        inp = torch.cat([x_trans, t_pe], dim=-1)  # (B, T, 1 + 2*num_pe_freqs)
        
        # 3. Process through Modulated SIREN
        h = self.siren_first(inp, mod)  # (B, T, siren_hidden_dim)
        for siren_layer in self.siren_hidden:
            h = siren_layer(h, mod)  # (B, T, siren_hidden_dim)
        
        # 4. Temporal Compression via Strided Convolutions
        # Transpose from (B, T, siren_hidden_dim) to (B, siren_hidden_dim, T) for conv ops
        h_trans = h.transpose(1, 2)  # (B, siren_hidden_dim, T)
        
        # Apply strided time compressor to gradually reduce temporal dimension
        # while preserving local acoustic structure (no information loss via global pooling)
        h_compressed = self.time_compressor(h_trans)  # (B, siren_hidden_dim, T_compressed)
        
        # 5. Dual Pooling on Compressed Timeline
        # Mean captures overall structure, Max captures transient peaks
        h_mean = h_compressed.mean(dim=-1)  # (B, siren_hidden_dim)
        h_max = h_compressed.max(dim=-1)[0]  # (B, siren_hidden_dim)
        h_pool = torch.cat([h_mean, h_max], dim=-1)  # (B, 2*siren_hidden_dim)
        
        # 6. Classification with improved head
        logits = self.classifier(h_pool)  # (B, num_classes)
        return logits
