"""
TransformEEG encoder + SelfEEG SSL wrapper.
Architecture from Del Pup et al. (2025), Neurocomputing.
"""

import torch
import torch.nn as nn


class DepthwiseConvTokenizer(nn.Module):
    """Channel-specific convolutional tokenizer from TransformEEG."""

    def __init__(self, n_channels: int = 61, temporal_kernel: int = 25, feat_dim: int = 244):
        super().__init__()
        self.depthwise = nn.Conv2d(
            1, n_channels, kernel_size=(n_channels, temporal_kernel),
            padding=(0, temporal_kernel // 2), groups=1
        )
        self.feat_dim = feat_dim
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.depthwise(x)  # [B, C, 1, T']
        x = x.squeeze(2)  # [B, C, T']
        return x


class TransformEEGEncoder(nn.Module):
    """
    TransformEEG encoder — convolutional tokenizer + transformer.
    Outputs pooled latent features suitable for SSL or classification.
    """

    def __init__(
        self,
        n_channels: int = 61,
        feat_dim: int = 244,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = DepthwiseConvTokenizer(n_channels, feat_dim=feat_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_channels, feat_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        tokens = self.tokenizer(x)  # [B, C, feat_dim] after adaptive pool
        # MPS doesn't support non-divisible adaptive_avg_pool1d — do on CPU, move back
        dev = tokens.device
        tokens = nn.functional.adaptive_avg_pool1d(tokens.cpu(), self.feat_dim).to(dev)
        tokens = tokens + self.pos_embed
        out = self.transformer(tokens)   # [B, C, feat_dim]
        out = self.norm(out)
        out = out.mean(dim=1)            # [B, feat_dim] — global average pool
        return out


class EEGClassifier(nn.Module):
    """Fine-tuning head: encoder + linear classifier."""

    def __init__(self, encoder: TransformEEGEncoder, n_classes: int = 2, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(encoder.feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def build_encoder(n_channels: int = 61, feat_dim: int = 244) -> TransformEEGEncoder:
    return TransformEEGEncoder(n_channels=n_channels, feat_dim=feat_dim)
