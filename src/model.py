"""
TransformEEG architecture — adapted from MedMaxLab/transformeeg (MIT License).
Original: https://github.com/MedMaxLab/transformeeg/blob/main/AllFnc/models.py

Modified to expose encoder separately for SSL pretraining.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn


def _reset_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


class Conv1DEncoder(nn.Module):
    """Depthwise convolutional tokenizer from TransformEEG."""

    def __init__(self, Chans, D1=2, D2=2, kernLength1=5, kernLength2=5,
                 pool=4, stridePool=2, dropRate=0.2, ELUAlpha=0.1,
                 batchMomentum=0.25, seed=None):
        _reset_seed(seed)
        super().__init__()
        self.D1 = D1
        F1 = Chans * D1
        self.blck1 = nn.Sequential(
            nn.Conv1d(Chans, F1, kernLength1, padding='same', groups=Chans),
            nn.BatchNorm1d(F1, momentum=batchMomentum),
            nn.ELU(ELUAlpha),
        )
        self.pool1 = nn.AvgPool1d(pool, stridePool)
        self.drop1 = nn.Dropout1d(dropRate)
        self.blck2 = nn.Sequential(
            nn.Conv1d(F1, F1, kernLength2, padding='same', groups=F1),
            nn.BatchNorm1d(F1, momentum=batchMomentum),
            nn.ELU(ELUAlpha),
        )
        self.D2 = D2
        F2 = Chans * D1 * D2
        self.blck3 = nn.Sequential(
            nn.Conv1d(F1, F2, kernLength2, padding='same', groups=F1),
            nn.BatchNorm1d(F2, momentum=batchMomentum),
            nn.ELU(ELUAlpha),
        )
        self.pool2 = nn.AvgPool1d(pool, stridePool)
        self.drop2 = nn.Dropout1d(dropRate)
        self.blck4 = nn.Sequential(
            nn.Conv1d(F2, F2, kernLength2, padding='same', groups=F2),
            nn.BatchNorm1d(F2, momentum=batchMomentum),
            nn.ELU(ELUAlpha),
        )

    def forward(self, x):
        x1 = self.blck1(x)
        x1 = self.pool1(x1)
        x1 = self.drop1(x1)
        x2 = self.blck2(x1)
        x2 = x1 + x2
        x3 = self.blck3(x2)
        x3 = self.pool2(x3)
        x3 = self.drop2(x3)
        x4 = self.blck4(x3)
        x4 = x3 + x4
        return x4


class TransformEEGEncoder(nn.Module):
    """
    TransformEEG encoder (tokenizer + transformer), without classification head.
    Output: pooled feature vector [B, Features] suitable for SSL or fine-tuning.
    """

    def __init__(self, Chan=61, Features=244, seed=None):
        _reset_seed(seed)
        super().__init__()
        self.Chan = Chan
        self.feat_dim = Features

        self.token_gen = Conv1DEncoder(
            Chan, D1=2, D2=2, kernLength1=5, kernLength2=5,
            pool=4, stridePool=2, dropRate=0.2, ELUAlpha=0.1, batchMomentum=0.25,
        )
        _reset_seed(seed)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                Features, nhead=1, dim_feedforward=Features, dropout=0.2,
                activation=torch.nn.functional.hardswish, batch_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.pool_lay = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, C, T]
        x = self.token_gen(x)           # [B, Features, T']
        x = x.permute(0, 2, 1)         # [B, T', Features]
        x = self.transformer(x)         # [B, T', Features]
        x = x.permute(0, 2, 1)         # [B, Features, T']
        # MPS workaround for AdaptiveAvgPool1d
        dev = x.device
        x = self.pool_lay(x.cpu()).to(dev)  # [B, Features, 1]
        x = x.squeeze(-1)               # [B, Features]
        return x


class EEGClassifier(nn.Module):
    """TransformEEG encoder + classification head (matches original paper)."""

    def __init__(self, encoder: TransformEEGEncoder, nb_classes=2, seed=None):
        _reset_seed(seed)
        super().__init__()
        self.encoder = encoder
        F = encoder.feat_dim
        self.linear_lay = nn.Sequential(
            nn.Linear(F, F // 2 if F // 2 > 64 else 64),
            nn.LeakyReLU(),
            nn.Linear(F // 2 if F // 2 > 64 else 64, 1 if nb_classes <= 2 else nb_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.linear_lay(x)


def build_encoder(Chan=61, Features=None, seed=None) -> TransformEEGEncoder:
    if Features is None:
        # Features = Chan * D1 * D2 (output of Conv1DEncoder with D1=2, D2=2)
        Features = Chan * 4
    return TransformEEGEncoder(Chan=Chan, Features=Features, seed=seed)
