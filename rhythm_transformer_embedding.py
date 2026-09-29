# 移植自 rhythm-transformer 專案 src/model/rhythm_transformer/embedding.py
# （2026-09-29）；僅清理行尾空白，ProSDD 的 Rhythm2 模型直接重用這裡的模組。

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:,:x.size(1)]
        return x


# class RhythmEmbedding(nn.Module):
#     def __init__(self, d_model: int, dropout: float):
#         super().__init__()
#         self.linear = nn.Linear(1, d_model)
#         self.pos = PositionalEncoding(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.layernorm = nn.LayerNorm(d_model, eps=1e-12)

#     def forward(self, duration: Tensor) -> Tensor:
#         # rhythm: [B, T]
#         x = duration.unsqueeze(-1) # [B, T, 1]
#         x = self.linear(x) # [B, T, D]
#         x = self.pos(x)
#         x = self.layernorm(x)
#         x = self.dropout(x)
#         return x


class RhythmEmbeddingWithDiff(nn.Module):
    # v4
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(2, d_model)
        self.pos = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, duration: Tensor) -> Tensor:
        # rhythm: [B, T]

        diff = torch.zeros_like(duration) # first diff is 0
        # diff = duration.clone() # firset diff is duration[0]
        # diff = n - (n - 1)
        diff[:, 1:] = duration[:, 1:] - duration[:, :-1]
        x = torch.cat([duration.unsqueeze(-1), diff.unsqueeze(-1)], dim=-1)  # [B, T, 2]
        x = self.linear(x) # [B, T, D]
        x = self.pos(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x

class RhythmEmbeddingWithVowelDiffAndMu(nn.Module):
    # v4.1
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(3, d_model)
        self.pos = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, duration: Tensor, deviation: Tensor, difference: Tensor) -> Tensor:
        # rhythm: [B, T]
        x = torch.stack([duration, deviation, difference], dim=-1) # [B, T, 3]
        x = self.linear(x) # [B, T, D]
        x = self.pos(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x

class RhythmEmbeddingWithVowelAndConsonantDiffAndMu(nn.Module):
    # v16
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(6, d_model)
        self.pos = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self,
                vowel_duration: Tensor,
                vowel_deviation: Tensor,
                vowel_difference: Tensor,
                consonant_duration: Tensor,
                consonant_deviation: Tensor,
                consonant_difference: Tensor
        ) -> Tensor:
        # rhythm: [B, T]

        # vowel
        x = torch.stack([vowel_duration, vowel_deviation, vowel_difference,consonant_duration,
                         consonant_deviation, consonant_difference], dim=-1)              # [B, T, 6]

        x = self.linear(x) # [B, T, D]
        x = self.pos(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x

class RhythmEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, *features: Tensor) -> Tensor:
        """
        features:多個 [B, T] tensor
            EX: [vowel_duration, vowel_deviation, vowel_difference,consonant_duration,consonant_deviation, consonant_difference] each shape [B, T]
        """
        x = torch.stack(features, dim=-1)  # [B, T, F]
        x = self.linear(x) # [B, T, D]
        x = self.pos(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x
