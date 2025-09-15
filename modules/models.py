from __future__ import annotations

import torch
import torch.nn as nn

from typing import Optional

"""
A simple MLP classifier on top of text embeddings.
"""


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridTextCUIClassifier(nn.Module):
    """Hybrid text + concept (CUI) classifier with attention pooling over CUIs.

    Expects forward(text_emb, concept_ids, concept_mask)
    """

    def __init__(
        self,
        text_dim: int,
        cui_vocab_size: int,
        cui_emb_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        concept_dropout: float = 0.05,
        pad_id: int = 0,
        attn_dim: int | None = None,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.cui_emb = nn.Embedding(cui_vocab_size, cui_emb_dim, padding_idx=pad_id)
        self.concept_dropout = nn.Dropout(concept_dropout)

        # Attention setup: project text and CUI embeddings to a shared space
        self.attn_dim = int(attn_dim) if attn_dim is not None else max(64, min(256, cui_emb_dim))
        self.text_to_attn = nn.Linear(text_dim, self.attn_dim)
        self.cui_to_attn = nn.Linear(cui_emb_dim, self.attn_dim)
        self.attn = nn.MultiheadAttention(self.attn_dim, num_heads=attn_heads, batch_first=True)
        self.context_out = nn.Linear(self.attn_dim, cui_emb_dim)

        fused_dim = text_dim + cui_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def attend_concepts(self, text_emb: torch.Tensor, cui_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        text_emb: [B, D_t]
        cui_emb: [B, Lc, D_c]
        mask:    [B, Lc] (1 for real, 0 for pad)
        Returns context vector [B, D_c] after attention and projection.
        """
        # If all sequences have zero valid CUIs, return zeros directly
        if mask.numel() == 0 or (mask.sum(dim=1) == 0).all():
            return torch.zeros(text_emb.size(0), self.cui_emb.embedding_dim, device=text_emb.device, dtype=text_emb.dtype)

        # Project to attention space
        q = self.text_to_attn(text_emb).unsqueeze(1)  # [B,1,D_a]
        k = self.cui_to_attn(cui_emb)  # [B,Lc,D_a]
        v = k  # use same proj for V
        # key_padding_mask: True for pads
        key_padding_mask = (mask == 0)  # [B, Lc]
        # MultiheadAttention expects (B, S, E) with batch_first=True
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)  # [B,1,D_a]
        ctx = attn_out.squeeze(1)  # [B,D_a]
        # Guard against NaNs if a row had all positions masked
        ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        ctx_cui_space = self.context_out(ctx)  # [B,D_c]
        return ctx_cui_space

    def forward(self, text_emb: torch.Tensor, concept_ids: Optional[torch.Tensor] = None, concept_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if concept_ids is not None and concept_mask is not None:
            cui_emb = self.cui_emb(concept_ids)  # [B,Lc,D_c]
            ctx = self.attend_concepts(text_emb, cui_emb, concept_mask)
            ctx = self.concept_dropout(ctx)
            fused = torch.cat([text_emb, ctx], dim=-1)
        else:
            fused = torch.cat([text_emb, torch.zeros(text_emb.size(0), self.cui_emb.embedding_dim, device=text_emb.device)], dim=-1)
        return self.classifier(fused)
