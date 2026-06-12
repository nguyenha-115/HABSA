from __future__ import annotations

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.text_to_ontology = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ontology_to_text = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.ontology_norm = nn.LayerNorm(hidden_dim)
        self.text_ffn = self._ffn(hidden_dim, dropout)
        self.ontology_ffn = self._ffn(hidden_dim, dropout)

    @staticmethod
    def _ffn(hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, text, ontology, attention_mask):
        text_context, text_weights = self.text_to_ontology(
            text, ontology, ontology, need_weights=True
        )
        fused_text = self.text_norm(text + text_context)
        fused_text = self.text_norm(fused_text + self.text_ffn(fused_text))
        ontology_context, ontology_weights = self.ontology_to_text(
            ontology,
            text,
            text,
            key_padding_mask=~attention_mask.bool(),
            need_weights=True,
        )
        fused_ontology = self.ontology_norm(ontology + ontology_context)
        fused_ontology = self.ontology_norm(
            fused_ontology + self.ontology_ffn(fused_ontology)
        )
        return fused_text, fused_ontology, {
            "text_to_ontology": text_weights,
            "ontology_to_text": ontology_weights,
        }
