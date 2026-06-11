from __future__ import annotations

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    """Text-to-ontology and ontology-to-text cross attention."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.text_to_ontology = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ontology_to_text = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.ontology_norm = nn.LayerNorm(hidden_dim)
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ontology_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        text_states: torch.Tensor,
        ontology_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        ontology_context, text_to_ontology_weights = self.text_to_ontology(
            text_states,
            ontology_states,
            ontology_states,
            need_weights=True,
            average_attn_weights=True,
        )
        fused_text = self.text_norm(text_states + ontology_context)
        fused_text = self.text_norm(fused_text + self.text_ffn(fused_text))

        text_context, ontology_to_text_weights = self.ontology_to_text(
            ontology_states,
            text_states,
            text_states,
            key_padding_mask=~attention_mask.bool(),
            need_weights=True,
            average_attn_weights=True,
        )
        fused_ontology = self.ontology_norm(ontology_states + text_context)
        fused_ontology = self.ontology_norm(
            fused_ontology + self.ontology_ffn(fused_ontology)
        )
        return fused_text, fused_ontology, {
            "text_to_ontology": text_to_ontology_weights,
            "ontology_to_text": ontology_to_text_weights,
        }
