from __future__ import annotations

import torch
from torch import nn

from .embedders import (
    DependencyGraphEncoder,
    HierarchyEncoder,
    TextEncoder,
    TransEEmbedding,
)
from .fusion import CrossAttentionFusion
from .gnn_propagator import BottomUpGNN


class OKEHABSANet(nn.Module):
    def __init__(self, config, ontology, vocab_size: int):
        super().__init__()
        self.config = config
        self.ontology = ontology
        model_config = config.model
        hidden_dim = int(model_config.hidden_dim)
        ontology_dim = int(model_config.ontology_dim)
        relations = ontology.relation_names()
        self.relation_to_id = {name: index for index, name in enumerate(relations)}

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=int(model_config.embedding_dim),
            hidden_dim=hidden_dim,
            dropout=float(model_config.dropout),
            backend=str(model_config.text_backend),
            pretrained_model=str(model_config.pretrained_model),
            local_files_only=bool(model_config.local_files_only),
        )
        self.dependency_encoder = DependencyGraphEncoder(
            hidden_dim, float(model_config.dropout)
        )
        self.transe = TransEEmbedding(
            len(ontology.names), max(1, len(relations)), ontology_dim
        )
        self.hierarchy_encoder = HierarchyEncoder(
            self.transe, ontology_dim, hidden_dim
        )
        self.fusion = CrossAttentionFusion(
            hidden_dim,
            int(model_config.num_heads),
            float(model_config.dropout),
        )
        self.aspect_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gnn = BottomUpGNN(
            hidden_dim,
            int(model_config.num_sentiments),
            ontology,
            float(model_config.dropout),
        )
        features = ontology.concept_features()
        self.register_buffer("depths", torch.tensor(features["depth"], dtype=torch.long))
        self.register_buffer(
            "critical_mask", torch.tensor(features["critical"], dtype=torch.bool)
        )
        self.register_buffer(
            "sentiment_weights", torch.tensor(features["weight"], dtype=torch.float)
        )

    def ontology_triples(self, device: torch.device | None = None) -> torch.Tensor:
        triples = [
            (
                self.ontology.name_to_id[head],
                self.relation_to_id[relation],
                self.ontology.name_to_id[tail],
            )
            for head, relation, tail in self.ontology.triples()
        ]
        return torch.tensor(triples, dtype=torch.long, device=device or self.depths.device)

    def _forward_states(
        self,
        text_states: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
        concept_id: torch.Tensor,
        main_concept_id: torch.Tensor | None,
        dependency: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        text_states = self.dependency_encoder(text_states, dependency)
        batch_size = text_states.size(0)
        ontology_states = self.hierarchy_encoder(self.depths)
        ontology_states = ontology_states.unsqueeze(0).expand(batch_size, -1, -1)
        fused_text, fused_nodes, cross_attention = self.fusion(
            text_states, ontology_states, attention_mask
        )

        normalized_mask = aspect_mask / aspect_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        aspect_state = (fused_text * normalized_mask.unsqueeze(-1)).sum(dim=1)
        aspect_update = self.aspect_projection(aspect_state)
        selector = torch.nn.functional.one_hot(
            concept_id, num_classes=len(self.ontology.names)
        ).to(fused_nodes.dtype)
        fused_nodes = fused_nodes + selector.unsqueeze(-1) * aspect_update.unsqueeze(1)

        propagated, node_logits, propagation_weights = self.gnn(fused_nodes)
        rows = torch.arange(batch_size, device=node_logits.device)
        aspect_logits = node_logits[rows, concept_id]
        if main_concept_id is None:
            main_concept_id = concept_id
        main_logits = node_logits[rows, main_concept_id]
        return {
            "logits": aspect_logits,
            "main_logits": main_logits,
            "node_logits": node_logits,
            "text_states": fused_text,
            "node_states": propagated,
            "cross_attention": cross_attention,
            "propagation_weights": propagation_weights,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
        concept_id: torch.Tensor,
        main_concept_id: torch.Tensor | None = None,
        dependency: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        text_states = self.text_encoder(input_ids, attention_mask)
        return self._forward_states(
            text_states,
            attention_mask,
            aspect_mask,
            concept_id,
            main_concept_id,
            dependency,
        )

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
        concept_id: torch.Tensor,
        main_concept_id: torch.Tensor | None = None,
        dependency: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        text_states = self.text_encoder.encode_embeddings(embeddings, attention_mask)
        return self._forward_states(
            text_states,
            attention_mask,
            aspect_mask,
            concept_id,
            main_concept_id,
            dependency,
        )
