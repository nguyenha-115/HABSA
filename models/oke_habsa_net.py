from __future__ import annotations

import torch
from torch import nn

from .embedding import (
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
        cfg = config.model
        hidden_dim = int(cfg.hidden_dim)
        relations = ontology.relation_names()
        self.relation_to_id = {name: index for index, name in enumerate(relations)}
        self.text_encoder = TextEncoder(
            vocab_size,
            int(cfg.embedding_dim),
            hidden_dim,
            float(cfg.dropout),
            str(cfg.text_backend),
            str(cfg.pretrained_model),
            bool(cfg.local_files_only),
        )
        self.dependency_encoder = DependencyGraphEncoder(hidden_dim, float(cfg.dropout))
        self.transe = TransEEmbedding(
            len(ontology.names), max(1, len(relations)), int(cfg.ontology_dim)
        )
        self.hierarchy_encoder = HierarchyEncoder(
            self.transe, int(cfg.ontology_dim), hidden_dim
        )
        self.fusion = CrossAttentionFusion(
            hidden_dim, int(cfg.num_heads), float(cfg.dropout)
        )
        self.aspect_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gnn = BottomUpGNN(
            hidden_dim, int(cfg.num_sentiments), ontology, float(cfg.dropout)
        )
        self.register_buffer(
            "depths",
            torch.tensor(ontology.concept_features()["depth"], dtype=torch.long),
        )

    def ontology_triples(self, device=None) -> torch.Tensor:
        return torch.tensor(
            [
                (
                    self.ontology.name_to_id[head],
                    self.relation_to_id[relation],
                    self.ontology.name_to_id[tail],
                )
                for head, relation, tail in self.ontology.triples()
            ],
            dtype=torch.long,
            device=device or self.depths.device,
        )

    def freeze_transe(self) -> None:
        for parameter in self.transe.parameters():
            parameter.requires_grad = False

    def _forward_states(
        self,
        text_states,
        attention_mask,
        aspect_mask,
        concept_id,
        main_concept_id,
        dependency,
    ):
        text_states = self.dependency_encoder(text_states, dependency)
        ontology_states = self.hierarchy_encoder(self.depths).unsqueeze(0)
        ontology_states = ontology_states.expand(text_states.size(0), -1, -1)
        fused_text, fused_nodes, attention = self.fusion(
            text_states, ontology_states, attention_mask
        )
        normalized = aspect_mask / aspect_mask.sum(1, keepdim=True).clamp_min(1.0)
        aspect_state = (fused_text * normalized.unsqueeze(-1)).sum(1)
        selector = torch.nn.functional.one_hot(
            concept_id, num_classes=len(self.ontology.names)
        ).to(fused_nodes.dtype)
        fused_nodes = fused_nodes + selector.unsqueeze(-1) * self.aspect_projection(
            aspect_state
        ).unsqueeze(1)
        node_states, node_logits, propagation_weights = self.gnn(fused_nodes)
        rows = torch.arange(text_states.size(0), device=text_states.device)
        main_concept_id = concept_id if main_concept_id is None else main_concept_id
        return {
            "logits": node_logits[rows, concept_id],
            "main_logits": node_logits[rows, main_concept_id],
            "node_logits": node_logits,
            "text_states": fused_text,
            "node_states": node_states,
            "cross_attention": attention,
            "propagation_weights": propagation_weights,
        }

    def forward(
        self,
        input_ids,
        attention_mask,
        aspect_mask,
        concept_id,
        main_concept_id=None,
        dependency=None,
        **_,
    ):
        return self._forward_states(
            self.text_encoder(input_ids, attention_mask),
            attention_mask,
            aspect_mask,
            concept_id,
            main_concept_id,
            dependency,
        )

    def forward_from_embeddings(
        self,
        embeddings,
        attention_mask,
        aspect_mask,
        concept_id,
        main_concept_id=None,
        dependency=None,
    ):
        return self._forward_states(
            self.text_encoder.encode_embeddings(embeddings, attention_mask),
            attention_mask,
            aspect_mask,
            concept_id,
            main_concept_id,
            dependency,
        )
