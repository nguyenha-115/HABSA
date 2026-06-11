from __future__ import annotations

import torch
from torch import nn


class BottomUpGNN(nn.Module):
    """Attention aggregation following ontology edges from leaves to root."""

    def __init__(
        self,
        hidden_dim: int,
        num_sentiments: int,
        ontology,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.ontology = ontology
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_sentiments)
        self.dropout = nn.Dropout(dropout)
        self.bottom_up_parents = [
            name
            for name in ontology.topological_order(bottom_up=True)
            if ontology.children(name)
        ]

    def forward(
        self, node_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = list(node_states.unbind(dim=1))
        batch_size, num_nodes, _ = node_states.shape
        contribution = node_states.new_zeros((batch_size, num_nodes))

        for parent_name in self.bottom_up_parents:
            parent_id = self.ontology.name_to_id[parent_name]
            child_names = self.ontology.children(parent_name)
            child_ids = [self.ontology.name_to_id[name] for name in child_names]
            children = torch.stack([states[index] for index in child_ids], dim=1)
            learned = self.attention(children).squeeze(-1)
            prior = node_states.new_tensor(
                [
                    self.ontology.concepts[name].sentiment_weight
                    * (2.0 if self.ontology.concepts[name].is_critical else 1.0)
                    for name in child_names
                ]
            ).clamp_min(1e-6)
            weights = torch.softmax(learned + prior.log().unsqueeze(0), dim=-1)
            message = (weights.unsqueeze(-1) * self.message(children)).sum(dim=1)
            states[parent_id] = self.update(
                self.dropout(message), states[parent_id]
            )
            for offset, child_id in enumerate(child_ids):
                contribution[:, child_id] = weights[:, offset]

        stacked = torch.stack(states, dim=1)
        logits = self.classifier(stacked)
        return stacked, logits, contribution
