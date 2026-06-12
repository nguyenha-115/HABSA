from __future__ import annotations

import torch
from torch import nn


class BottomUpGNN(nn.Module):
    """Propagate child representations to parents using learned and ontology priors."""

    def __init__(self, hidden_dim: int, num_sentiments: int, ontology, dropout: float):
        super().__init__()
        self.ontology = ontology
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_sentiments)
        self.dropout = nn.Dropout(dropout)
        self.parents = [
            name
            for name in ontology.topological_order(bottom_up=True)
            if ontology.children(name)
        ]

    def forward(self, node_states: torch.Tensor):
        states = list(node_states.unbind(1))
        batch_size, num_nodes, _ = node_states.shape
        contribution = node_states.new_zeros((batch_size, num_nodes))
        for parent in self.parents:
            parent_id = self.ontology.name_to_id[parent]
            children = self.ontology.children(parent)
            child_ids = [self.ontology.name_to_id[name] for name in children]
            child_states = torch.stack([states[index] for index in child_ids], dim=1)
            learned = self.attention(child_states).squeeze(-1)
            priors = node_states.new_tensor(
                [
                    self.ontology.concepts[name].sentiment_weight
                    * (2.0 if self.ontology.concepts[name].is_critical else 1.0)
                    for name in children
                ]
            )
            weights = torch.softmax(learned + priors.clamp_min(1e-6).log(), dim=-1)
            message = (
                weights.unsqueeze(-1) * self.message(child_states)
            ).sum(dim=1)
            states[parent_id] = self.update(
                self.dropout(message), states[parent_id]
            )
            for offset, child_id in enumerate(child_ids):
                contribution[:, child_id] = weights[:, offset]
        stacked = torch.stack(states, dim=1)
        return stacked, self.classifier(stacked), contribution
