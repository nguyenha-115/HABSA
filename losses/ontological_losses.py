from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


SENTIMENT_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0)


def expected_sentiment(logits: torch.Tensor) -> torch.Tensor:
    values = logits.new_tensor(SENTIMENT_VALUES)
    return (torch.softmax(logits, dim=-1) * values).sum(dim=-1)


class OntologicalLoss(nn.Module):
    def __init__(
        self,
        ontology,
        alpha_main: float = 0.6,
        beta_sub: float = 0.4,
        lambda_mono: float = 0.1,
        lambda_dom: float = 0.2,
        lambda_cons: float = 0.1,
        lambda_kge: float = 0.05,
        epsilon: float = 0.05,
    ):
        super().__init__()
        self.ontology = ontology
        self.alpha_main = alpha_main
        self.beta_sub = beta_sub
        self.lambda_mono = lambda_mono
        self.lambda_dom = lambda_dom
        self.lambda_cons = lambda_cons
        self.lambda_kge = lambda_kge
        self.epsilon = epsilon

    def monotonicity_loss(self, node_logits: torch.Tensor) -> torch.Tensor:
        scores = expected_sentiment(node_logits)
        penalties: list[torch.Tensor] = []
        for parent in self.ontology.names:
            children = self.ontology.children(parent)
            if not children:
                continue
            parent_id = self.ontology.name_to_id[parent]
            child_ids = [self.ontology.name_to_id[name] for name in children]
            child_scores = scores[:, child_ids]
            all_negative = (child_scores < 0).all(dim=1)
            violation = F.relu(scores[:, parent_id] + self.epsilon)
            if all_negative.any():
                penalties.append(violation[all_negative].mean())
        return torch.stack(penalties).mean() if penalties else scores.sum() * 0.0

    def dominance_loss(self, propagation_weights: torch.Tensor) -> torch.Tensor:
        penalties: list[torch.Tensor] = []
        for name, concept in self.ontology.concepts.items():
            if not concept.is_critical or concept.parent is None:
                continue
            siblings = self.ontology.children(concept.parent)
            priors = torch.tensor(
                [
                    self.ontology.concepts[item].sentiment_weight
                    * (2.0 if self.ontology.concepts[item].is_critical else 1.0)
                    for item in siblings
                ],
                device=propagation_weights.device,
                dtype=propagation_weights.dtype,
            )
            target = priors[siblings.index(name)] / priors.sum()
            actual = propagation_weights[:, self.ontology.name_to_id[name]]
            penalties.append(F.mse_loss(actual, target.expand_as(actual)))
        if penalties:
            return torch.stack(penalties).mean()
        return propagation_weights.sum() * 0.0

    def consistency_loss(self, node_logits: torch.Tensor) -> torch.Tensor:
        scores = expected_sentiment(node_logits)
        penalties: list[torch.Tensor] = []
        for left, relation, right in self.ontology.triples():
            if relation != "contradicts":
                continue
            left_id = self.ontology.name_to_id[left]
            right_id = self.ontology.name_to_id[right]
            penalties.append(F.relu(1.0 - (scores[:, left_id] - scores[:, right_id]).abs()).mean())
        return torch.stack(penalties).mean() if penalties else scores.sum() * 0.0

    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor,
        node_targets: torch.Tensor | None = None,
        constraint_scale: float = 1.0,
        kge_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        sub = F.cross_entropy(outputs["logits"], labels)
        main = F.cross_entropy(outputs["main_logits"], labels)
        hierarchical = self.alpha_main * main + self.beta_sub * sub

        # supervise tất cả ancestor nodes
        node_loss = hierarchical.new_zeros(())
        if node_targets is not None:
            B, N, C = outputs["node_logits"].shape
            node_loss = F.cross_entropy(
                outputs["node_logits"].view(B * N, C),
                node_targets.view(B * N),
                ignore_index=-100,
            )
            hierarchical = hierarchical + 0.3 * node_loss

        monotonicity = self.monotonicity_loss(outputs["node_logits"])
        dominance = self.dominance_loss(outputs["propagation_weights"])
        consistency = self.consistency_loss(outputs["node_logits"])
        if kge_loss is None:
            kge_loss = hierarchical.new_zeros(())
        total = (
            hierarchical
            + constraint_scale * self.lambda_mono * monotonicity
            + constraint_scale * self.lambda_dom * dominance
            + constraint_scale * self.lambda_cons * consistency
            + self.lambda_kge * kge_loss
        )
        return {
            "total": total,
            "hierarchical": hierarchical,
            "node": node_loss,
            "main": main,
            "sub": sub,
            "monotonicity": monotonicity,
            "dominance": dominance,
            "consistency": consistency,
            "kge": kge_loss,
        }
