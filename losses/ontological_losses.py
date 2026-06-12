from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def sentiment_values(num_classes: int, reference: torch.Tensor) -> torch.Tensor:
    if num_classes == 3:
        return reference.new_tensor([-1.0, 0.0, 1.0])
    if num_classes == 5:
        return reference.new_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    raise ValueError("Only 3-class and 5-class sentiment heads are supported")


def expected_sentiment(logits: torch.Tensor) -> torch.Tensor:
    values = sentiment_values(logits.size(-1), logits)
    return (torch.softmax(logits, dim=-1) * values).sum(-1)


class OntologicalLoss(nn.Module):
    def __init__(
        self,
        ontology,
        alpha_main: float = 0.4,
        beta_sub: float = 0.6,
        lambda_mono: float = 0.1,
        lambda_dom: float = 0.2,
        lambda_cons: float = 0.1,
        node_loss_weight: float = 0.15,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        epsilon: float = 0.05,
    ):
        super().__init__()
        if abs(alpha_main + beta_sub - 1.0) > 1e-6:
            raise ValueError("alpha_main + beta_sub must equal 1")
        self.ontology = ontology
        self.alpha_main = alpha_main
        self.beta_sub = beta_sub
        self.lambda_mono = lambda_mono
        self.lambda_dom = lambda_dom
        self.lambda_cons = lambda_cons
        self.node_loss_weight = node_loss_weight
        self.label_smoothing = label_smoothing
        self.epsilon = epsilon
        self.register_buffer("class_weights", class_weights)

    def _ce(self, logits, labels):
        return F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def monotonicity_loss(self, node_logits: torch.Tensor) -> torch.Tensor:
        scores = expected_sentiment(node_logits)
        penalties = []
        for parent in self.ontology.names:
            children = self.ontology.children(parent)
            if not children:
                continue
            child_ids = [self.ontology.name_to_id[name] for name in children]
            all_negative = (scores[:, child_ids] < 0).all(1)
            if all_negative.any():
                parent_scores = scores[:, self.ontology.name_to_id[parent]]
                penalties.append(F.relu(parent_scores[all_negative] + self.epsilon).mean())
        return torch.stack(penalties).mean() if penalties else scores.sum() * 0.0

    def dominance_loss(self, propagation_weights: torch.Tensor) -> torch.Tensor:
        penalties = []
        for name, concept in self.ontology.concepts.items():
            if not concept.is_critical or concept.parent is None:
                continue
            siblings = self.ontology.children(concept.parent)
            priors = propagation_weights.new_tensor(
                [
                    self.ontology.concepts[item].sentiment_weight
                    * (2.0 if self.ontology.concepts[item].is_critical else 1.0)
                    for item in siblings
                ]
            )
            target = priors[siblings.index(name)] / priors.sum()
            actual = propagation_weights[:, self.ontology.name_to_id[name]]
            penalties.append(F.mse_loss(actual, target.expand_as(actual)))
        return (
            torch.stack(penalties).mean()
            if penalties
            else propagation_weights.sum() * 0.0
        )

    def consistency_loss(self, node_logits: torch.Tensor) -> torch.Tensor:
        scores = expected_sentiment(node_logits)
        penalties = []
        for left, relation, right in self.ontology.triples():
            if relation == "contradicts":
                difference = (
                    scores[:, self.ontology.name_to_id[left]]
                    - scores[:, self.ontology.name_to_id[right]]
                ).abs()
                penalties.append(F.relu(1.0 - difference).mean())
        return torch.stack(penalties).mean() if penalties else scores.sum() * 0.0

    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor,
        node_targets: torch.Tensor | None = None,
        constraint_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        sub = self._ce(outputs["logits"], labels)
        main = self._ce(outputs["main_logits"], labels)
        hierarchical = self.alpha_main * main + self.beta_sub * sub
        node = hierarchical.new_zeros(())
        if node_targets is not None:
            batch, nodes, classes = outputs["node_logits"].shape
            node = F.cross_entropy(
                outputs["node_logits"].reshape(batch * nodes, classes),
                node_targets.reshape(batch * nodes),
                ignore_index=-100,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
        mono = self.monotonicity_loss(outputs["node_logits"])
        dom = self.dominance_loss(outputs["propagation_weights"])
        cons = self.consistency_loss(outputs["node_logits"])
        total = (
            hierarchical
            + self.node_loss_weight * node
            + constraint_scale
            * (
                self.lambda_mono * mono
                + self.lambda_dom * dom
                + self.lambda_cons * cons
            )
        )
        return {
            "total": total,
            "hierarchical": hierarchical,
            "node": node,
            "main": main,
            "sub": sub,
            "monotonicity": mono,
            "dominance": dom,
            "consistency": cons,
        }
