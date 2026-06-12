from __future__ import annotations

from collections import defaultdict

import torch

from losses import expected_sentiment, sentiment_values


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float | dict]:
    predictions = logits.argmax(-1)
    num_classes = logits.size(-1)
    per_class = {}
    f1_scores = []
    for class_id in range(num_classes):
        predicted = predictions == class_id
        actual = labels == class_id
        tp = (predicted & actual).sum().item()
        precision = tp / max(1, predicted.sum().item())
        recall = tp / max(1, actual.sum().item())
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        per_class[str(class_id)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(actual.sum().item()),
        }
    values = sentiment_values(num_classes, logits)
    return {
        "accuracy": (predictions == labels).float().mean().item(),
        "macro_f1": sum(f1_scores) / num_classes,
        "mae": (values[predictions] - values[labels]).abs().mean().item(),
        "per_class": per_class,
    }


def ontological_consistency(node_logits: torch.Tensor, ontology) -> float:
    scores = expected_sentiment(node_logits)
    valid = torch.ones(scores.size(0), dtype=torch.bool, device=scores.device)
    for parent in ontology.names:
        children = ontology.children(parent)
        if children:
            child_ids = [ontology.name_to_id[name] for name in children]
            all_negative = (scores[:, child_ids] < 0).all(1)
            valid &= ~(
                all_negative & (scores[:, ontology.name_to_id[parent]] > 0)
            )
    for left, relation, right in ontology.triples():
        if relation == "contradicts":
            difference = (
                scores[:, ontology.name_to_id[left]]
                - scores[:, ontology.name_to_id[right]]
            ).abs()
            valid &= difference >= 1.0
    return valid.float().mean().item()


def aopc(original_scores: torch.Tensor, perturbed_scores: torch.Tensor) -> float:
    if perturbed_scores.ndim == 1:
        perturbed_scores = perturbed_scores.unsqueeze(1)
    return (original_scores.unsqueeze(1) - perturbed_scores).mean().item()


def log_odds(original_probability, perturbed_probability) -> float:
    epsilon = 1e-7
    original = original_probability.clamp(epsilon, 1 - epsilon)
    perturbed = perturbed_probability.clamp(epsilon, 1 - epsilon)
    return (
        torch.log(original / (1 - original))
        - torch.log(perturbed / (1 - perturbed))
    ).mean().item()


class GlobalConceptImportance:
    def __init__(self):
        self.values = defaultdict(list)

    def update(self, concept_scores: dict[str, float]) -> None:
        for concept, score in concept_scores.items():
            self.values[concept].append(abs(float(score)))

    def compute(self) -> dict[str, float]:
        return dict(
            sorted(
                (
                    (concept, sum(values) / len(values))
                    for concept, values in self.values.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        )
