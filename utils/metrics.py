from __future__ import annotations

import math
from collections import defaultdict

import torch

from losses.ontological_losses import expected_sentiment


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    classes = sorted(set(labels.detach().cpu().tolist()) | set(predictions.detach().cpu().tolist()))
    f1_scores: list[float] = []
    for class_id in classes:
        predicted = predictions == class_id
        actual = labels == class_id
        true_positive = (predicted & actual).sum().item()
        precision = true_positive / max(1, predicted.sum().item())
        recall = true_positive / max(1, actual.sum().item())
        f1_scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    values = logits.new_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    predicted_score = values[predictions]
    actual_score = values[labels]
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "mae": (predicted_score - actual_score).abs().mean().item(),
    }


def ontological_consistency(node_logits: torch.Tensor, ontology) -> float:
    scores = expected_sentiment(node_logits)
    valid = torch.ones(scores.size(0), dtype=torch.bool, device=scores.device)
    for parent in ontology.names:
        children = ontology.children(parent)
        if not children:
            continue
        parent_id = ontology.name_to_id[parent]
        child_ids = [ontology.name_to_id[name] for name in children]
        all_negative = (scores[:, child_ids] < 0).all(dim=1)
        valid &= ~(all_negative & (scores[:, parent_id] > 0))
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


def log_odds(original_probability: torch.Tensor, perturbed_probability: torch.Tensor) -> float:
    epsilon = 1e-7
    original = original_probability.clamp(epsilon, 1 - epsilon)
    perturbed = perturbed_probability.clamp(epsilon, 1 - epsilon)
    odds_original = torch.log(original / (1 - original))
    odds_perturbed = torch.log(perturbed / (1 - perturbed))
    return (odds_original - odds_perturbed).mean().item()


class GlobalConceptImportance:
    def __init__(self):
        self.values: dict[str, list[float]] = defaultdict(list)

    def update(self, concept_scores: dict[str, float]) -> None:
        for concept, score in concept_scores.items():
            self.values[concept].append(abs(float(score)))

    def compute(self) -> dict[str, float]:
        return {
            concept: sum(values) / len(values)
            for concept, values in sorted(
                self.values.items(),
                key=lambda item: sum(item[1]) / len(item[1]),
                reverse=True,
            )
        }
