from __future__ import annotations

from collections import defaultdict

import torch

from losses import expected_sentiment


class StructuralExplainer:
    def __init__(self, ontology):
        self.ontology = ontology

    def propagation_tree(self, outputs: dict, sample_index: int = 0) -> dict:
        scores = expected_sentiment(outputs["node_logits"])[sample_index]
        weights = outputs["propagation_weights"][sample_index]

        def visit(name: str) -> dict:
            concept_id = self.ontology.name_to_id[name]
            children = self.ontology.children(name)
            return {
                "concept": name,
                "sentiment": float(scores[concept_id].item()),
                "weight_in_parent": float(weights[concept_id].item()),
                "critical": self.ontology.concepts[name].is_critical,
                "children": [visit(child) for child in children],
            }

        return visit(self.ontology.root)

    def active_path(self, outputs: dict, concept: str, sample_index: int = 0) -> list[dict]:
        scores = expected_sentiment(outputs["node_logits"])[sample_index]
        weights = outputs["propagation_weights"][sample_index]
        path = [concept] + self.ontology.ancestors(concept)
        return [
            {
                "concept": name,
                "sentiment": float(scores[self.ontology.name_to_id[name]].item()),
                "propagation_weight": float(
                    weights[self.ontology.name_to_id[name]].item()
                ),
            }
            for name in path
        ]

    def counterfactual(
        self,
        outputs: dict,
        concept: str,
        replacement_score: float,
        sample_index: int = 0,
    ) -> dict:
        original = expected_sentiment(outputs["node_logits"])[sample_index]
        weights = outputs["propagation_weights"][sample_index]
        updated = original.clone()
        concept_id = self.ontology.name_to_id[concept]
        updated[concept_id] = replacement_score
        delta = updated[concept_id] - original[concept_id]
        current = concept
        for parent in self.ontology.ancestors(concept):
            current_id = self.ontology.name_to_id[current]
            propagation_weight = weights[current_id]
            if propagation_weight <= 0:
                propagation_weight = original.new_tensor(
                    1.0 / max(1, len(self.ontology.children(parent)))
                )
            delta = delta * propagation_weight
            parent_id = self.ontology.name_to_id[parent]
            updated[parent_id] = original[parent_id] + delta
            current = parent
        root_id = self.ontology.name_to_id[self.ontology.root]
        return {
            "changed_concept": concept,
            "replacement_score": replacement_score,
            "original_root_score": float(original[root_id].item()),
            "counterfactual_root_score": float(updated[root_id].item()),
            "delta": float((updated[root_id] - original[root_id]).item()),
        }


class GlobalExplanationReport:
    def __init__(self):
        self.scores: dict[str, list[float]] = defaultdict(list)

    def add(self, concept_values: dict[str, float]) -> None:
        for concept, value in concept_values.items():
            self.scores[concept].append(abs(float(value)))

    def report(self, top_k: int = 10) -> list[dict]:
        ranked = sorted(
            (
                (concept, sum(values) / len(values), len(values))
                for concept, values in self.scores.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {"rank": index + 1, "concept": concept, "mean_abs_importance": score, "count": count}
            for index, (concept, score, count) in enumerate(ranked[:top_k])
        ]
