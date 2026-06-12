from __future__ import annotations

from collections import defaultdict

from losses import expected_sentiment


class StructuralExplainer:
    def __init__(self, ontology):
        self.ontology = ontology

    def active_path(self, outputs, concept: str, sample_index: int = 0):
        scores = expected_sentiment(outputs["node_logits"])[sample_index]
        weights = outputs["propagation_weights"][sample_index]
        return [
            {
                "concept": name,
                "sentiment": float(scores[self.ontology.name_to_id[name]]),
                "propagation_weight": float(weights[self.ontology.name_to_id[name]]),
            }
            for name in [concept, *self.ontology.ancestors(concept)]
        ]

    def propagation_tree(self, outputs, sample_index: int = 0):
        scores = expected_sentiment(outputs["node_logits"])[sample_index]
        weights = outputs["propagation_weights"][sample_index]

        def visit(name):
            concept_id = self.ontology.name_to_id[name]
            return {
                "concept": name,
                "sentiment": float(scores[concept_id]),
                "weight_in_parent": float(weights[concept_id]),
                "children": [visit(child) for child in self.ontology.children(name)],
            }

        return visit(self.ontology.root)


class GlobalExplanationReport:
    def __init__(self):
        self.scores = defaultdict(list)

    def add(self, values):
        for concept, value in values.items():
            self.scores[concept].append(abs(float(value)))

    def report(self, top_k=10):
        ranked = sorted(
            (
                (concept, sum(values) / len(values), len(values))
                for concept, values in self.scores.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {
                "rank": index + 1,
                "concept": concept,
                "mean_abs_importance": score,
                "count": count,
            }
            for index, (concept, score, count) in enumerate(ranked[:top_k])
        ]
