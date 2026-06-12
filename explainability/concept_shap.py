from __future__ import annotations

from collections import defaultdict

import torch


class ConceptAttribution:
    """Concept aggregation and perturbation attribution (SHAP-style approximation)."""

    def __init__(self, model, ontology):
        self.model = model
        self.ontology = ontology

    def aggregate(self, token_scores, token_concept_ids):
        results = []
        for row in range(token_scores.size(0)):
            grouped = defaultdict(list)
            for position, concept_id in enumerate(token_concept_ids[row].tolist()):
                if concept_id >= 0:
                    grouped[self.ontology.names[concept_id]].append(
                        float(token_scores[row, position])
                    )
            results.append(
                {
                    concept: sum(values) / len(values)
                    for concept, values in grouped.items()
                }
            )
        return results

    @torch.no_grad()
    def perturbation_values(self, batch):
        original = self.model(**batch)["logits"]
        target = original.argmax(-1)
        results = []
        for row in range(len(target)):
            values = {}
            for concept_id in {
                value for value in batch["token_concept_ids"][row].tolist() if value >= 0
            }:
                perturbed = {
                    key: value[row : row + 1]
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }
                perturbed["input_ids"] = perturbed["input_ids"].clone()
                mask = batch["token_concept_ids"][row : row + 1] == concept_id
                perturbed["input_ids"][mask] = 0
                score = self.model(**perturbed)["logits"][0, target[row]]
                values[self.ontology.names[concept_id]] = float(
                    original[row, target[row]] - score
                )
            results.append(values)
        return results
