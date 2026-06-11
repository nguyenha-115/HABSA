from __future__ import annotations

from collections import defaultdict

import torch


class ConceptAttribution:
    """Groups token attribution by ontology concept and supports perturbation SHAP."""

    def __init__(self, model, ontology):
        self.model = model
        self.ontology = ontology

    def aggregate(
        self,
        token_scores: torch.Tensor,
        token_concept_ids: torch.Tensor,
    ) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        for row in range(token_scores.size(0)):
            grouped: dict[str, list[float]] = defaultdict(list)
            for position, concept_id in enumerate(token_concept_ids[row].tolist()):
                if concept_id >= 0:
                    grouped[self.ontology.names[concept_id]].append(
                        float(token_scores[row, position].item())
                    )
            results.append(
                {
                    concept: sum(values) / len(values)
                    for concept, values in grouped.items()
                }
            )
        return results

    @torch.no_grad()
    def perturbation_values(self, batch: dict) -> list[dict[str, float]]:
        self.model.eval()
        original = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            aspect_mask=batch["aspect_mask"],
            concept_id=batch["concept_id"],
            main_concept_id=batch.get("main_concept_id"),
            dependency=batch.get("dependency"),
        )["logits"]
        target = original.argmax(dim=-1)
        original_score = original.gather(1, target.unsqueeze(1)).squeeze(1)
        results: list[dict[str, float]] = []

        for row in range(batch["input_ids"].size(0)):
            values: dict[str, float] = {}
            concept_ids = set(
                value
                for value in batch["token_concept_ids"][row].tolist()
                if value >= 0
            )
            for concept_id in concept_ids:
                perturbed_ids = batch["input_ids"][row : row + 1].clone()
                positions = batch["token_concept_ids"][row : row + 1] == concept_id
                perturbed_ids[positions] = 0
                output = self.model(
                    input_ids=perturbed_ids,
                    attention_mask=batch["attention_mask"][row : row + 1],
                    aspect_mask=batch["aspect_mask"][row : row + 1],
                    concept_id=batch["concept_id"][row : row + 1],
                    main_concept_id=(
                        batch["main_concept_id"][row : row + 1]
                        if "main_concept_id" in batch
                        else None
                    ),
                    dependency=(
                        batch["dependency"][row : row + 1]
                        if "dependency" in batch
                        else None
                    ),
                )["logits"]
                score = output[0, target[row]]
                values[self.ontology.names[concept_id]] = float(
                    (original_score[row] - score).item()
                )
            results.append(values)
        return results
