from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleResult:
    rule: str
    concept: str
    value: bool
    evidence: tuple[str, ...] = ()


class SWRLRuleEngine:
    """Differentiable-model companion for the SWRL rules in the specification."""

    def __init__(self, ontology):
        self.ontology = ontology

    def infer_text(self, tokens: list[str]) -> list[RuleResult]:
        return [
            RuleResult("implicitAspect", concept, True, tuple(tokens))
            for concept in self.ontology.infer_implicit(tokens)
        ]

    def evaluate_sentiments(self, scores: dict[str, float]) -> list[RuleResult]:
        results: list[RuleResult] = []
        for parent in self.ontology.names:
            children = self.ontology.children(parent)
            present = [child for child in children if child in scores]
            if present and all(scores[child] < 0 for child in present):
                results.append(
                    RuleResult(
                        "allSubAspectsNegative",
                        parent,
                        scores.get(parent, 0.0) <= 0,
                        tuple(present),
                    )
                )
        for name, concept in self.ontology.concepts.items():
            if concept.is_critical and scores.get(name, 0.0) < 0:
                results.append(RuleResult("criticalNegativeHighImpact", name, True))
        for left, relation, right in self.ontology.triples():
            if relation == "contradicts" and left in scores and right in scores:
                results.append(
                    RuleResult(
                        "contradictorySentiments",
                        left,
                        abs(scores[left] - scores[right]) >= 1.0,
                        (right,),
                    )
                )
        return results
