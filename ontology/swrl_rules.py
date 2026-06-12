from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ontology_manager import OntologyManager


@dataclass(frozen=True)
class RuleDefinition:
    name: str
    description: str
    swrl: str
    executable_in_owl: bool


@dataclass(frozen=True)
class RuleResult:
    rule: str
    concept: str
    satisfied: bool
    evidence: tuple[str, ...] = ()


RULE_DEFINITIONS = (
    RuleDefinition(
        name="criticalChildInfluencesParent",
        description=(
            "A critical child concept structurally influences its parent."
        ),
        swrl=(
            "hasSubAspect(?parent, ?child) ^ isCritical(?child, true) "
            "-> influences(?child, ?parent)"
        ),
        executable_in_owl=True,
    ),
    RuleDefinition(
        name="criticalNegativeHasHighImpact",
        description=(
            "A critical concept with a negative model score has high impact."
        ),
        swrl=(
            "Operational numeric rule evaluated after sentiment prediction; "
            "HermiT does not support the required swrlb:lessThan atom."
        ),
        executable_in_owl=False,
    ),
    RuleDefinition(
        name="allObservedChildrenNegative",
        description=(
            "When every observed child is negative, the parent should not be "
            "positive. Closed-world evaluation is required."
        ),
        swrl=(
            "Operational closed-world rule; not representable faithfully as "
            "a plain OWL 2 DL/SWRL universal quantification."
        ),
        executable_in_owl=False,
    ),
    RuleDefinition(
        name="contradictoryConceptSeparation",
        description=(
            "Concepts linked by contradicts should not receive nearly equal "
            "sentiment scores."
        ),
        swrl=(
            "Operational numeric rule evaluated by the training constraint "
            "layer."
        ),
        executable_in_owl=False,
    ),
)


class SWRLRuleEngine:
    """Closed-world companion for rules OWL reasoners cannot express safely."""

    def __init__(self, ontology: OntologyManager) -> None:
        self.ontology = ontology

    def evaluate_sentiments(
        self,
        scores: dict[str, float],
        *,
        contradiction_margin: float = 1.0,
    ) -> list[RuleResult]:
        results: list[RuleResult] = []

        for parent in self.ontology.names:
            observed_children = [
                child
                for child in self.ontology.children(parent)
                if child in scores
            ]
            if observed_children and all(
                scores[child] < 0 for child in observed_children
            ):
                results.append(
                    RuleResult(
                        rule="allObservedChildrenNegative",
                        concept=parent,
                        satisfied=scores.get(parent, 0.0) <= 0,
                        evidence=tuple(observed_children),
                    )
                )

        for name, concept in self.ontology.concepts.items():
            if concept.is_critical and scores.get(name, 0.0) < 0:
                results.append(
                    RuleResult(
                        rule="criticalNegativeHasHighImpact",
                        concept=name,
                        satisfied=True,
                        evidence=(f"score={scores[name]:.4f}",),
                    )
                )

        for left, relation, right in self.ontology.triples():
            if (
                relation == "contradicts"
                and left in scores
                and right in scores
            ):
                results.append(
                    RuleResult(
                        rule="contradictoryConceptSeparation",
                        concept=left,
                        satisfied=(
                            abs(scores[left] - scores[right])
                            >= contradiction_margin
                        ),
                        evidence=(right,),
                    )
                )
        return results
