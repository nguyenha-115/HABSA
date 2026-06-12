from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from types import new_class
from typing import Iterable

from owlready2 import (
    AllDisjoint,
    DataProperty,
    FunctionalProperty,
    Imp,
    ObjectProperty,
    Thing,
    World,
    locstr,
    sync_reasoner,
)


BASE_IRI = "https://example.org/oke-habsa/laptop.owl#"
OBJECT_PROPERTIES = (
    "hasSubAspect",
    "relatesTo",
    "influences",
    "contradicts",
    "synonymOf",
)
DATA_PROPERTIES = (
    "aspectLabel",
    "domainName",
    "sentimentWeight",
    "isCritical",
    "ontologyDepth",
)


@dataclass
class Concept:
    name: str
    parent: str | None
    labels: list[str]
    domain: str = "laptop"
    sentiment_weight: float = 1.0
    is_critical: bool = False
    multilingual_labels: dict[str, list[str]] = field(default_factory=dict)
    relations: dict[str, list[str]] = field(default_factory=dict)
    depth: int = 0


def _concept(
    name: str,
    parent: str | None,
    labels: Iterable[str],
    *,
    weight: float = 1.0,
    critical: bool = False,
    vi: Iterable[str] = (),
    de: Iterable[str] = (),
    fr: Iterable[str] = (),
    **relations: list[str],
) -> Concept:
    multilingual = {
        language: list(values)
        for language, values in (("vi", vi), ("de", de), ("fr", fr))
        if values
    }
    return Concept(
        name=name,
        parent=parent,
        labels=list(labels),
        sentiment_weight=weight,
        is_critical=critical,
        multilingual_labels=multilingual,
        relations=relations,
    )


def laptop_concepts() -> list[Concept]:
    """Return the reviewed 28-concept Laptop taxonomy.

    Broad or ambiguous dataset terms are attached to a broad concept. This keeps
    the lexicon useful without pretending that a term such as "use" identifies a
    narrow hardware component.
    """

    return [
        _concept(
            "LaptopAspect",
            None,
            ["laptop", "laptops", "computer", "computers", "notebook", "use", "usage"],
            vi=["máy tính xách tay"],
            de=["Laptop", "Notebook"],
            fr=["ordinateur portable"],
        ),
        _concept("HardwareAspect", "LaptopAspect", ["hardware", "device", "specs", "motherboard", "mother board", "bios"]),
        _concept("DisplayAspect", "HardwareAspect", ["screen", "screens", "display", "monitor", "lcd", "graphics", "screen size"]),
        _concept("BrightnessAspect", "DisplayAspect", ["brightness", "bright", "dim", "backlight", "glare"]),
        _concept("ResolutionAspect", "DisplayAspect", ["resolution", "pixel", "pixels", "retina"]),
        _concept("ColorAccuracyAspect", "DisplayAspect", ["color", "colors", "colour", "colours", "contrast"]),
        _concept(
            "BatteryAspect",
            "HardwareAspect",
            ["battery", "batteries", "power", "charge", "charger", "power supply"],
            weight=1.5,
            critical=True,
        ),
        _concept(
            "BatteryLifeAspect",
            "BatteryAspect",
            ["battery life", "battery duration", "runtime", "unplugged time"],
            weight=1.6,
            critical=True,
        ),
        _concept("ChargingSpeedAspect", "BatteryAspect", ["charging speed", "charge time", "fast charge", "recharge"]),
        _concept("KeyboardAspect", "HardwareAspect", ["keyboard", "keyboards", "key", "keys", "typing"]),
        _concept("KeyTravelAspect", "KeyboardAspect", ["key travel", "keystroke", "keystrokes"]),
        _concept("KeyLayoutAspect", "KeyboardAspect", ["key layout", "keyboard layout", "layout", "numpad"]),
        _concept("TouchpadAspect", "HardwareAspect", ["touchpad", "touch pad", "trackpad", "mousepad", "mouse pad", "mouse"]),
        _concept("TouchpadResponsivenessAspect", "TouchpadAspect", ["touchpad responsiveness", "responsive touchpad", "cursor", "navigate"]),
        _concept("TouchpadSizeAspect", "TouchpadAspect", ["touchpad size", "trackpad size", "large touchpad"]),
        _concept("StorageAspect", "HardwareAspect", ["storage", "hard drive", "hard drives", "disk", "ssd", "drive", "cd drive", "space"]),
        _concept("AudioAspect", "HardwareAspect", ["audio", "sound", "speaker", "speakers", "fan"]),
        _concept("ConnectivityAspect", "HardwareAspect", ["wifi", "wi-fi", "wireless", "bluetooth", "port", "ports", "usb", "webcam"]),
        _concept("PerformanceAspect", "LaptopAspect", ["performance", "speed", "fast", "slow", "lag", "runs", "works", "work", "gaming", "games", "boot up"], weight=1.3),
        _concept("ProcessingSpeedAspect", "PerformanceAspect", ["processor", "cpu", "processing speed", "programs", "applications", "web browsing"]),
        _concept("MemoryAspect", "PerformanceAspect", ["memory", "ram"]),
        _concept("SoftwareAspect", "LaptopAspect", ["software", "windows", "windows 7", "vista", "operating system", "system", "os", "program", "drivers", "features", "feature", "iwork", "ilife", "iphoto"]),
        _concept("PortabilityAspect", "LaptopAspect", ["portability", "portable", "travel", "mobility", "carry", "shipping", "delivery"]),
        _concept("WeightAspect", "PortabilityAspect", ["weight", "lightweight", "heavy"]),
        _concept("DimensionsAspect", "PortabilityAspect", ["size", "dimensions", "thin", "thickness", "edges"]),
        _concept("DesignAspect", "LaptopAspect", ["design", "look", "appearance", "quality", "build", "build quality", "hinge"]),
        _concept("PriceAspect", "LaptopAspect", ["price", "prices", "cost", "value", "expensive", "cheap"]),
        _concept("SupportAspect", "LaptopAspect", ["support", "tech support", "warranty", "warrenty", "extended warranty", "service", "customer service"]),
    ]


class OntologyManager:
    """Validated Laptop ontology and conservative entity mapper for Phase 1."""

    def __init__(
        self,
        domain: str = "laptop",
        owl_path: str | Path | None = None,
    ) -> None:
        normalized_domain = domain.lower()
        if normalized_domain in {"laptops", "semeval-laptops"}:
            normalized_domain = "laptop"
        if normalized_domain != "laptop":
            raise ValueError("Phase 1 currently supports only the Laptop ontology")

        self.domain = normalized_domain
        self.owl_path = Path(owl_path) if owl_path else None
        self.concepts = {concept.name: concept for concept in laptop_concepts()}
        self._assign_depths()
        self.names = list(self.concepts)
        self.name_to_id = {name: index for index, name in enumerate(self.names)}
        self._lexicon = self._build_lexicon()

    @staticmethod
    def normalize(text: str | Iterable[str]) -> str:
        if not isinstance(text, str):
            text = " ".join(text)
        normalized = re.sub(r"[^\w\s-]", " ", text.lower(), flags=re.UNICODE)
        return re.sub(r"\s+", " ", normalized).strip()

    @property
    def root(self) -> str:
        return next(name for name, item in self.concepts.items() if item.parent is None)

    @property
    def lexicon(self) -> dict[str, str]:
        return dict(self._lexicon)

    def _assign_depths(self) -> None:
        for concept in self.concepts.values():
            depth = 0
            parent = concept.parent
            visited = {concept.name}
            while parent is not None:
                if parent in visited:
                    raise ValueError(f"Cycle detected at {concept.name}")
                if parent not in self.concepts:
                    raise ValueError(f"Missing parent {parent} for {concept.name}")
                visited.add(parent)
                depth += 1
                parent = self.concepts[parent].parent
            concept.depth = depth

    def _build_lexicon(self) -> dict[str, str]:
        lexicon: dict[str, str] = {}
        for concept in self.concepts.values():
            labels = list(concept.labels)
            labels.extend(
                label
                for language_labels in concept.multilingual_labels.values()
                for label in language_labels
            )
            for label in labels:
                normalized = self.normalize(label)
                previous = lexicon.get(normalized)
                if previous is not None and previous != concept.name:
                    raise ValueError(
                        f"Ambiguous lexicon label {normalized!r}: "
                        f"{previous} and {concept.name}"
                    )
                lexicon[normalized] = concept.name
        return lexicon

    def children(self, name: str) -> list[str]:
        return [
            concept.name
            for concept in self.concepts.values()
            if concept.parent == name
        ]

    def ancestors(self, name: str, include_self: bool = False) -> list[str]:
        if name not in self.concepts:
            raise KeyError(name)
        result = [name] if include_self else []
        parent = self.concepts[name].parent
        while parent is not None:
            result.append(parent)
            parent = self.concepts[parent].parent
        return result

    def hierarchy_edges(self) -> list[tuple[str, str]]:
        return [
            (concept.parent, concept.name)
            for concept in self.concepts.values()
            if concept.parent is not None
        ]

    def triples(self) -> list[tuple[str, str, str]]:
        triples = [
            (parent, "hasSubAspect", child)
            for parent, child in self.hierarchy_edges()
        ]
        for concept in self.concepts.values():
            for relation, targets in concept.relations.items():
                triples.extend(
                    (concept.name, relation, target) for target in targets
                )
        return triples

    def relation_names(self) -> list[str]:
        return sorted({relation for _, relation, _ in self.triples()})

    def topological_order(self, bottom_up: bool = False) -> list[str]:
        order = sorted(
            self.names,
            key=lambda name: (self.concepts[name].depth, self.name_to_id[name]),
        )
        return list(reversed(order)) if bottom_up else order

    def map_entity(
        self,
        text: str | Iterable[str],
        threshold: float = 0.75,
    ) -> tuple[str, float]:
        """Map an explicit mention, falling back to the root below threshold."""

        phrase = self.normalize(text)
        if not phrase:
            return self.root, 0.0
        if phrase in self._lexicon:
            return self._lexicon[phrase], 1.0

        phrase_terms = set(phrase.split())
        best_concept = self.root
        best_score = 0.0
        for label, concept in self._lexicon.items():
            label_terms = set(label.split())
            jaccard = len(phrase_terms & label_terms) / max(
                1, len(phrase_terms | label_terms)
            )
            sequence = SequenceMatcher(None, phrase, label).ratio()
            score = 0.7 * jaccard + 0.3 * sequence
            if score > best_score:
                best_concept = concept
                best_score = score
        if best_score < threshold:
            return self.root, best_score
        return best_concept, best_score

    def map_tokens(
        self,
        tokens: list[str],
        threshold: float = 0.75,
    ) -> list[str | None]:
        mapped: list[str | None] = [None] * len(tokens)
        max_width = min(4, len(tokens))
        for width in range(max_width, 0, -1):
            for start in range(len(tokens) - width + 1):
                if any(mapped[start : start + width]):
                    continue
                concept, score = self.map_entity(
                    tokens[start : start + width], threshold
                )
                if score >= threshold and concept != self.root:
                    mapped[start : start + width] = [concept] * width
        return mapped

    def concept_features(self) -> dict[str, list[float] | list[int]]:
        return {
            "depth": [self.concepts[name].depth for name in self.names],
            "weight": [
                self.concepts[name].sentiment_weight for name in self.names
            ],
            "critical": [
                int(self.concepts[name].is_critical) for name in self.names
            ],
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        roots = [
            concept.name
            for concept in self.concepts.values()
            if concept.parent is None
        ]
        if roots != ["LaptopAspect"]:
            errors.append(f"Expected LaptopAspect as the only root, found {roots}")
        if len(self.concepts) != 28:
            errors.append(f"Expected 28 concepts, found {len(self.concepts)}")
        if len(self.hierarchy_edges()) != 27:
            errors.append(
                f"Expected 27 hierarchy edges, found {len(self.hierarchy_edges())}"
            )
        max_depth = max(concept.depth for concept in self.concepts.values())
        if max_depth != 3:
            errors.append(f"Expected maximum depth 3, found {max_depth}")
        critical_count = sum(
            concept.is_critical for concept in self.concepts.values()
        )
        if critical_count != 2:
            errors.append(
                f"Expected 2 critical concepts, found {critical_count}"
            )
        for concept in self.concepts.values():
            if concept.parent and concept.parent not in self.concepts:
                errors.append(
                    f"{concept.name} has missing parent {concept.parent}"
                )
            if not 0 <= concept.depth <= 3:
                errors.append(
                    f"{concept.name} has invalid depth {concept.depth}"
                )
            for relation, targets in concept.relations.items():
                if relation not in OBJECT_PROPERTIES:
                    errors.append(
                        f"{concept.name} uses unknown relation {relation}"
                    )
                for target in targets:
                    if target not in self.concepts:
                        errors.append(
                            f"{concept.name} references missing concept {target}"
                        )
        return errors

    def _build_owlready_ontology(self):
        world = World()
        ontology = world.get_ontology(BASE_IRI)
        with ontology:
            class Review(Thing):
                pass

            class Aspect(Thing):
                pass

            class hasSubAspect(ObjectProperty):
                domain = [Aspect]
                range = [Aspect]

            class relatesTo(ObjectProperty):
                domain = [Aspect]
                range = [Aspect]

            class influences(ObjectProperty):
                domain = [Aspect]
                range = [Aspect]

            class contradicts(ObjectProperty):
                domain = [Aspect]
                range = [Aspect]
                symmetric = True

            class synonymOf(ObjectProperty):
                domain = [Aspect]
                range = [Aspect]
                symmetric = True

            class aspectLabel(DataProperty):
                domain = [Aspect]
                range = [str]

            class domainName(DataProperty, FunctionalProperty):
                domain = [Aspect]
                range = [str]

            class sentimentWeight(DataProperty, FunctionalProperty):
                domain = [Aspect]
                range = [float]

            class isCritical(DataProperty, FunctionalProperty):
                domain = [Aspect]
                range = [bool]

            class ontologyDepth(DataProperty, FunctionalProperty):
                domain = [Aspect]
                range = [int]

            class hasSentimentScore(DataProperty, FunctionalProperty):
                domain = [Aspect]
                range = [float]

            class hasHighImpact(DataProperty, FunctionalProperty):
                domain = [Aspect]
                range = [bool]

        owl_classes: dict[str, type] = {}
        nodes: dict[str, Thing] = {}
        for name in self.topological_order():
            concept = self.concepts[name]
            parent_class = (
                ontology.Aspect
                if concept.parent is None
                else owl_classes[concept.parent]
            )
            with ontology:
                owl_class = new_class(name, (parent_class,))
            owl_classes[name] = owl_class

            for label in concept.labels:
                owl_class.label.append(locstr(label, lang="en"))
            for language, labels in concept.multilingual_labels.items():
                for label in labels:
                    owl_class.label.append(locstr(label, lang=language))

            node = owl_class(f"{name}Node")
            node.aspectLabel = list(concept.labels)
            node.domainName = concept.domain
            node.sentimentWeight = float(concept.sentiment_weight)
            node.isCritical = bool(concept.is_critical)
            node.ontologyDepth = int(concept.depth)
            nodes[name] = node

        for parent, child in self.hierarchy_edges():
            nodes[parent].hasSubAspect.append(nodes[child])
        for concept in self.concepts.values():
            for relation, targets in concept.relations.items():
                property_type = ontology[relation]
                for target in targets:
                    property_type[nodes[concept.name]].append(nodes[target])

        with ontology:
            top_level = [
                owl_classes[name]
                for name in self.children(self.root)
            ]
            if len(top_level) > 1:
                AllDisjoint(top_level)
            critical_rule = Imp("criticalChildInfluencesParent")
            critical_rule.set_as_rule(
                "hasSubAspect(?parent, ?child), isCritical(?child, true) "
                "-> influences(?child, ?parent)"
            )
        return world, ontology

    def export_owl(self, path: str | Path | None = None) -> Path:
        target = Path(
            path or self.owl_path or "ontology/laptop_domain.owl"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        _, ontology = self._build_owlready_ontology()
        ontology.save(file=str(target), format="rdfxml")
        return target

    def export_json(self, path: str | Path) -> Path:
        from .swrl_rules import RULE_DEFINITIONS

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "domain": self.domain,
            "base_iri": BASE_IRI,
            "concepts": [
                asdict(self.concepts[name]) for name in self.names
            ],
            "triples": self.triples(),
            "lexicon": self._lexicon,
            "rules": [asdict(rule) for rule in RULE_DEFINITIONS],
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

    def export_tsv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["label", "language", "concept", "domain", "depth"]
            )
            for concept in self.concepts.values():
                for label in concept.labels:
                    writer.writerow(
                        [label, "en", concept.name, self.domain, concept.depth]
                    )
                for language, labels in concept.multilingual_labels.items():
                    for label in labels:
                        writer.writerow(
                            [
                                label,
                                language,
                                concept.name,
                                self.domain,
                                concept.depth,
                            ]
                        )
        return target

    def export_all(self, owl_path: str | Path) -> dict[str, Path | str]:
        from .swrl_rules import RULE_DEFINITIONS

        owl_target = self.export_owl(owl_path)
        json_target = self.export_json(owl_target.with_suffix(".json"))
        tsv_target = self.export_tsv(owl_target.with_suffix(".tsv"))
        canonical_payload = {
            "schema_version": 1,
            "domain": self.domain,
            "concepts": [
                asdict(self.concepts[name]) for name in self.names
            ],
            "triples": self.triples(),
            "rules": [asdict(rule) for rule in RULE_DEFINITIONS],
        }
        canonical_bytes = json.dumps(
            canonical_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical_bytes).hexdigest()
        hash_target = owl_target.with_name("ontology_hash.txt")
        hash_target.write_text(
            digest + "\n",
            encoding="ascii",
            newline="\n",
        )
        return {
            "owl": owl_target,
            "json": json_target,
            "tsv": tsv_target,
            "hash": hash_target,
            "sha256": digest,
        }

    @staticmethod
    def validate_owl(
        path: str | Path,
        *,
        run_reasoner: bool = False,
    ) -> list[str]:
        target = Path(path).resolve()
        errors: list[str] = []
        try:
            world = World()
            with target.open("rb") as handle:
                ontology = world.get_ontology(BASE_IRI).load(fileobj=handle)
        except Exception as exc:
            return [f"Cannot load OWL file: {exc}"]

        required_classes = {
            "Aspect",
            "LaptopAspect",
            "HardwareAspect",
            "BatteryLifeAspect",
        }
        class_names = {item.name for item in ontology.classes()}
        missing_classes = sorted(required_classes - class_names)
        if missing_classes:
            errors.append(
                f"OWL file is missing classes: {', '.join(missing_classes)}"
            )

        property_names = {
            item.name for item in ontology.object_properties()
        } | {item.name for item in ontology.data_properties()}
        missing_properties = sorted(
            (set(OBJECT_PROPERTIES) | set(DATA_PROPERTIES)) - property_names
        )
        if missing_properties:
            errors.append(
                "OWL file is missing properties: "
                + ", ".join(missing_properties)
            )

        if run_reasoner and not errors:
            try:
                with ontology:
                    sync_reasoner(
                        world,
                        infer_property_values=True,
                        debug=0,
                    )
                inconsistent = list(world.inconsistent_classes())
                if inconsistent:
                    errors.append(
                        "Reasoner found inconsistent classes: "
                        + ", ".join(item.name for item in inconsistent)
                    )
            except Exception as exc:
                errors.append(f"Reasoner failed: {exc}")
        return errors
