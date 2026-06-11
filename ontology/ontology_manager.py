from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


@dataclass
class Concept:
    name: str
    parent: str | None
    labels: list[str]
    domain: str
    sentiment_weight: float = 1.0
    is_critical: bool = False
    relations: dict[str, list[str]] = field(default_factory=dict)
    depth: int = 0


def _concept(
    name: str,
    parent: str | None,
    labels: Iterable[str],
    domain: str,
    weight: float = 1.0,
    critical: bool = False,
    **relations: list[str],
) -> Concept:
    return Concept(name, parent, list(labels), domain, weight, critical, relations)


def _laptop_concepts() -> list[Concept]:
    d = "laptop"
    return [
        _concept("LaptopAspect", None, ["laptop", "computer", "notebook"], d),
        _concept("HardwareAspect", "LaptopAspect", ["hardware", "build", "device"], d),
        _concept("DisplayAspect", "HardwareAspect", ["screen", "display", "monitor", "lcd", "graphics"], d),
        _concept("BrightnessAspect", "DisplayAspect", ["brightness", "bright", "dim", "backlight"], d),
        _concept("ResolutionAspect", "DisplayAspect", ["resolution", "pixels", "retina"], d),
        _concept("ColorAccuracyAspect", "DisplayAspect", ["color", "colour", "contrast"], d),
        _concept("BatteryAspect", "HardwareAspect", ["battery", "power", "charge", "charger", "battery life"], d, 1.5, True),
        _concept("BatteryLifeAspect", "BatteryAspect", ["battery life", "runtime", "battery duration"], d, 1.6, True),
        _concept("ChargingSpeedAspect", "BatteryAspect", ["charging speed", "fast charge", "recharge"], d),
        _concept("KeyboardAspect", "HardwareAspect", ["keyboard", "keys", "typing"], d),
        _concept("KeyTravelAspect", "KeyboardAspect", ["key travel", "keystroke"], d),
        _concept("KeyLayoutAspect", "KeyboardAspect", ["key layout", "layout", "numpad"], d),
        _concept("TouchpadAspect", "HardwareAspect", ["touchpad", "trackpad", "mouse pad"], d),
        _concept("TouchpadResponsivenessAspect", "TouchpadAspect", ["touchpad responsiveness", "responsive touchpad", "cursor"], d),
        _concept("TouchpadSizeAspect", "TouchpadAspect", ["touchpad size", "large touchpad"], d),
        _concept("StorageAspect", "HardwareAspect", ["storage", "hard drive", "disk", "ssd", "drive"], d),
        _concept("AudioAspect", "HardwareAspect", ["audio", "sound", "speaker", "speakers"], d),
        _concept("ConnectivityAspect", "HardwareAspect", ["wifi", "wireless", "bluetooth", "port", "ports", "usb"], d),
        _concept("PerformanceAspect", "LaptopAspect", ["performance", "speed", "fast", "slow", "lag", "runs"], d, 1.3),
        _concept("ProcessingSpeedAspect", "PerformanceAspect", ["processor", "cpu", "processing speed", "programs", "applications"], d),
        _concept("MemoryAspect", "PerformanceAspect", ["memory", "ram"], d),
        _concept("SoftwareAspect", "LaptopAspect", ["software", "windows", "operating system", "program"], d),
        _concept("PortabilityAspect", "LaptopAspect", ["portability", "portable", "travel", "mobility"], d),
        _concept("WeightAspect", "PortabilityAspect", ["weight", "lightweight", "heavy"], d),
        _concept("DimensionsAspect", "PortabilityAspect", ["size", "dimensions", "thin", "thickness"], d),
        _concept("DesignAspect", "LaptopAspect", ["design", "look", "appearance", "quality", "build quality"], d),
        _concept("PriceAspect", "LaptopAspect", ["price", "cost", "value", "expensive", "cheap"], d),
        _concept("SupportAspect", "LaptopAspect", ["support", "warranty", "service", "customer service"], d),
    ]


def _restaurant_concepts() -> list[Concept]:
    d = "restaurant"
    return [
        _concept("RestaurantAspect", None, ["restaurant", "place", "experience"], d),
        _concept("FoodAspect", "RestaurantAspect", ["food", "meal", "dish", "dishes", "cuisine", "dinner", "lunch"], d, 1.4),
        _concept("TasteAspect", "FoodAspect", ["taste", "flavor", "flavour", "delicious", "spicy", "sweet"], d),
        _concept("QualityAspect", "FoodAspect", ["quality", "fresh", "freshness", "temperature", "cooked"], d),
        _concept("PortionAspect", "QualityAspect", ["portion", "portions", "serving"], d),
        _concept("PresentationAspect", "FoodAspect", ["presentation", "plating", "garnish"], d),
        _concept("MenuAspect", "FoodAspect", ["menu", "selection", "choices", "wine list"], d),
        _concept("DrinkAspect", "FoodAspect", ["drink", "drinks", "wine", "bar", "cocktail"], d),
        _concept("ServiceAspect", "RestaurantAspect", ["service", "staff", "waiter", "waitress", "server"], d, 1.5, True),
        _concept("SpeedAspect", "ServiceAspect", ["speed", "wait", "waiting", "slow service", "quick service"], d),
        _concept("FriendlinessAspect", "ServiceAspect", ["friendly", "rude", "polite", "attitude"], d),
        _concept("AttentivenessAspect", "ServiceAspect", ["attentive", "attention", "responsive"], d),
        _concept("AmbianceAspect", "RestaurantAspect", ["ambiance", "ambience", "atmosphere", "environment"], d),
        _concept("DecorationAspect", "AmbianceAspect", ["decor", "decoration", "design", "interior"], d),
        _concept("NoiseAspect", "AmbianceAspect", ["noise", "noisy", "quiet", "music"], d),
        _concept("CleanlinessAspect", "AmbianceAspect", ["clean", "cleanliness", "dirty", "hygiene"], d, 1.4, True),
        _concept("LocationAspect", "RestaurantAspect", ["location", "neighborhood", "area"], d),
        _concept("PriceAspect", "RestaurantAspect", ["price", "prices", "cost", "priced", "expensive", "cheap"], d),
        _concept("ValueForMoneyAspect", "PriceAspect", ["value", "worth", "value for money"], d),
        _concept("MenuPriceAspect", "PriceAspect", ["menu price", "bill", "check"], d),
    ]


def _social_concepts() -> list[Concept]:
    d = "social"
    return [
        _concept("EntityAspect", None, ["entity", "topic", "subject"], d),
        _concept("PersonAspect", "EntityAspect", ["person", "people", "president", "actor", "candidate"], d),
        _concept("OrganizationAspect", "EntityAspect", ["company", "organization", "team", "government"], d),
        _concept("ProductAspect", "EntityAspect", ["product", "device", "service", "brand"], d),
        _concept("EventAspect", "EntityAspect", ["event", "movie", "game", "show", "prize"], d),
    ]


class OntologyManager:
    """In-memory domain ontology with deterministic OWL export."""

    def __init__(self, domain: str = "laptop", owl_path: str | Path | None = None):
        aliases = {
            "laptops": "laptop",
            "restaurants": "restaurant",
            "restaurants16": "restaurant",
            "tweets": "social",
        }
        self.domain = aliases.get(domain.lower(), domain.lower())
        builders = {
            "laptop": _laptop_concepts,
            "restaurant": _restaurant_concepts,
            "social": _social_concepts,
        }
        if self.domain not in builders:
            raise ValueError(f"Unsupported ontology domain: {domain}")
        self.concepts = {item.name: item for item in builders[self.domain]()}
        self._assign_depths()
        self.names = list(self.concepts)
        self.name_to_id = {name: index for index, name in enumerate(self.names)}
        self.owl_path = Path(owl_path) if owl_path else None
        self._lexicon = self._build_lexicon()

    def _assign_depths(self) -> None:
        for concept in self.concepts.values():
            depth, parent = 0, concept.parent
            visited = {concept.name}
            while parent is not None:
                if parent in visited or parent not in self.concepts:
                    raise ValueError(f"Invalid ontology hierarchy at {concept.name}")
                visited.add(parent)
                depth += 1
                parent = self.concepts[parent].parent
            concept.depth = depth

    def _build_lexicon(self) -> dict[str, str]:
        lexicon: dict[str, str] = {}
        for concept in self.concepts.values():
            for label in concept.labels:
                lexicon[label.lower()] = concept.name
        return lexicon

    @property
    def root(self) -> str:
        return next(name for name, concept in self.concepts.items() if concept.parent is None)

    def children(self, name: str) -> list[str]:
        return [item.name for item in self.concepts.values() if item.parent == name]

    def ancestors(self, name: str, include_self: bool = False) -> list[str]:
        path = [name] if include_self else []
        parent = self.concepts[name].parent
        while parent is not None:
            path.append(parent)
            parent = self.concepts[parent].parent
        return path

    def hierarchy_edges(self) -> list[tuple[str, str]]:
        return [
            (concept.parent, concept.name)
            for concept in self.concepts.values()
            if concept.parent is not None
        ]

    def triples(self) -> list[tuple[str, str, str]]:
        triples = [(parent, "hasSubAspect", child) for parent, child in self.hierarchy_edges()]
        for concept in self.concepts.values():
            for relation, targets in concept.relations.items():
                triples.extend((concept.name, relation, target) for target in targets)
        return triples

    def relation_names(self) -> list[str]:
        return sorted({relation for _, relation, _ in self.triples()})

    def topological_order(self, bottom_up: bool = False) -> list[str]:
        order = sorted(self.names, key=lambda name: self.concepts[name].depth)
        return list(reversed(order)) if bottom_up else order

    @staticmethod
    def _normalize(text: str | Iterable[str]) -> str:
        if not isinstance(text, str):
            text = " ".join(text)
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", text.lower())).strip()

    def map_entity(self, text: str | Iterable[str], threshold: float = 0.45) -> tuple[str, float]:
        phrase = self._normalize(text)
        if phrase in self._lexicon:
            return self._lexicon[phrase], 1.0

        padded = f" {phrase} "
        contained = [
            (label, concept)
            for label, concept in self._lexicon.items()
            if f" {label} " in padded or f" {phrase} " in f" {label} "
        ]
        if contained:
            label, concept = max(contained, key=lambda item: len(item[0]))
            return concept, min(0.95, 0.7 + 0.02 * len(label.split()))

        phrase_terms = set(phrase.split())
        best_name, best_score = self.root, 0.0
        for label, concept in self._lexicon.items():
            label_terms = set(label.split())
            jaccard = len(phrase_terms & label_terms) / max(1, len(phrase_terms | label_terms))
            sequence = SequenceMatcher(None, phrase, label).ratio()
            score = 0.65 * jaccard + 0.35 * sequence
            if score > best_score:
                best_name, best_score = concept, score
        return (best_name, best_score) if best_score >= threshold else (self.root, best_score)

    def map_tokens(self, tokens: list[str], threshold: float = 0.45) -> list[str | None]:
        result: list[str | None] = [None] * len(tokens)
        normalized = [self._normalize(token) for token in tokens]
        max_ngram = min(4, len(tokens))
        for width in range(max_ngram, 0, -1):
            for start in range(len(tokens) - width + 1):
                if any(result[start : start + width]):
                    continue
                name, score = self.map_entity(normalized[start : start + width], threshold)
                if score >= threshold and name != self.root:
                    result[start : start + width] = [name] * width
        return result

    def infer_implicit(self, tokens: list[str]) -> list[str]:
        text = f" {' '.join(token.lower() for token in tokens)} "
        rules = {
            "laptop": {
                "PerformanceAspect": [" lag ", " slow ", " freezes ", " hanging "],
                "DisplayAspect": [" brightness ", " screen glare ", " pixels "],
                "BatteryAspect": [" charge ", " unplugged ", " power outlet "],
            },
            "restaurant": {
                "SpeedAspect": [" waited ", " waiting ", " took forever "],
                "AmbianceAspect": [" romantic ", " cozy ", " loud "],
                "CleanlinessAspect": [" dirty ", " spotless ", " hygiene "],
            },
            "social": {},
        }
        return [
            concept
            for concept, patterns in rules[self.domain].items()
            if any(pattern in text for pattern in patterns)
        ]

    def concept_features(self) -> dict[str, list[float] | list[int]]:
        return {
            "depth": [self.concepts[name].depth for name in self.names],
            "weight": [self.concepts[name].sentiment_weight for name in self.names],
            "critical": [int(self.concepts[name].is_critical) for name in self.names],
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        roots = [item.name for item in self.concepts.values() if item.parent is None]
        if len(roots) != 1:
            errors.append(f"Expected one root, found {len(roots)}")
        for concept in self.concepts.values():
            if concept.parent and concept.parent not in self.concepts:
                errors.append(f"{concept.name} has missing parent {concept.parent}")
            for targets in concept.relations.values():
                errors.extend(
                    f"{concept.name} references missing concept {target}"
                    for target in targets
                    if target not in self.concepts
                )
        return errors

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "domain": self.domain,
            "concepts": [asdict(self.concepts[name]) for name in self.names],
            "triples": self.triples(),
            "lexicon": self._lexicon,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def export_tsv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = ["label\tconcept\tdomain\tdepth"]
        for label, name in sorted(self._lexicon.items()):
            rows.append(f"{label}\t{name}\t{self.domain}\t{self.concepts[name].depth}")
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return target

    def export_owl(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.owl_path or f"ontology/{self.domain}_domain.owl")
        target.parent.mkdir(parents=True, exist_ok=True)
        base = f"http://example.org/oke-habsa/{self.domain}#"
        rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        rdfs = "http://www.w3.org/2000/01/rdf-schema#"
        owl = "http://www.w3.org/2002/07/owl#"
        oke = base
        ET.register_namespace("rdf", rdf)
        ET.register_namespace("rdfs", rdfs)
        ET.register_namespace("owl", owl)
        ET.register_namespace("oke", oke)
        root = ET.Element(f"{{{rdf}}}RDF")
        ET.SubElement(root, f"{{{owl}}}Ontology", {f"{{{rdf}}}about": base[:-1]})

        for prop in ["hasSubAspect", "relatesTo", "influences", "contradicts", "synonymOf"]:
            ET.SubElement(root, f"{{{owl}}}ObjectProperty", {f"{{{rdf}}}about": base + prop})
        for prop in ["aspectLabel", "domainName", "sentimentWeight", "isCritical", "ontologyDepth"]:
            ET.SubElement(root, f"{{{owl}}}DatatypeProperty", {f"{{{rdf}}}about": base + prop})

        for concept in self.concepts.values():
            node = ET.SubElement(root, f"{{{owl}}}Class", {f"{{{rdf}}}about": base + concept.name})
            if concept.parent:
                ET.SubElement(node, f"{{{rdfs}}}subClassOf", {f"{{{rdf}}}resource": base + concept.parent})
            for label in concept.labels:
                label_node = ET.SubElement(node, f"{{{rdfs}}}label", {"{http://www.w3.org/XML/1998/namespace}lang": "en"})
                label_node.text = label
            ET.SubElement(node, f"{{{oke}}}domainName").text = concept.domain
            ET.SubElement(node, f"{{{oke}}}sentimentWeight").text = str(concept.sentiment_weight)
            ET.SubElement(node, f"{{{oke}}}isCritical").text = str(concept.is_critical).lower()
            ET.SubElement(node, f"{{{oke}}}ontologyDepth").text = str(concept.depth)

        ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
        return target
