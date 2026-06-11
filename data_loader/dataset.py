from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .preprocessor import TextPreprocessor


POLARITY_TO_INDEX = {
    "very negative": 0,
    "very_negative": 0,
    "negative": 1,
    "neutral": 2,
    "conflict": 2,
    "positive": 3,
    "very positive": 4,
    "very_positive": 4,
}
INDEX_TO_SCORE = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


class OKEHABSADataset(Dataset):
    """One training item per annotated aspect mention."""

    def __init__(
        self,
        records: list[dict],
        preprocessor: TextPreprocessor,
        mapping_threshold: float = 0.45,
    ):
        self.preprocessor = preprocessor
        self.ontology = preprocessor.ontology
        self.mapping_threshold = mapping_threshold
        self.items: list[dict] = []
        for sentence_id, record in enumerate(records):
            prepared = preprocessor.prepare_record(record)
            for aspect in prepared["aspects"]:
                start = max(0, int(aspect.get("from", 0)))
                end = min(len(prepared["tokens"]), int(aspect.get("to", start + 1)))
                if start >= end:
                    continue
                term = aspect.get("term") or prepared["tokens"][start:end]
                concept, confidence = self.ontology.map_entity(term, mapping_threshold)
                # Unknown social entities are valid instances of the ontology root.
                if confidence < mapping_threshold and self.ontology.domain != "social":
                    token_mapped = [
                        name for name in prepared["token_concepts"][start:end] if name is not None
                    ]
                    concept = token_mapped[0] if token_mapped else self.ontology.root
                label = POLARITY_TO_INDEX.get(str(aspect.get("polarity", "neutral")).lower(), 2)
                self.items.append(
                    {
                        **prepared,
                        "sentence_id": sentence_id,
                        "aspect": aspect,
                        "aspect_start": start,
                        "aspect_end": end,
                        "concept": concept,
                        "mapping_confidence": confidence,
                        "label": label,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        length = len(item["input_ids"])
        aspect_mask = [0.0] * length
        for position in range(item["aspect_start"], item["aspect_end"]):
            aspect_mask[position] = 1.0

        concept_id = self.ontology.name_to_id[item["concept"]]
        ancestors = self.ontology.ancestors(item["concept"])
        branch = next(
            (
                name
                for name in reversed(ancestors)
                if self.ontology.concepts[name].depth == 1
            ),
            item["concept"],
        )
        node_targets = torch.full((len(self.ontology.names),), -100, dtype=torch.long)
        node_targets[concept_id] = item["label"]
        for name in ancestors:
            node_targets[self.ontology.name_to_id[name]] = item["label"]

        token_concept_ids = [
            self.ontology.name_to_id[name] if name is not None else -1
            for name in item["token_concepts"]
        ]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(length, dtype=torch.bool),
            "aspect_mask": torch.tensor(aspect_mask, dtype=torch.float),
            "dependency": torch.tensor(
                self.preprocessor.dependency_adjacency(item["heads"]), dtype=torch.float
            ),
            "token_concept_ids": torch.tensor(token_concept_ids, dtype=torch.long),
            "concept_id": torch.tensor(concept_id, dtype=torch.long),
            "main_concept_id": torch.tensor(
                self.ontology.name_to_id[branch], dtype=torch.long
            ),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "node_targets": node_targets,
            "tokens": item["tokens"],
            "concept": item["concept"],
            "sentence_id": item["sentence_id"],
            "mapping_confidence": item["mapping_confidence"],
        }


def collate_oke_habsa(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(item["input_ids"].numel() for item in batch)
    batch_size = len(batch)
    input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    aspect_mask = torch.zeros((batch_size, max_length), dtype=torch.float)
    dependency = torch.zeros((batch_size, max_length, max_length), dtype=torch.float)
    token_concept_ids = torch.full((batch_size, max_length), -1, dtype=torch.long)

    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        aspect_mask[row, :length] = item["aspect_mask"]
        dependency[row, :length, :length] = item["dependency"]
        token_concept_ids[row, :length] = item["token_concept_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "aspect_mask": aspect_mask,
        "dependency": dependency,
        "token_concept_ids": token_concept_ids,
        "concept_id": torch.stack([item["concept_id"] for item in batch]),
        "main_concept_id": torch.stack([item["main_concept_id"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "node_targets": torch.stack([item["node_targets"] for item in batch]),
        "tokens": [item["tokens"] for item in batch],
        "concept": [item["concept"] for item in batch],
        "sentence_id": [item["sentence_id"] for item in batch],
        "mapping_confidence": torch.tensor(
            [item["mapping_confidence"] for item in batch], dtype=torch.float
        ),
    }
