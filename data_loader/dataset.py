from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .preprocessor import TextPreprocessor


LABEL_SCHEMES = {
    3: {
        "negative": 0,
        "neutral": 1,
        "conflict": 1,
        "positive": 2,
    },
    5: {
        "very negative": 0,
        "very_negative": 0,
        "negative": 1,
        "neutral": 2,
        "conflict": 2,
        "positive": 3,
        "very positive": 4,
        "very_positive": 4,
    },
}
SENTIMENT_VALUES = {
    3: torch.tensor([-1.0, 0.0, 1.0]),
    5: torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


class OKEHABSADataset(Dataset):
    """One sample per aspect mention, preserving sentence-level split boundaries."""

    def __init__(
        self,
        records: list[dict],
        preprocessor: TextPreprocessor,
        mapping_threshold: float = 0.75,
        num_sentiments: int = 3,
        sentence_ids: list[int] | None = None,
    ):
        if num_sentiments not in LABEL_SCHEMES:
            raise ValueError("num_sentiments must be 3 or 5")
        self.preprocessor = preprocessor
        self.ontology = preprocessor.ontology
        self.num_sentiments = num_sentiments
        self.items: list[dict] = []
        ids = sentence_ids or list(range(len(records)))
        for record, sentence_id in zip(records, ids, strict=True):
            prepared = preprocessor.prepare_record(record)
            for aspect in prepared["aspects"]:
                start = int(aspect.get("from", 0))
                end = int(aspect.get("to", start + 1))
                if start < 0 or end > len(prepared["tokens"]) or start >= end:
                    continue
                polarity = str(aspect.get("polarity", "neutral")).lower()
                if polarity not in LABEL_SCHEMES[num_sentiments]:
                    raise ValueError(
                        f"Polarity {polarity!r} is invalid for {num_sentiments} classes"
                    )
                term = aspect.get("term") or prepared["tokens"][start:end]
                concept, confidence = self.ontology.map_entity(term, mapping_threshold)
                if confidence < mapping_threshold:
                    mapped = [
                        concept
                        for concept, word_id in zip(
                            prepared["token_concepts"], prepared["word_ids"], strict=True
                        )
                        if word_id is not None
                        and start <= word_id < end
                        and concept is not None
                    ]
                    concept = mapped[0] if mapped else self.ontology.root
                self.items.append(
                    {
                        **prepared,
                        "sentence_id": sentence_id,
                        "aspect_start": start,
                        "aspect_end": end,
                        "concept": concept,
                        "mapping_confidence": confidence,
                        "label": LABEL_SCHEMES[num_sentiments][polarity],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def label_counts(self) -> Counter:
        return Counter(item["label"] for item in self.items)

    def class_weights(self) -> torch.Tensor:
        counts = self.label_counts()
        total = len(self.items)
        return torch.tensor(
            [
                total / max(1, self.num_sentiments * counts.get(index, 0))
                for index in range(self.num_sentiments)
            ],
            dtype=torch.float,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        length = len(item["input_ids"])
        aspect_mask = torch.tensor(
            [
                float(
                    word_id is not None
                    and item["aspect_start"] <= word_id < item["aspect_end"]
                )
                for word_id in item["word_ids"]
            ]
        )
        if not aspect_mask.any():
            raise ValueError("Aspect mask is empty after tokenization/truncation")
        concept_id = self.ontology.name_to_id[item["concept"]]
        ancestors = self.ontology.ancestors(item["concept"])
        main = next(
            (
                name
                for name in reversed(ancestors)
                if self.ontology.concepts[name].depth == 1
            ),
            item["concept"],
        )
        node_targets = torch.full((len(self.ontology.names),), -100, dtype=torch.long)
        for name in [item["concept"], *ancestors]:
            node_targets[self.ontology.name_to_id[name]] = item["label"]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(length, dtype=torch.bool),
            "aspect_mask": aspect_mask,
            "dependency": torch.tensor(item["dependency"], dtype=torch.float),
            "token_concept_ids": torch.tensor(
                [
                    self.ontology.name_to_id[name] if name is not None else -1
                    for name in item["token_concepts"]
                ],
                dtype=torch.long,
            ),
            "concept_id": torch.tensor(concept_id),
            "main_concept_id": torch.tensor(self.ontology.name_to_id[main]),
            "label": torch.tensor(item["label"]),
            "node_targets": node_targets,
            "tokens": item["tokens"],
            "concept": item["concept"],
            "sentence_id": item["sentence_id"],
            "mapping_confidence": item["mapping_confidence"],
        }


def collate_oke_habsa(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(item["input_ids"].numel() for item in batch)
    size = len(batch)
    input_ids = torch.zeros((size, max_length), dtype=torch.long)
    attention_mask = torch.zeros((size, max_length), dtype=torch.bool)
    aspect_mask = torch.zeros((size, max_length))
    dependency = torch.zeros((size, max_length, max_length))
    token_concept_ids = torch.full((size, max_length), -1, dtype=torch.long)
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
