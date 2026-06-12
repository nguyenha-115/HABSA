from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<unk>"]


class VocabularyTokenizer:
    """Offline tokenizer that preserves the tokenized SemEval inputs."""

    def __init__(self, vocabulary: dict[str, int] | None = None, lowercase: bool = True):
        self.lowercase = lowercase
        self.vocabulary = vocabulary or {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }

    @property
    def pad_token_id(self) -> int:
        return self.vocabulary["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.vocabulary["<unk>"]

    def normalize(self, token: str) -> str:
        return token.lower() if self.lowercase else token

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text, flags=re.UNICODE)

    def fit(
        self,
        records: Iterable[dict],
        min_frequency: int = 2,
        max_size: int = 30000,
    ) -> None:
        counts: Counter[str] = Counter()
        for record in records:
            tokens = record.get("token") or self.tokenize(record.get("text", ""))
            counts.update(self.normalize(token) for token in tokens)
        for token, frequency in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        ):
            if frequency < min_frequency or len(self.vocabulary) >= max_size:
                break
            self.vocabulary.setdefault(token, len(self.vocabulary))

    def encode_tokens(self, tokens: list[str], max_length: int | None = None) -> list[int]:
        tokens = tokens[:max_length] if max_length is not None else tokens
        return [
            self.vocabulary.get(self.normalize(token), self.unk_token_id)
            for token in tokens
        ]

    def encode_record(self, tokens: list[str], max_length: int) -> dict:
        tokens = tokens[:max_length]
        return {
            "input_ids": self.encode_tokens(tokens),
            "word_ids": list(range(len(tokens))),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"lowercase": self.lowercase, "vocabulary": self.vocabulary},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocabulary"], payload.get("lowercase", True))


class TransformerTokenizer:
    def __init__(self, model_name: str, local_files_only: bool = False):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=local_files_only
        )

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text, flags=re.UNICODE)

    def encode_record(self, tokens: list[str], max_length: int) -> dict:
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "word_ids": encoded.word_ids(),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"type": "transformer", "model_name": self.model_name}, indent=2
            ),
            encoding="utf-8",
        )
        return target


class TextPreprocessor:
    def __init__(self, tokenizer, ontology, max_length: int = 96):
        self.tokenizer = tokenizer
        self.ontology = ontology
        self.max_length = max_length

    def prepare_record(self, record: dict) -> dict:
        tokens = list(record.get("token") or self.tokenizer.tokenize(record.get("text", "")))
        pos = list(record.get("pos", ["X"] * len(tokens)))
        heads = list(record.get("head", [0] * len(tokens)))
        relations = list(record.get("deprel", ["dep"] * len(tokens)))
        if not (len(tokens) == len(pos) == len(heads) == len(relations)):
            raise ValueError("Token, POS, head and dependency arrays must have equal length")
        encoded = self.tokenizer.encode_record(tokens, self.max_length)
        word_ids = encoded["word_ids"]
        represented = [index for index in word_ids if index is not None]
        kept_words = max(represented, default=-1) + 1
        tokens = tokens[:kept_words]
        heads = [
            head if 0 <= int(head) <= kept_words else 0 for head in heads[:kept_words]
        ]
        word_dependency = self.dependency_adjacency(heads)
        piece_dependency = [[0.0] * len(word_ids) for _ in word_ids]
        for left, left_word in enumerate(word_ids):
            if left_word is None:
                piece_dependency[left][left] = 1.0
                continue
            for right, right_word in enumerate(word_ids):
                if right_word is not None:
                    piece_dependency[left][right] = word_dependency[left_word][right_word]
            total = sum(piece_dependency[left])
            if total:
                piece_dependency[left] = [
                    value / total for value in piece_dependency[left]
                ]
        word_concepts = self.ontology.map_tokens(tokens)
        return {
            "tokens": tokens,
            "input_ids": encoded["input_ids"],
            "word_ids": word_ids,
            "pos": pos[:kept_words],
            "heads": heads,
            "dependency": piece_dependency,
            "deprel": relations[:kept_words],
            "token_concepts": [
                word_concepts[word_id] if word_id is not None else None
                for word_id in word_ids
            ],
            "aspects": record.get("aspects", []),
        }

    @staticmethod
    def dependency_adjacency(heads: list[int]) -> list[list[float]]:
        size = len(heads)
        adjacency = [[0.0] * size for _ in range(size)]
        for index, head in enumerate(heads):
            adjacency[index][index] = 1.0
            if 0 < head <= size:
                adjacency[index][head - 1] = 1.0
                adjacency[head - 1][index] = 1.0
        for row in adjacency:
            total = sum(row)
            if total:
                row[:] = [value / total for value in row]
        return adjacency
