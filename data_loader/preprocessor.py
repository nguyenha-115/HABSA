from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class VocabularyTokenizer:
    """Small offline tokenizer preserving the tokenized SemEval inputs."""

    def __init__(self, vocabulary: dict[str, int] | None = None, lowercase: bool = True):
        self.lowercase = lowercase
        self.vocabulary = vocabulary or {token: index for index, token in enumerate(SPECIAL_TOKENS)}

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

    def fit(self, records: Iterable[dict], min_frequency: int = 2, max_size: int = 30000) -> None:
        counts: Counter[str] = Counter()
        for record in records:
            tokens = record.get("token") or self.tokenize(record.get("text", ""))
            counts.update(self.normalize(token) for token in tokens)
        words = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for token, frequency in words:
            if frequency < min_frequency or len(self.vocabulary) >= max_size:
                break
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)

    def encode_tokens(self, tokens: list[str], max_length: int | None = None) -> list[int]:
        if max_length is not None:
            tokens = tokens[:max_length]
        return [
            self.vocabulary.get(self.normalize(token), self.unk_token_id)
            for token in tokens
        ]

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        return self.encode_tokens(self.tokenize(text), max_length)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"lowercase": self.lowercase, "vocabulary": self.vocabulary}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocabulary"], payload.get("lowercase", True))


class TextPreprocessor:
    def __init__(self, tokenizer: VocabularyTokenizer, ontology, max_length: int = 96):
        self.tokenizer = tokenizer
        self.ontology = ontology
        self.max_length = max_length

    def prepare_record(self, record: dict) -> dict:
        tokens = list(record.get("token") or self.tokenizer.tokenize(record.get("text", "")))
        tokens = tokens[: self.max_length]
        token_ids = self.tokenizer.encode_tokens(tokens)
        pos = list(record.get("pos", ["X"] * len(tokens)))[: len(tokens)]
        heads = list(record.get("head", [0] * len(tokens)))[: len(tokens)]
        relations = list(record.get("deprel", ["dep"] * len(tokens)))[: len(tokens)]
        token_concepts = self.ontology.map_tokens(tokens)
        implicit = self.ontology.infer_implicit(tokens)
        return {
            "tokens": tokens,
            "input_ids": token_ids,
            "pos": pos,
            "heads": heads,
            "deprel": relations,
            "token_concepts": token_concepts,
            "implicit_concepts": implicit,
            "aspects": record.get("aspects", []),
        }

    @staticmethod
    def dependency_adjacency(heads: list[int]) -> list[list[float]]:
        size = len(heads)
        adjacency = [[0.0] * size for _ in range(size)]
        for index, head in enumerate(heads):
            adjacency[index][index] = 1.0
            if head > 0 and head - 1 < size:
                adjacency[index][head - 1] = 1.0
                adjacency[head - 1][index] = 1.0
        for row in adjacency:
            total = sum(row)
            if total:
                for index in range(size):
                    row[index] /= total
        return adjacency
