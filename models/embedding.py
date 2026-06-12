from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TextEncoder(nn.Module):
    """XLM-R encoder with an optional lightweight offline BiGRU backend."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        backend: str,
        pretrained_model: str,
        local_files_only: bool,
    ):
        super().__init__()
        self.backend = backend
        if backend == "transformer":
            from transformers import AutoModel

            try:
                self.transformer = AutoModel.from_pretrained(
                    pretrained_model, local_files_only=local_files_only
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot load {pretrained_model!r}. Cache/download it or use "
                    "model.text_backend=bigru for an offline run."
                ) from exc
            source_dim = int(self.transformer.config.hidden_size)
            self.embedding = self.transformer.get_input_embeddings()
            self.rnn = None
        elif backend == "bigru":
            self.transformer = None
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim // 2,
                batch_first=True,
                bidirectional=True,
            )
            source_dim = hidden_dim
        else:
            raise ValueError("text_backend must be 'transformer' or 'bigru'")
        self.projection = (
            nn.Identity() if source_dim == hidden_dim else nn.Linear(source_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def encode_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.backend == "transformer":
            states = self.transformer(
                inputs_embeds=embeddings,
                attention_mask=attention_mask.long(),
            ).last_hidden_state
        else:
            lengths = attention_mask.sum(1).clamp_min(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            packed_states, _ = self.rnn(packed)
            states, _ = nn.utils.rnn.pad_packed_sequence(
                packed_states,
                batch_first=True,
                total_length=embeddings.size(1),
            )
        return self.norm(self.projection(self.dropout(states)))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.backend == "transformer":
            states = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask.long()
            ).last_hidden_state
            return self.norm(self.projection(self.dropout(states)))
        return self.encode_embeddings(self.embed(input_ids), attention_mask)


class DependencyGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, adjacency: torch.Tensor | None) -> torch.Tensor:
        if adjacency is None:
            return states
        messages = torch.bmm(adjacency, states)
        candidate = torch.tanh(self.message(messages))
        gate = torch.sigmoid(self.gate(torch.cat([states, candidate], dim=-1)))
        return self.norm(states + self.dropout(gate * candidate))


class TransEEmbedding(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dimension: int):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dimension)
        self.relation = nn.Embedding(num_relations, dimension)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        head = self.entity(triples[:, 0])
        relation = self.relation(triples[:, 1])
        tail = self.entity(triples[:, 2])
        return torch.linalg.vector_norm(head + relation - tail, ord=1, dim=-1)

    def margin_loss(
        self, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0
    ) -> torch.Tensor:
        return F.relu(margin + self.score(positive) - self.score(negative)).mean()

    def normalize_(self) -> None:
        with torch.no_grad():
            self.entity.weight.copy_(F.normalize(self.entity.weight, dim=-1))


def sinusoidal_depth_encoding(depths: torch.Tensor, dimension: int) -> torch.Tensor:
    positions = depths.float().unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, dimension, 2, device=depths.device).float()
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros((len(depths), dimension), device=depths.device)
    encoding[:, 0::2] = torch.sin(positions * divisor)
    encoding[:, 1::2] = torch.cos(
        positions * divisor[: encoding[:, 1::2].shape[1]]
    )
    return encoding


class HierarchyEncoder(nn.Module):
    def __init__(self, transe: TransEEmbedding, ontology_dim: int, hidden_dim: int):
        super().__init__()
        self.transe = transe
        self.projection = nn.Linear(ontology_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, depths: torch.Tensor) -> torch.Tensor:
        entities = self.transe.entity.weight
        positions = sinusoidal_depth_encoding(depths, entities.size(-1)).to(entities.dtype)
        return self.norm(self.projection(torch.cat([entities, positions], dim=-1)))
