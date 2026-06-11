from __future__ import annotations

import torch


def ontology_adjacency(ontology, self_loops: bool = True) -> torch.Tensor:
    size = len(ontology.names)
    adjacency = torch.zeros((size, size), dtype=torch.float)
    for parent, child in ontology.hierarchy_edges():
        left = ontology.name_to_id[parent]
        right = ontology.name_to_id[child]
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    if self_loops:
        adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    return adjacency / degree.unsqueeze(1)


def blend_graphs(
    syntax: torch.Tensor, ontology_edges: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    if syntax.shape != ontology_edges.shape:
        raise ValueError("Graphs must have the same shape")
    return alpha * syntax + (1.0 - alpha) * ontology_edges


def shortest_hierarchy_path(ontology, source: str, target: str) -> list[str]:
    source_path = [source] + ontology.ancestors(source)
    target_path = [target] + ontology.ancestors(target)
    target_positions = {name: index for index, name in enumerate(target_path)}
    for source_index, name in enumerate(source_path):
        if name in target_positions:
            target_index = target_positions[name]
            return source_path[: source_index + 1] + list(
                reversed(target_path[:target_index])
            )
    return []
