import torch


def compute_embedding_separation(hidden_states: torch.Tensor) -> torch.Tensor:
    """Measure average distance of activations from their centroid."""

    center = torch.mean(hidden_states, dim=0)
    return torch.mean(torch.norm(hidden_states - center, dim=-1))
