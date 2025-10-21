import torch
import torch.nn.functional as F


def compute_activation_entropy(hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute the mean entropy of the activation distribution."""

    probs = F.softmax(hidden_states, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-10)) / probs.shape[0]
