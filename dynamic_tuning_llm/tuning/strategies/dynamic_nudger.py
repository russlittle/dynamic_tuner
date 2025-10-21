import torch

from tuning.metrics.embedding_separation import compute_embedding_separation
from tuning.metrics.entropy import compute_activation_entropy


class DynamicNudger:
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def adjust(self, model, hidden_states) -> None:
        if self.strategy == "activation_entropy":
            score = compute_activation_entropy(hidden_states)
            if score < 5.0:
                for param in model.parameters():
                    param.data.add_(0.01 * torch.randn_like(param))
        elif self.strategy == "embedding_separation":
            sep = compute_embedding_separation(hidden_states)
            if sep < 1.0:
                model.encoder.dropout.p = min(0.3, model.encoder.dropout.p + 0.05)
