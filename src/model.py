"""Feed-forward ANN for date-fruit classification."""
import torch
from torch import nn


class DateFruitANN(nn.Module):
    """ANN architecture: input -> 64 -> 64 -> class logits."""
    def __init__(self, input_features: int = 34, num_classes: int = 7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)
