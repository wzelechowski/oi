import torch
from torch import nn
import torch.nn.functional as F


class Mnist2DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.dim_reducer = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.final_classifier = nn.Linear(2, 10)

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        flat_features = torch.flatten(features, 1)

        features_2d = self.dim_reducer(flat_features)

        logits = self.final_classifier(features_2d)

        if return_features:
            return logits, features_2d
        return logits