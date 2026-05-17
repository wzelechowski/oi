import torch
from torch import nn

class Imagenette2DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.dim_reducer = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2)
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