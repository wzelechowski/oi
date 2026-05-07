import torch
from torch import nn
import torch.nn.functional as F


class Imagenette2DCNN(nn.Module):
    def __init__(self):
        super(Imagenette2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16384, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc_features = nn.Linear(256, 2)
        self.fc_classifier = nn.Linear(2, 10)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        features_2d = self.fc_features(x)
        logits = self.fc_classifier(features_2d)
        if return_features:
            return logits, features_2d

        return logits