import torch
from torch.utils.data import Dataset


class ExtractedFeaturesDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.X = torch.load(features_path)
        self.y = torch.load(labels_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
