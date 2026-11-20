import torch
from torch import nn

ARCHITECTURE = "mnist-cnn-v1"
MEAN, STD = 0.1307, 0.3081


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.layers(x)


def normalize(pixels):
    return (torch.as_tensor(pixels, dtype=torch.float32).reshape(1, 1, 28, 28) - MEAN) / STD
