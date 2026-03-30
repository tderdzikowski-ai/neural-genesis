"""
Neural Genesis — Evaluation Networks

Stała architektura sieci używana do testowania funkcji aktywacji.
Zmieniamy TYLKO aktywację — reszta jest identyczna dla każdego kandydata.
To gwarantuje fair comparison.
"""

from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn

from config import EVAL_NETWORK


class EvalNetwork(nn.Module):
    """
    4-warstwowy CNN + 2 FC do klasyfikacji obrazów.
    Aktywacja jest parametrem — wstawiamy naszych kandydatów.

    Args:
        activation_factory: Callable[[], nn.Module] — fabryka aktywacji.
            Wywoływana osobno dla każdej warstwy (osobne parametry).
        in_channels: Kanały wejściowe (1 dla grayscale, 3 dla RGB).
        num_classes: Liczba klas (10 dla CIFAR-10/FashionMNIST).
        img_size: Rozmiar obrazu (28 dla FashionMNIST, 32 dla CIFAR-10).
    """

    def __init__(
        self,
        activation_factory: Callable[[], nn.Module],
        in_channels: int = 1,
        num_classes: int = 10,
        img_size: int = 28,
    ):
        super().__init__()

        ch = EVAL_NETWORK["conv_channels"]  # [32, 32, 64, 64]
        fc_hidden = EVAL_NETWORK["fc_hidden"]  # 256

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, ch[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[0]),
            activation_factory(),
            nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[1]),
            activation_factory(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[2]),
            activation_factory(),
            nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[3]),
            activation_factory(),
            nn.MaxPool2d(2),
        )

        # Po 2x MaxPool: img_size / 4
        feature_size = ch[3] * (img_size // 4) ** 2

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, fc_hidden),
            activation_factory(),
            nn.Dropout(0.25),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_eval_network(
    activation_factory: Callable[[], nn.Module],
    dataset_name: str = "FashionMNIST",
) -> EvalNetwork:
    """
    Buduje sieć ewaluacyjną dopasowaną do datasetu.
    """
    dataset_config = {
        "FashionMNIST": {"in_channels": 1, "num_classes": 10, "img_size": 28},
        "CIFAR10":      {"in_channels": 3, "num_classes": 10, "img_size": 32},
        "CIFAR100":     {"in_channels": 3, "num_classes": 100, "img_size": 32},
    }

    if dataset_name not in dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(dataset_config.keys())}")

    cfg = dataset_config[dataset_name]
    return EvalNetwork(
        activation_factory=activation_factory,
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        img_size=cfg["img_size"],
    )
