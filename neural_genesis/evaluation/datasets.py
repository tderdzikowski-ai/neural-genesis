"""
Neural Genesis — Dataset Loading

Ładuje FashionMNIST (screening) i CIFAR-10 (pełna ewaluacja).
Wszystko z torchvision — zero zewnętrznych zależności.
"""

from __future__ import annotations
from typing import Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"


def get_data_loaders(
    dataset_name: str = "FashionMNIST",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Zwraca (train_loader, test_loader) dla danego datasetu.
    Dane pobierane automatycznie przy pierwszym uruchomieniu.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_name == "FashionMNIST":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.2860,), (0.3530,)),  # FashionMNIST mean/std
        ])
        train_set = torchvision.datasets.FashionMNIST(
            root=str(DATA_DIR), train=True, download=True, transform=transform,
        )
        test_set = torchvision.datasets.FashionMNIST(
            root=str(DATA_DIR), train=False, download=True, transform=transform,
        )

    elif dataset_name == "CIFAR10":
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_set = torchvision.datasets.CIFAR10(
            root=str(DATA_DIR), train=True, download=True, transform=transform_train,
        )
        test_set = torchvision.datasets.CIFAR10(
            root=str(DATA_DIR), train=False, download=True, transform=transform_test,
        )

    elif dataset_name == "CIFAR100":
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_set = torchvision.datasets.CIFAR100(
            root=str(DATA_DIR), train=True, download=True, transform=transform_train,
        )
        test_set = torchvision.datasets.CIFAR100(
            root=str(DATA_DIR), train=False, download=True, transform=transform_test,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # MPS (Mac) nie wspiera pin_memory i ma problemy z num_workers > 0
    import torch
    use_cuda = torch.cuda.is_available()
    pin = use_cuda
    workers = num_workers if use_cuda else 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    return train_loader, test_loader