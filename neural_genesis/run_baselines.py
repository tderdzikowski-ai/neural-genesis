#!/usr/bin/env python3
"""
Neural Genesis — Measure Baselines

Uruchom PRZED random search, żeby mieć punkt odniesienia.
Mierzy ReLU, GELU, Swish, Mish i inne na dokładnie tej samej
sieci i danych, których używamy do ewaluacji.

Usage:
    python run_baselines.py
    python run_baselines.py --dataset CIFAR10
"""

import argparse
import logging
import sys
import time

import torch

from baselines import BASELINES
from evaluation.networks import build_eval_network
from evaluation.datasets import get_data_loaders
from evaluation.trainer import train_and_evaluate
from evaluation.metrics import ActivationScore, compute_composite_score
from analysis.leaderboard import Leaderboard
from config import DEVICE, FULL_EVAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Measure baseline activations")
    parser.add_argument("--dataset", default="FashionMNIST",
                        choices=["FashionMNIST", "CIFAR10", "CIFAR100"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Test only these activations (e.g. --only ReLU GELU Mish)")
    args = parser.parse_args()

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Seeds: {args.seeds}")
    if args.only:
        logger.info(f"Only: {args.only}")

    leaderboard = Leaderboard()

    items = BASELINES.items()
    if args.only:
        items = [(n, f) for n, f in items if n in args.only]

    for name, factory in items:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Testing: {name}")
        start = time.time()

        accs = []
        for seed in args.seeds:
            score = train_and_evaluate(
                activation_factory=factory,
                dataset_name=args.dataset,
                epochs=args.epochs,
                batch_size=128,
                learning_rate=0.001,
                device=DEVICE,
                seed=seed,
                early_stop_epoch=999,  # Nie przerywaj baselines
                early_stop_min_acc=0.0,
            )
            accs.append(score.accuracy_mean)
            logger.info(f"  Seed {seed}: {score.accuracy_mean:.2f}%")

        import numpy as np
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        elapsed = time.time() - start

        final_score = ActivationScore(
            expression=name,
            accuracy_mean=mean_acc,
            accuracy_std=std_acc,
            accuracies=accs,
            forward_time_ms=score.forward_time_ms,
            tree_nodes=1,  # Proste aktywacje = 1 węzeł
        )
        final_score.composite_score = compute_composite_score(final_score)

        leaderboard.add_baseline(name, final_score)
        logger.info(f"  → {name}: {mean_acc:.2f}% ± {std_acc:.2f}  "
                    f"({elapsed:.0f}s)")

    leaderboard.report()
    logger.info("Baselines saved to leaderboard.json")


if __name__ == "__main__":
    main()
