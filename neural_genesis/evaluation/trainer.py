"""
Neural Genesis — Trainer

Pętla treningowa z:
  - Safety checks (NaN, Inf, timeout)
  - Early stopping (nie trać czasu na beznadziejnych kandydatów)
  - Zbieranie metryk gradientów
  - Pomiar czasu forward pass
"""

from __future__ import annotations
import math
import time
import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from expression.nodes import ExpressionNode
from expression.to_pytorch import compile_to_pytorch, DiscoveredActivation
from evaluation.networks import build_eval_network
from evaluation.datasets import get_data_loaders
from evaluation.metrics import ActivationScore, FAILED_SCORE, compute_composite_score
from config import DEVICE, SAFETY, SCREENING, FULL_EVAL

logger = logging.getLogger(__name__)


# =============================================================================
# Sanity checks — przed trenowaniem
# =============================================================================

def sanity_check(activation: DiscoveredActivation, device: torch.device) -> Optional[str]:
    """
    Szybki test czy aktywacja produkuje sensowne wartości.
    Zwraca None jeśli OK, albo string z opisem problemu.
    """
    activation = activation.to(device)

    test_input = torch.randn(256, device=device)
    try:
        test_output = activation(test_input)
    except Exception as e:
        return f"forward_error: {e}"

    # NaN / Inf
    if torch.isnan(test_output).any():
        return "output_nan"
    if torch.isinf(test_output).any():
        return "output_inf"

    # Zakres — czy wartości nie są absurdalnie duże?
    if test_output.abs().max().item() > SAFETY["max_output_value"]:
        return f"output_too_large: {test_output.abs().max().item():.1f}"

    # Czy output zależy od inputu? (nie jest stały)
    if test_output.std().item() < SAFETY["min_output_std"]:
        return "output_constant"

    # Gradient check
    test_input_grad = torch.randn(256, device=device, requires_grad=True)
    try:
        out = activation(test_input_grad)
        grad = torch.autograd.grad(
            out.sum(), test_input_grad, create_graph=False
        )[0]
    except Exception as e:
        return f"gradient_error: {e}"

    if torch.isnan(grad).any():
        return "gradient_nan"
    if grad.abs().max().item() > SAFETY["max_gradient"]:
        return f"gradient_too_large: {grad.abs().max().item():.1f}"
    if grad.abs().max().item() < 1e-10:
        return "gradient_zero"

    return None  # All checks passed


# =============================================================================
# Training loop
# =============================================================================

def train_and_evaluate(
    activation_factory: Callable[[], nn.Module],
    dataset_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int = 42,
    early_stop_epoch: int = 2,
    early_stop_min_acc: float = 15.0,
) -> ActivationScore:
    """
    Trenuje sieć z daną aktywacją i zwraca pełny profil metryk.
    """
    # Reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # Data
    train_loader, test_loader = get_data_loaders(dataset_name, batch_size)

    # Model
    model = build_eval_network(activation_factory, dataset_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Metryki
    epochs_to_90 = None
    best_test_acc = 0.0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # NaN check na output
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                return _make_failed_score("nan_during_training", epoch)

            loss = criterion(outputs, targets)

            if math.isnan(loss.item()):
                return _make_failed_score("nan_loss", epoch)
            if loss.item() > SAFETY["max_train_loss"]:
                return _make_failed_score(f"loss_exploded: {loss.item():.1f}", epoch)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        # Early stopping — nie trać czasu na beznadziejnych
        if epoch == early_stop_epoch and train_acc < early_stop_min_acc:
            return _make_failed_score(
                f"too_slow: {train_acc:.1f}% after {epoch + 1} epochs", epoch
            )

        # Test accuracy
        test_acc = _evaluate_accuracy(model, test_loader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if test_acc >= 90.0 and epochs_to_90 is None:
            epochs_to_90 = epoch + 1

    # Zbierz metryki gradientów
    grad_stats = _collect_gradient_stats(model)

    # Zmierz czas forward pass
    forward_time = _measure_forward_time(activation_factory, device)

    return ActivationScore(
        accuracy_mean=best_test_acc,
        accuracies=[best_test_acc],
        epochs_to_90pct=epochs_to_90,
        final_train_loss=train_loss,
        grad_mean=grad_stats["mean"],
        grad_std=grad_stats["std"],
        grad_vanish_ratio=grad_stats["vanish_ratio"],
        grad_explode_ratio=grad_stats["explode_ratio"],
        forward_time_ms=forward_time,
    )


# =============================================================================
# Top-level safe evaluation
# =============================================================================

def safe_evaluate(
    expression_tree: ExpressionNode,
    phase: str = "screening",
) -> ActivationScore:
    """
    Bezpieczna ewaluacja: sanity checks → trening → scoring.
    Nigdy nie rzuca wyjątkiem — zwraca FAILED_SCORE w razie problemu.

    Args:
        expression_tree: Drzewo wyrażenia do przetestowania.
        phase: "screening" lub "full_eval"
    """
    device = DEVICE
    cfg = SCREENING if phase == "screening" else FULL_EVAL

    # 1. Kompiluj do PyTorch
    try:
        factory = compile_to_pytorch(expression_tree)
        test_activation = factory()
    except Exception as e:
        return _make_failed_score(f"compile_error: {e}")

    # 2. Sanity check
    problem = sanity_check(test_activation, device)
    if problem is not None:
        return _make_failed_score(f"sanity: {problem}")

    # 3. Trenuj i ewaluuj
    seeds = cfg.get("seeds", [42]) if phase == "full_eval" else [42]
    if cfg["num_seeds"] == 1:
        seeds = [42]

    all_scores: list[ActivationScore] = []
    for seed in seeds[:cfg["num_seeds"]]:
        try:
            score = train_and_evaluate(
                activation_factory=factory,
                dataset_name=cfg["dataset"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                device=device,
                seed=seed,
                early_stop_epoch=cfg.get("early_stop_epoch", 2),
                early_stop_min_acc=cfg.get("early_stop_min_acc", 15.0),
            )
            if score.training_crashed:
                return score
            all_scores.append(score)
        except Exception as e:
            logger.warning(f"Training exception: {e}")
            return _make_failed_score(f"train_exception: {e}")

    # 4. Agreguj wyniki z wielu seedów
    merged = _merge_scores(all_scores, expression_tree)
    merged.composite_score = compute_composite_score(merged)
    return merged


# =============================================================================
# Helpers
# =============================================================================

def _evaluate_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Oblicza test accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def _collect_gradient_stats(model: nn.Module) -> dict:
    """Zbiera statystyki gradientów z parametrów modelu."""
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.data.abs().mean().item())

    if not grad_norms:
        return {"mean": 0.0, "std": 0.0, "vanish_ratio": 1.0, "explode_ratio": 0.0}

    import numpy as np
    arr = np.array(grad_norms)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "vanish_ratio": float((arr < 1e-7).mean()),
        "explode_ratio": float((arr > 100).mean()),
    }


def _measure_forward_time(
    activation_factory: Callable,
    device: torch.device,
    num_runs: int = 100,
) -> float:
    """Mierzy średni czas forward pass aktywacji [ms]."""
    act = activation_factory().to(device)
    x = torch.randn(1024, device=device)

    # Warmup
    for _ in range(10):
        act(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        act(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / num_runs * 1000  # ms


def _merge_scores(
    scores: list[ActivationScore],
    tree: ExpressionNode,
) -> ActivationScore:
    """Łączy wyniki z wielu seedów w jeden score."""
    import numpy as np

    accs = [s.accuracy_mean for s in scores]

    merged = ActivationScore(
        expression=tree.to_string(),
        tree_hash=tree.structural_hash(),
        tree_depth=tree.depth(),
        tree_nodes=tree.node_count(),
        num_params=len(tree.get_learnable_params()),
        accuracy_mean=float(np.mean(accs)),
        accuracy_std=float(np.std(accs)) if len(accs) > 1 else 0.0,
        accuracies=accs,
        epochs_to_90pct=scores[0].epochs_to_90pct,
        final_train_loss=float(np.mean([s.final_train_loss for s in scores])),
        grad_mean=float(np.mean([s.grad_mean for s in scores])),
        grad_std=float(np.mean([s.grad_std for s in scores])),
        grad_vanish_ratio=float(np.mean([s.grad_vanish_ratio for s in scores])),
        grad_explode_ratio=float(np.mean([s.grad_explode_ratio for s in scores])),
        forward_time_ms=scores[0].forward_time_ms,
    )
    return merged


def _make_failed_score(reason: str, epoch: int = -1) -> ActivationScore:
    """Tworzy wynik dla kandydata, który nie przeszedł."""
    return ActivationScore(
        training_crashed=True,
        crash_reason=reason,
        composite_score=-1.0,
    )
