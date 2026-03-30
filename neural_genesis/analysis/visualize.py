"""
Neural Genesis — Visualization

Generuje wykresy dla odkrytych funkcji aktywacji:
  1. Kształt funkcji f(x) vs x
  2. Pochodna f'(x)
  3. Porównanie z baseline'ami
"""

from __future__ import annotations
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Bez okienka GUI

from expression.nodes import ExpressionNode
from expression.to_pytorch import compile_to_pytorch
from config import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"


def plot_activation(
    tree: ExpressionNode,
    save_path: Path | None = None,
    x_range: tuple[float, float] = (-5.0, 5.0),
    num_points: int = 1000,
):
    """
    Rysuje kształt funkcji aktywacji i jej pochodnej.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    factory = compile_to_pytorch(tree)
    activation = factory()

    x = torch.linspace(x_range[0], x_range[1], num_points, requires_grad=True)
    y = activation(x)

    # Pochodna
    grad = torch.autograd.grad(y.sum(), x, create_graph=False)[0]

    x_np = x.detach().numpy()
    y_np = y.detach().numpy()
    grad_np = grad.detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Activation: {tree.to_string()}", fontsize=12, fontweight="bold")

    # Kształt funkcji
    ax1.plot(x_np, y_np, "b-", linewidth=2, label="f(x)")
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax1.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("Activation Function")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-10, 10)

    # Pochodna
    ax2.plot(x_np, grad_np, "r-", linewidth=2, label="f'(x)")
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.axhline(y=1, color="green", linewidth=0.5, linestyle=":", label="grad=1")
    ax2.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f'(x)")
    ax2.set_title("Derivative (Gradient Flow)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-3, 3)

    plt.tight_layout()

    if save_path is None:
        save_path = PLOTS_DIR / f"activation_{tree.structural_hash()}.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_comparison(
    discovered: list[tuple[str, ExpressionNode]],
    save_path: Path | None = None,
):
    """
    Porównanie odkrytych aktywacji z baseline'ami na jednym wykresie.
    """
    import torch.nn as nn

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    x = torch.linspace(-5, 5, 1000)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Baseline'y — szare, cienkie
    baselines = {
        "ReLU": nn.ReLU(),
        "GELU": nn.GELU(),
        "SiLU (Swish)": nn.SiLU(),
    }
    for name, act in baselines.items():
        y = act(x).detach().numpy()
        ax.plot(x.numpy(), y, "--", linewidth=1, alpha=0.5, label=f"{name} (baseline)")

    # Odkryte — kolorowe, grube
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(discovered), 1)))
    for i, (name, tree) in enumerate(discovered[:5]):
        factory = compile_to_pytorch(tree)
        act = factory()
        y = act(x).detach().numpy()
        ax.plot(x.numpy(), y, "-", linewidth=2.5, color=colors[i],
                label=f"#{i+1}: {name[:40]}")

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.set_title("Neural Genesis — Discovered vs Baseline Activations", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 5)

    plt.tight_layout()

    if save_path is None:
        save_path = PLOTS_DIR / "comparison.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
