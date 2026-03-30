"""
Neural Genesis — Baseline Activations

Znane funkcje aktywacji do porównania.
Mierzymy je dokładnie tą samą siecią i tymi samymi danymi.
"""

import torch.nn as nn


BASELINES = {
    "ReLU":         lambda: nn.ReLU(),
    "LeakyReLU":    lambda: nn.LeakyReLU(0.01),
    "ELU":          lambda: nn.ELU(),
    "SELU":         lambda: nn.SELU(),
    "GELU":         lambda: nn.GELU(),
    "SiLU_Swish":   lambda: nn.SiLU(),
    "Mish":         lambda: nn.Mish(),
    "Sigmoid":      lambda: nn.Sigmoid(),
    "Tanh":         lambda: nn.Tanh(),
    "Softplus":     lambda: nn.Softplus(),
}
