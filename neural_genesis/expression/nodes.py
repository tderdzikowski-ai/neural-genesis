"""
Neural Genesis — Expression Tree Nodes

Drzewo wyrażeń to Abstract Syntax Tree (AST) reprezentujący funkcję aktywacji.
Każdy węzeł jest jednym z typów:
  - InputNode:     wejście neuronu (x)
  - ConstantNode:  stała matematyczna (0, 1, π, e, ...)
  - ParameterNode: uczony parametr (α, β, γ)
  - UnaryNode:     operacja jednoargumentowa (sin, exp, sigmoid, ...)
  - BinaryNode:    operacja dwuargumentowa (+, *, max, ...)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import hashlib


# =============================================================================
# Base class
# =============================================================================

@dataclass
class ExpressionNode:
    """Bazowa klasa dla wszystkich węzłów drzewa wyrażeń."""

    def depth(self) -> int:
        """Głębokość poddrzewa zakorzenionego w tym węźle."""
        raise NotImplementedError

    def node_count(self) -> int:
        """Liczba węzłów w poddrzewie."""
        raise NotImplementedError

    def contains_input(self) -> bool:
        """Czy poddrzewo zawiera InputNode (x)?"""
        raise NotImplementedError

    def get_learnable_params(self) -> list[str]:
        """Lista nazw uczonych parametrów w poddrzewie."""
        raise NotImplementedError

    def to_string(self) -> str:
        """Czytelna reprezentacja tekstowa wyrażenia."""
        raise NotImplementedError

    def structural_hash(self) -> str:
        """Hash strukturalny do deduplikacji (ignoruje wartości stałych)."""
        h = hashlib.md5(self.to_string().encode()).hexdigest()[:12]
        return h


# =============================================================================
# Leaf nodes
# =============================================================================

@dataclass
class InputNode(ExpressionNode):
    """Wejście neuronu — x."""

    def depth(self) -> int:
        return 0

    def node_count(self) -> int:
        return 1

    def contains_input(self) -> bool:
        return True

    def get_learnable_params(self) -> list[str]:
        return []

    def to_string(self) -> str:
        return "x"


@dataclass
class ConstantNode(ExpressionNode):
    """Stała matematyczna."""

    value: float
    name: str = ""  # Czytelna nazwa, np. "pi", "e"

    def depth(self) -> int:
        return 0

    def node_count(self) -> int:
        return 1

    def contains_input(self) -> bool:
        return False

    def get_learnable_params(self) -> list[str]:
        return []

    def to_string(self) -> str:
        if self.name:
            return self.name
        if self.value == int(self.value):
            return str(int(self.value))
        return f"{self.value:.4f}"


@dataclass
class ParameterNode(ExpressionNode):
    """Uczony parametr skalarny (optymalizowany razem z siecią)."""

    param_name: str        # "alpha", "beta", "gamma"
    init_value: float = 1.0  # Wartość inicjalna

    def depth(self) -> int:
        return 0

    def node_count(self) -> int:
        return 1

    def contains_input(self) -> bool:
        return False

    def get_learnable_params(self) -> list[str]:
        return [self.param_name]

    def to_string(self) -> str:
        return self.param_name


# =============================================================================
# Internal nodes
# =============================================================================

@dataclass
class UnaryNode(ExpressionNode):
    """Operacja jednoargumentowa: op(child)."""

    op: str                     # Nazwa operacji, np. "sigmoid", "sin"
    child: ExpressionNode

    def depth(self) -> int:
        return 1 + self.child.depth()

    def node_count(self) -> int:
        return 1 + self.child.node_count()

    def contains_input(self) -> bool:
        return self.child.contains_input()

    def get_learnable_params(self) -> list[str]:
        return self.child.get_learnable_params()

    def to_string(self) -> str:
        return f"{self.op}({self.child.to_string()})"


@dataclass
class BinaryNode(ExpressionNode):
    """Operacja dwuargumentowa: op(left, right)."""

    op: str                     # Nazwa operacji, np. "mul", "add"
    left: ExpressionNode
    right: ExpressionNode

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self) -> int:
        return 1 + self.left.node_count() + self.right.node_count()

    def contains_input(self) -> bool:
        return self.left.contains_input() or self.right.contains_input()

    def get_learnable_params(self) -> list[str]:
        return self.left.get_learnable_params() + self.right.get_learnable_params()

    def to_string(self) -> str:
        # Ładniejszy zapis dla podstawowych operacji arytmetycznych
        infix_ops = {
            "add": "+", "sub": "-", "mul": "*", "div_safe": "/",
        }
        if self.op in infix_ops:
            symbol = infix_ops[self.op]
            return f"({self.left.to_string()} {symbol} {self.right.to_string()})"
        return f"{self.op}({self.left.to_string()}, {self.right.to_string()})"


# =============================================================================
# Available constants
# =============================================================================

import math

CONSTANTS = [
    ConstantNode(value=0.0,    name="0"),
    ConstantNode(value=1.0,    name="1"),
    ConstantNode(value=-1.0,   name="-1"),
    ConstantNode(value=0.5,    name="0.5"),
    ConstantNode(value=2.0,    name="2"),
    ConstantNode(value=math.pi, name="pi"),
    ConstantNode(value=math.e,  name="e"),
    ConstantNode(value=(1 + math.sqrt(5)) / 2, name="phi"),
]

# =============================================================================
# Available learnable parameters
# =============================================================================

PARAMETERS = [
    ParameterNode(param_name="alpha", init_value=1.0),
    ParameterNode(param_name="beta",  init_value=1.0),
    ParameterNode(param_name="gamma", init_value=0.5),
]

# =============================================================================
# Available operations
# =============================================================================

UNARY_OPS = [
    "identity", "negate", "abs", "square", "cube",
    "sqrt_abs", "sin", "cos", "exp_safe", "log_safe",
    "sigmoid", "tanh", "relu", "sign", "inv_safe",
    "gaussian", "softplus", "erf_safe", "softsign", "arctan", "sinc_safe",
    "gcu", "ncu", "hermite", "mexican_hat", "logcosh", "sig_deriv",
]

BINARY_OPS = [
    "add", "sub", "mul", "div_safe",
    "max", "min", "pow_safe",
]
