"""
Neural Genesis — Expression Tree to PyTorch Compiler

Konwertuje drzewo wyrażeń na nn.Module, który można użyć
jako funkcję aktywacji w dowolnej sieci PyTorch.
Parametry uczone (α, β, γ) stają się nn.Parameter.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from expression.nodes import (
    ExpressionNode, InputNode, ConstantNode, ParameterNode,
    UnaryNode, BinaryNode,
)
from expression.operations import UNARY_FN, BINARY_FN
from config import SAFETY


class DiscoveredActivation(nn.Module):
    """
    Funkcja aktywacji skompilowana z drzewa wyrażeń.

    Usage:
        tree = BinaryNode("mul", InputNode(), UnaryNode("sigmoid", InputNode()))
        act = DiscoveredActivation(tree)
        output = act(torch.randn(32, 64))  # Działa jak nn.ReLU()
    """

    def __init__(self, expression_tree: ExpressionNode):
        super().__init__()
        self.tree = expression_tree
        self.expression_string = expression_tree.to_string()

        # Rejestruj parametry uczone jako nn.Parameter
        self._param_names: list[str] = []
        for param_name in expression_tree.get_learnable_params():
            if param_name not in self._param_names:
                self._param_names.append(param_name)
                self.register_parameter(
                    param_name,
                    nn.Parameter(torch.tensor(1.0)),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self._evaluate(self.tree, x)
        # Safety clamp na wyjściu — zapobiegamy eksplozji wartości
        return torch.clamp(result, -SAFETY["output_clamp"], SAFETY["output_clamp"])

    def _evaluate(self, node: ExpressionNode, x: torch.Tensor) -> torch.Tensor:
        if isinstance(node, InputNode):
            return x

        elif isinstance(node, ConstantNode):
            return torch.full_like(x, node.value)

        elif isinstance(node, ParameterNode):
            param = getattr(self, node.param_name)
            return param.expand_as(x)

        elif isinstance(node, UnaryNode):
            child_val = self._evaluate(node.child, x)
            fn = UNARY_FN[node.op]
            return fn(child_val)

        elif isinstance(node, BinaryNode):
            left_val = self._evaluate(node.left, x)
            right_val = self._evaluate(node.right, x)
            fn = BINARY_FN[node.op]
            return fn(left_val, right_val)

        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def extra_repr(self) -> str:
        return f"expr='{self.expression_string}'"


class ActivationFactory:
    """
    Tworzy fabrykę (callable) z drzewa wyrażeń.

    Potrzebne bo nn.Sequential wymaga callable, które tworzy
    NOWE instancje aktywacji (osobne parametry per warstwa).

    Usage:
        factory = ActivationFactory(tree)
        layer1_act = factory()  # Nowa instancja
        layer2_act = factory()  # Inna instancja (osobne parametry)
    """

    def __init__(self, expression_tree: ExpressionNode):
        self.tree = expression_tree

    def __call__(self) -> DiscoveredActivation:
        return DiscoveredActivation(self.tree)


def compile_to_pytorch(expression_tree: ExpressionNode) -> ActivationFactory:
    """
    Kompiluje drzewo wyrażeń do fabryki aktywacji PyTorch.

    Returns:
        ActivationFactory — callable, które tworzy nn.Module aktywacji.
    """
    return ActivationFactory(expression_tree)
