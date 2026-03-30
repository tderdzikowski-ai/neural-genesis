"""
Neural Genesis — Expression Tree Serializer

Serializacja drzew wyrażeń do/z JSON-a.
Potrzebne do zapisywania najlepszych kandydatów i leaderboardu.
"""

from __future__ import annotations
from typing import Any

from expression.nodes import (
    ExpressionNode, InputNode, ConstantNode, ParameterNode,
    UnaryNode, BinaryNode,
)


def tree_to_dict(node: ExpressionNode) -> dict[str, Any]:
    """Konwertuje drzewo do słownika (serializable to JSON)."""

    if isinstance(node, InputNode):
        return {"type": "input"}

    elif isinstance(node, ConstantNode):
        return {
            "type": "constant",
            "value": node.value,
            "name": node.name,
        }

    elif isinstance(node, ParameterNode):
        return {
            "type": "parameter",
            "param_name": node.param_name,
            "init_value": node.init_value,
        }

    elif isinstance(node, UnaryNode):
        return {
            "type": "unary",
            "op": node.op,
            "child": tree_to_dict(node.child),
        }

    elif isinstance(node, BinaryNode):
        return {
            "type": "binary",
            "op": node.op,
            "left": tree_to_dict(node.left),
            "right": tree_to_dict(node.right),
        }

    raise ValueError(f"Unknown node type: {type(node)}")


def dict_to_tree(d: dict[str, Any]) -> ExpressionNode:
    """Odtwarza drzewo ze słownika."""

    node_type = d["type"]

    if node_type == "input":
        return InputNode()

    elif node_type == "constant":
        return ConstantNode(value=d["value"], name=d.get("name", ""))

    elif node_type == "parameter":
        return ParameterNode(
            param_name=d["param_name"],
            init_value=d.get("init_value", 1.0),
        )

    elif node_type == "unary":
        child = dict_to_tree(d["child"])
        return UnaryNode(op=d["op"], child=child)

    elif node_type == "binary":
        left = dict_to_tree(d["left"])
        right = dict_to_tree(d["right"])
        return BinaryNode(op=d["op"], left=left, right=right)

    raise ValueError(f"Unknown node type in dict: {node_type}")


def tree_to_string(node: ExpressionNode) -> str:
    """Czytelna reprezentacja tekstowa (deleguje do node.to_string)."""
    return node.to_string()
