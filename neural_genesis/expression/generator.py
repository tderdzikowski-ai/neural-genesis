"""
Neural Genesis — Random Expression Generator

Generuje losowe drzewa wyrażeń z kontrolowaną głębokością,
walidacją (wyrażenie MUSI zależeć od x) i deduplikacją.
"""

from __future__ import annotations
import random
import copy
from dataclasses import dataclass

from expression.nodes import (
    ExpressionNode, InputNode, ConstantNode, ParameterNode,
    UnaryNode, BinaryNode,
    CONSTANTS, PARAMETERS, UNARY_OPS, BINARY_OPS,
)
from config import EXPRESSION


# =============================================================================
# Core generator
# =============================================================================

class ExpressionGenerator:
    """
    Generates random expression trees.

    Usage:
        gen = ExpressionGenerator(seed=42)
        tree = gen.generate()            # Jedno wyrażenie
        trees = gen.generate_batch(100)  # 100 unikalnych wyrażeń
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.seen_hashes: set[str] = set()
        self._param_counter = 0

    def generate(self, max_attempts: int = 50) -> ExpressionNode | None:
        """
        Generuje jedno losowe, zwalidowane wyrażenie.
        Zwraca None jeśli nie uda się wygenerować poprawnego
        wyrażenia w max_attempts próbach.
        """
        for _ in range(max_attempts):
            self._param_counter = 0
            tree = self._generate_node(
                max_depth=EXPRESSION["max_depth"],
                current_depth=0,
            )

            # Walidacja
            if not self._validate(tree):
                continue

            # Deduplikacja
            h = tree.structural_hash()
            if h in self.seen_hashes:
                continue

            self.seen_hashes.add(h)
            return tree

        return None

    def generate_batch(self, count: int, max_attempts_per: int = 50) -> list[ExpressionNode]:
        """Generuje batch unikalnych wyrażeń."""
        results = []
        for _ in range(count * 3):  # 3x nadmiar na wypadek odrzuceń
            tree = self.generate(max_attempts=max_attempts_per)
            if tree is not None:
                results.append(tree)
            if len(results) >= count:
                break
        return results

    # -------------------------------------------------------------------------
    # Internal: tree building
    # -------------------------------------------------------------------------

    def _generate_node(self, max_depth: int, current_depth: int) -> ExpressionNode:
        """Rekurencyjnie buduje losowy węzeł drzewa."""

        # Prawdopodobieństwo liścia rośnie z głębokością
        p_leaf = (
            EXPRESSION["p_leaf_base"]
            + EXPRESSION["p_leaf_depth_scale"] * (current_depth / max(max_depth, 1))
        )

        if current_depth >= max_depth or self.rng.random() < p_leaf:
            return self._random_leaf()

        if self.rng.random() < (1.0 - EXPRESSION["p_binary"]):
            # Operacja unarna
            op = self.rng.choice(UNARY_OPS)
            child = self._generate_node(max_depth, current_depth + 1)
            return UnaryNode(op=op, child=child)
        else:
            # Operacja binarna
            op = self.rng.choice(BINARY_OPS)
            left = self._generate_node(max_depth, current_depth + 1)
            right = self._generate_node(max_depth, current_depth + 1)
            return BinaryNode(op=op, left=left, right=right)

    def _random_leaf(self) -> ExpressionNode:
        """Wybiera losowy liść z odpowiednim rozkładem."""
        r = self.rng.random()

        if r < EXPRESSION["p_input"]:
            return InputNode()
        elif r < EXPRESSION["p_input"] + EXPRESSION["p_constant"]:
            const = self.rng.choice(CONSTANTS)
            return ConstantNode(value=const.value, name=const.name)
        else:
            # Parametr uczony — ale max EXPRESSION["max_params"]
            if self._param_counter >= EXPRESSION["max_params"]:
                # Fallback na input lub stałą
                if self.rng.random() < 0.6:
                    return InputNode()
                const = self.rng.choice(CONSTANTS)
                return ConstantNode(value=const.value, name=const.name)

            param = PARAMETERS[self._param_counter]
            self._param_counter += 1
            return ParameterNode(
                param_name=param.param_name,
                init_value=param.init_value,
            )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate(self, tree: ExpressionNode) -> bool:
        """
        Sprawdza czy wyrażenie jest sensowne:
        1. Zawiera x (zależy od wejścia)
        2. Nie jest za duże
        3. Nie jest trywialne (sam x)
        4. Ma minimalną głębokość
        """
        # Musi zawierać input
        if not tree.contains_input():
            return False

        # Ograniczenie rozmiaru
        if tree.node_count() > EXPRESSION["max_nodes"]:
            return False

        # Nie za płytkie (sam 'x' lub 'relu(x)' to znane rzeczy)
        if tree.depth() < EXPRESSION["min_depth"]:
            return False

        # Nie może być trywialne identity
        if isinstance(tree, InputNode):
            return False
        if isinstance(tree, UnaryNode) and tree.op == "identity":
            return False

        return True


# =============================================================================
# Convenience functions
# =============================================================================

def generate_random_expression(seed: int | None = None) -> ExpressionNode | None:
    """Wygeneruj jedno losowe wyrażenie (convenience wrapper)."""
    gen = ExpressionGenerator(seed=seed)
    return gen.generate()
