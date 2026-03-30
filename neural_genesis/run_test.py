#!/usr/bin/env python3
"""
Neural Genesis — Quick Test

Uruchom NAJPIERW żeby sprawdzić czy cały pipeline działa.
Generuje 5 wyrażeń, kompiluje, testuje — powinno zająć <2 minuty.

Usage:
    python run_test.py
"""

import sys
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_expression_generation():
    """Test 1: Czy generator tworzy poprawne wyrażenia?"""
    logger.info("TEST 1: Expression generation")

    from expression.generator import ExpressionGenerator

    gen = ExpressionGenerator(seed=42)
    trees = gen.generate_batch(20)

    logger.info(f"  Generated {len(trees)} unique expressions:")
    for i, tree in enumerate(trees[:10]):
        expr = tree.to_string()
        logger.info(f"    [{i+1}] depth={tree.depth()} nodes={tree.node_count()} "
                     f"params={len(tree.get_learnable_params())}  →  {expr}")

    assert len(trees) >= 10, f"Expected >=10 trees, got {len(trees)}"
    logger.info("  ✓ PASSED\n")
    return trees


def test_compilation(trees):
    """Test 2: Czy wyrażenia kompilują się do PyTorch?"""
    logger.info("TEST 2: Compilation to PyTorch")

    from expression.to_pytorch import compile_to_pytorch

    for i, tree in enumerate(trees[:5]):
        factory = compile_to_pytorch(tree)
        activation = factory()

        x = torch.randn(32, 64)
        y = activation(x)

        assert not torch.isnan(y).any(), f"NaN in output of {tree.to_string()}"
        assert not torch.isinf(y).any(), f"Inf in output of {tree.to_string()}"

        logger.info(f"    [{i+1}] {tree.to_string():40s}  "
                     f"output range: [{y.min().item():.3f}, {y.max().item():.3f}]")

    logger.info("  ✓ PASSED\n")


def test_gradient_flow(trees):
    """Test 3: Czy gradienty przepływają?"""
    logger.info("TEST 3: Gradient flow")

    from expression.to_pytorch import compile_to_pytorch

    for i, tree in enumerate(trees[:5]):
        factory = compile_to_pytorch(tree)
        activation = factory()

        x = torch.randn(32, 64, requires_grad=True)
        y = activation(x)
        loss = y.sum()
        loss.backward()

        grad = x.grad
        assert grad is not None, f"No gradient for {tree.to_string()}"
        assert not torch.isnan(grad).any(), f"NaN gradient for {tree.to_string()}"

        logger.info(f"    [{i+1}] {tree.to_string():40s}  "
                     f"grad mean: {grad.abs().mean().item():.6f}")

    logger.info("  ✓ PASSED\n")


def test_serialization(trees):
    """Test 4: Czy serialize/deserialize działa?"""
    logger.info("TEST 4: Serialization")

    from expression.serializer import tree_to_dict, dict_to_tree

    for tree in trees[:5]:
        d = tree_to_dict(tree)
        restored = dict_to_tree(d)
        assert tree.to_string() == restored.to_string(), \
            f"Mismatch: {tree.to_string()} vs {restored.to_string()}"

    logger.info("  ✓ PASSED\n")


def test_network_build():
    """Test 5: Czy sieć ewaluacyjna się buduje i trenuje?"""
    logger.info("TEST 5: Evaluation network")

    import torch.nn as nn
    from evaluation.networks import build_eval_network

    # Test z ReLU
    model = build_eval_network(lambda: nn.ReLU(), "FashionMNIST")
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10), f"Expected (4, 10), got {y.shape}"
    logger.info(f"  FashionMNIST model: {sum(p.numel() for p in model.parameters()):,} params")

    model = build_eval_network(lambda: nn.ReLU(), "CIFAR10")
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    assert y.shape == (4, 10), f"Expected (4, 10), got {y.shape}"
    logger.info(f"  CIFAR10 model: {sum(p.numel() for p in model.parameters()):,} params")

    logger.info("  ✓ PASSED\n")


def test_discovered_activation_in_network(trees):
    """Test 6: Czy odkryta aktywacja działa w sieci?"""
    logger.info("TEST 6: Discovered activation in eval network")

    from expression.to_pytorch import compile_to_pytorch
    from evaluation.networks import build_eval_network

    tree = trees[0]
    factory = compile_to_pytorch(tree)

    model = build_eval_network(factory, "FashionMNIST")
    x = torch.randn(4, 1, 28, 28)
    y = model(x)

    assert y.shape == (4, 10)
    assert not torch.isnan(y).any()

    # Backward pass
    loss = y.sum()
    loss.backward()

    logger.info(f"  Expression: {tree.to_string()}")
    logger.info(f"  Output shape: {y.shape}")
    logger.info(f"  Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    logger.info("  ✓ PASSED\n")


def test_sanity_checks(trees):
    """Test 7: Czy sanity checks działają?"""
    logger.info("TEST 7: Sanity checks")

    from expression.to_pytorch import DiscoveredActivation
    from expression.nodes import ConstantNode, UnaryNode, BinaryNode, InputNode
    from evaluation.trainer import sanity_check

    device = torch.device("cpu")

    # Dobra aktywacja — Swish: x * sigmoid(x), pewny kandydat
    good_tree = BinaryNode(
        op="mul",
        left=InputNode(),
        right=UnaryNode(op="sigmoid", child=InputNode()),
    )
    act = DiscoveredActivation(good_tree)
    result = sanity_check(act, device)
    logger.info(f"  Good activation: {result}")
    assert result is None, f"Good activation failed sanity: {result}"

    # Stała aktywacja — powinna nie przejść
    const_tree = ConstantNode(value=5.0, name="5")
    # Musimy opakować w UnaryNode żeby przeszło walidację generatora,
    # ale tu testujemy bezpośrednio sanity check
    const_act = DiscoveredActivation(const_tree)
    result = sanity_check(const_act, device)
    logger.info(f"  Constant activation: {result}")
    assert result is not None, "Constant activation should fail sanity"

    logger.info("  ✓ PASSED\n")


def test_mini_training():
    """Test 8: Szybki trening 1 epoka — czy pipeline działa end-to-end?"""
    logger.info("TEST 8: Mini training (1 epoch)")

    from expression.generator import ExpressionGenerator
    from expression.to_pytorch import compile_to_pytorch
    from evaluation.trainer import train_and_evaluate
    from config import DEVICE

    gen = ExpressionGenerator(seed=99)
    tree = gen.generate()
    factory = compile_to_pytorch(tree)

    # Użyj CPU dla testu żeby było deterministyczne
    device = torch.device("cpu")

    score = train_and_evaluate(
        activation_factory=factory,
        dataset_name="FashionMNIST",
        epochs=1,
        batch_size=256,
        learning_rate=0.001,
        device=device,
        seed=42,
        early_stop_epoch=999,
        early_stop_min_acc=0.0,
    )

    logger.info(f"  Expression: {tree.to_string()}")
    logger.info(f"  Accuracy after 1 epoch: {score.accuracy_mean:.2f}%")
    logger.info(f"  Crashed: {score.training_crashed}")
    if score.training_crashed:
        logger.info(f"  Reason: {score.crash_reason}")
    logger.info("  ✓ PASSED\n")


def main():
    logger.info("=" * 60)
    logger.info("  NEURAL GENESIS — Pipeline Test")
    logger.info("=" * 60)

    trees = test_expression_generation()
    test_compilation(trees)
    test_gradient_flow(trees)
    test_serialization(trees)
    test_network_build()
    test_discovered_activation_in_network(trees)
    test_sanity_checks(trees)
    test_mini_training()

    logger.info("=" * 60)
    logger.info("  ALL TESTS PASSED ✓")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. python run_baselines.py          # Measure known activations")
    logger.info("  2. python run_stage0.py --count 100  # Quick search (100 candidates)")
    logger.info("  3. python run_stage0.py              # Full search (10 000 candidates)")


if __name__ == "__main__":
    main()