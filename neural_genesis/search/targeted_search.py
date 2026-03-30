"""
Neural Genesis — Targeted Search

Ukierunkowane przeszukiwanie: testuje ręcznie zaprojektowane warianty
obiecujących wzorców (QuadGate, periodic, residual).
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path

from expression.nodes import (
    InputNode, ConstantNode, ParameterNode,
    UnaryNode, BinaryNode,
)
from expression.serializer import tree_to_dict
from evaluation.trainer import safe_evaluate
from evaluation.metrics import compute_composite_score
from config import RESULTS_DIR

logger = logging.getLogger(__name__)

X = InputNode()
ALPHA = ParameterNode("alpha")
BETA = ParameterNode("beta")
GAMMA = ParameterNode("gamma")


def _const(val, name=""):
    return ConstantNode(value=val, name=name or str(val))


def _un(op, child):
    return UnaryNode(op=op, child=child)


def _bin(op, left, right):
    return BinaryNode(op=op, left=left, right=right)


# =============================================================================
# Warianty QuadGate
# =============================================================================

def quadgate_variants():
    """Rodzina QuadGate — warianty σ²(α-x)·x"""
    variants = {}

    # Original: σ²(α-x)·x
    variants["QG_original"] = _bin("mul",
        _un("square", _un("sigmoid", _bin("sub", ALPHA, X))),
        X
    )

    # σ³(α-x)·x — cube zamiast square
    variants["QG_cube"] = _bin("mul",
        _un("cube", _un("sigmoid", _bin("sub", ALPHA, X))),
        X
    )

    # σ²(α-βx)·x — skalowane x
    variants["QG_scaled"] = _bin("mul",
        _un("square", _un("sigmoid", _bin("sub", ALPHA, _bin("mul", BETA, X)))),
        X
    )

    # σ²(α-x)·x + γx — residual connection
    variants["QG_residual"] = _bin("add",
        _bin("mul",
            _un("square", _un("sigmoid", _bin("sub", ALPHA, X))),
            X
        ),
        _bin("mul", GAMMA, X)
    )

    # σ(α-x)·x — bez kwadratu (prostsze)
    variants["QG_linear"] = _bin("mul",
        _un("sigmoid", _bin("sub", ALPHA, X)),
        X
    )

    # tanh²(α-x)·x — tanh zamiast sigmoid
    variants["QG_tanh"] = _bin("mul",
        _un("square", _un("tanh", _bin("sub", ALPHA, X))),
        X
    )

    # σ²(α-x)·x + x — skip connection (identity + gate)
    variants["QG_skip"] = _bin("add",
        _bin("mul",
            _un("square", _un("sigmoid", _bin("sub", ALPHA, X))),
            X
        ),
        X
    )

    # σ²(x-α)·x — odwrócony gate (x-α zamiast α-x)
    variants["QG_flipped"] = _bin("mul",
        _un("square", _un("sigmoid", _bin("sub", X, ALPHA))),
        X
    )

    return variants


def periodic_variants():
    """Rodzina periodyczna — sin/cos z parametrami"""
    variants = {}

    # max(x, α·sin(βx))
    variants["P_max_sin"] = _bin("max",
        X,
        _bin("mul", ALPHA, _un("sin", _bin("mul", BETA, X)))
    )

    # x + α·sin(βx) — residual + perturbacja
    variants["P_res_sin"] = _bin("add",
        X,
        _bin("mul", ALPHA, _un("sin", _bin("mul", BETA, X)))
    )

    # x·cos(αx) — modulowany cosinus
    variants["P_xcos"] = _bin("mul",
        X,
        _un("cos", _bin("mul", ALPHA, X))
    )

    # relu(x) + α·sin(βx) — ReLU + periodic perturbation
    variants["P_relu_sin"] = _bin("add",
        _un("relu", X),
        _bin("mul", ALPHA, _un("sin", _bin("mul", BETA, X)))
    )

    # max(x, sin(x)) — parametric-free sinusoidal ReLU
    variants["P_max_sinx"] = _bin("max", X, _un("sin", X))

    # x + α·cos(βx) — cosine perturbation
    variants["P_res_cos"] = _bin("add",
        X,
        _bin("mul", ALPHA, _un("cos", _bin("mul", BETA, X)))
    )

    return variants


def new_primitives_variants():
    """Warianty wykorzystujące nowe prymitywy (gaussian, erf, sinc, arctan, softsign)."""
    variants = {}

    # x * erf(α·x) — parametryczny GELU
    variants["N_param_gelu"] = _bin("mul",
        X,
        _un("erf_safe", _bin("mul", ALPHA, X))
    )

    # x * gaussian(α·x) — gaussian gate
    variants["N_gauss_gate"] = _bin("mul",
        X,
        _un("gaussian", _bin("mul", ALPHA, X))
    )

    # max(x, arctan(α·x)) — bounded max
    variants["N_max_arctan"] = _bin("max",
        X,
        _un("arctan", _bin("mul", ALPHA, X))
    )

    # softplus(x) * sigmoid(α-x) — ShiftGate z softplus
    variants["N_shift_softplus"] = _bin("mul",
        _un("softplus", X),
        _un("sigmoid", _bin("sub", ALPHA, X))
    )

    # x * sinc(α·x) — sinc modulation
    variants["N_sinc_mod"] = _bin("mul",
        X,
        _un("sinc_safe", _bin("mul", ALPHA, X))
    )

    # x * softsign(α·x) — softsign gate
    variants["N_softsign_gate"] = _bin("mul",
        X,
        _un("softsign", _bin("mul", ALPHA, X))
    )

    # softplus(x) * erf(α·x) — smooth GELU variant
    variants["N_softplus_erf"] = _bin("mul",
        _un("softplus", X),
        _un("erf_safe", _bin("mul", ALPHA, X))
    )

    # x * gaussian(α-x) — gaussian ShiftGate
    variants["N_gauss_shift"] = _bin("mul",
        X,
        _un("gaussian", _bin("sub", ALPHA, X))
    )

    return variants


def bio_oscillatory_variants():
    """Warianty z nowymi biologicznymi/oscylacyjnymi prymitywami."""
    variants = {}

    # x * gcu(αx) — parametryczny GCU
    variants["B_param_gcu"] = _bin("mul",
        X,
        _un("gcu", _bin("mul", ALPHA, X))
    )

    # hermite(αx) * x — double Hermite gate
    variants["B_hermite_gate"] = _bin("mul",
        _un("hermite", _bin("mul", ALPHA, X)),
        X
    )

    # max(x, gcu(x)) — GCU-ReLU hybrid
    variants["B_max_gcu"] = _bin("max",
        X,
        _un("gcu", X)
    )

    # ncu(x) * sigmoid(αx) — NCU gated
    variants["B_ncu_gated"] = _bin("mul",
        _un("ncu", X),
        _un("sigmoid", _bin("mul", ALPHA, X))
    )

    # logcosh(x) * sigmoid(α-x) — ShiftGate z logcosh
    variants["B_logcosh_shift"] = _bin("mul",
        _un("logcosh", X),
        _un("sigmoid", _bin("sub", ALPHA, X))
    )

    # mexican_hat(αx) * x — wavelet gate
    variants["B_mexhat_gate"] = _bin("mul",
        _un("mexican_hat", _bin("mul", ALPHA, X)),
        X
    )

    # gcu(x) + α·hermite(x) — GCU + Hermite combo
    variants["B_gcu_hermite"] = _bin("add",
        _un("gcu", X),
        _bin("mul", ALPHA, _un("hermite", X))
    )

    # sig_deriv(αx) * x — bell-shaped gate
    variants["B_bell_gate"] = _bin("mul",
        _un("sig_deriv", _bin("mul", ALPHA, X)),
        X
    )

    return variants


# =============================================================================
# Runner
# =============================================================================

def run_targeted_search():
    """Testuje wszystkie warianty na CIFAR-10 screening."""

    all_variants = {}
    all_variants.update(quadgate_variants())
    all_variants.update(periodic_variants())
    all_variants.update(new_primitives_variants())
    all_variants.update(bio_oscillatory_variants())

    logger.info("=" * 60)
    logger.info("  TARGETED SEARCH — %d variants", len(all_variants))
    logger.info("=" * 60)

    results = []

    for name, tree in all_variants.items():
        expr_str = tree.to_string()
        logger.info("\n  Testing: %s", name)
        logger.info("    Expression: %s", expr_str)
        logger.info("    Nodes: %d, Depth: %d, Params: %d",
                    tree.node_count(), tree.depth(),
                    len(tree.get_learnable_params()))

        start = time.time()
        score = safe_evaluate(tree, phase="screening")

        score.expression = expr_str
        score.tree_hash = tree.structural_hash()
        score.tree_depth = tree.depth()
        score.tree_nodes = tree.node_count()
        score.num_params = len(tree.get_learnable_params())

        if not score.training_crashed:
            score.composite_score = compute_composite_score(score)

        elapsed = time.time() - start

        entry = {
            "name": name,
            "expression": expr_str,
            "tree": tree_to_dict(tree),
            "accuracy": score.accuracy_mean,
            "composite_score": score.composite_score,
            "crashed": score.training_crashed,
            "crash_reason": score.crash_reason,
            "nodes": tree.node_count(),
            "params": len(tree.get_learnable_params()),
            "time_sec": round(elapsed, 1),
        }
        results.append(entry)

        if score.training_crashed:
            logger.info("    CRASHED: %s  (%.0fs)", score.crash_reason, elapsed)
        else:
            logger.info("    → %s: %.2f%%  score=%.4f  (%.0fs)",
                        name, score.accuracy_mean, score.composite_score, elapsed)

    # Save results
    save_path = RESULTS_DIR / "targeted_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to %s", save_path)

    # Print ranking
    valid = [r for r in results if not r["crashed"]]
    valid.sort(key=lambda r: r["accuracy"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"  TARGETED SEARCH RESULTS — CIFAR-10 screening (5 epochs, 1 seed)")
    print(f"{'=' * 80}")
    print(f"\n  QuadGate baseline (screening): 76.67%")
    print(f"  Swish baseline (screening):    75.72%")
    print(f"\n  {'#':>3s}  {'Name':20s}  {'Acc':>7s}  {'Nodes':>5s}  {'Params':>6s}  Expression")
    print(f"  {'-'*3}  {'-'*20}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*30}")

    for i, r in enumerate(valid):
        beats = ""
        if r["accuracy"] > 76.67:
            beats = " ★ BEATS QG"
        elif r["accuracy"] > 75.72:
            beats = " ✓ beats Swish"
        print(f"  {i+1:3d}  {r['name']:20s}  {r['accuracy']:6.2f}%  {r['nodes']:5d}  "
              f"{r['params']:6d}  {r['expression'][:40]}{beats}")

    crashed = [r for r in results if r["crashed"]]
    if crashed:
        print(f"\n  CRASHED ({len(crashed)}):")
        for r in crashed:
            print(f"    {r['name']:20s}  {r['crash_reason']}")

    print(f"\n{'=' * 80}")
    return results


if __name__ == "__main__":
    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("neural_genesis.log")],
    )
    run_targeted_search()
