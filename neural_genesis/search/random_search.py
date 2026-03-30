"""
Neural Genesis — Stage 0: Random Search

Główna pętla przeszukiwania losowego.
Generuje wyrażenia, ewaluuje, rankinguje.
Checkpointuje co N kandydatów — można przerywać i wznawiać.
"""

from __future__ import annotations
import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from expression.generator import ExpressionGenerator
from expression.serializer import tree_to_dict, tree_to_string
from evaluation.trainer import safe_evaluate
from evaluation.metrics import ActivationScore, compute_composite_score
from analysis.leaderboard import Leaderboard
from config import STAGE0, SCREENING_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

PROMISING_PATH = RESULTS_DIR / "promising.json"
PROMISING_MIN_ACCURACY = 92.0


def _save_promising(tree, score: ActivationScore, search_seed: int):
    """Atomowo dopisz obiecującego kandydata do promising.json."""
    entry = {
        "expression": score.expression,
        "tree": tree_to_dict(tree),
        "accuracy": score.accuracy_mean,
        "composite_score": score.composite_score,
        "seed": search_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    PROMISING_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Atomic read-modify-write: czytaj istniejące → dopisz → zapisz do tmp → rename
    existing = []
    if PROMISING_PATH.exists():
        try:
            with open(PROMISING_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(entry)

    # Atomic write — zapisz do tmp w tym samym katalogu, potem rename
    fd, tmp_path = tempfile.mkstemp(
        dir=PROMISING_PATH.parent, suffix=".tmp", prefix=".promising_"
    )
    try:
        with open(fd, "w") as f:
            json.dump(existing, f, indent=2)
        Path(tmp_path).replace(PROMISING_PATH)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def run_random_search(
    total: int | None = None,
    start_from: int = 0,
    seed: int = 42,
    leaderboard: Leaderboard | None = None,
):
    """
    Uruchom losowe przeszukiwanie.

    Args:
        total: Ile kandydatów wygenerować (None → z configu)
        start_from: Od którego kandydata kontynuować (po przerwie)
        seed: Seed generatora (różne seedy → różne wyrażenia)
        leaderboard: Istniejący leaderboard (do kontynuacji)
    """
    if total is None:
        total = STAGE0["total_candidates"]

    if leaderboard is None:
        leaderboard = Leaderboard()

    SCREENING_DIR.mkdir(parents=True, exist_ok=True)

    gen = ExpressionGenerator(seed=seed)

    # Statystyki
    stats = {
        "total_generated": 0,
        "passed_sanity": 0,
        "failed_sanity": 0,
        "best_accuracy": 0.0,
        "best_expression": "",
        "start_time": time.time(),
    }

    logger.info(f"Starting random search: {total} candidates, seed={seed}, "
                f"start_from={start_from}")

    # Jeśli wznawiamy — przeskocz do odpowiedniego miejsca
    for _ in range(start_from):
        gen.generate()

    pbar = tqdm(range(start_from, start_from + total), desc="Stage 0 Search")

    for i in pbar:
        # 1. Generuj wyrażenie
        tree = gen.generate()
        if tree is None:
            continue

        stats["total_generated"] += 1
        expr_str = tree.to_string()

        # 2. Ewaluuj (bezpiecznie)
        score = safe_evaluate(tree, phase="screening")

        # 3. Uzupełnij metadane
        score.expression = expr_str
        score.tree_hash = tree.structural_hash()
        score.tree_depth = tree.depth()
        score.tree_nodes = tree.node_count()
        score.num_params = len(tree.get_learnable_params())

        if not score.training_crashed:
            score.composite_score = compute_composite_score(score)

        # 4. Dodaj do leaderboardu
        if score.composite_score > 0:
            stats["passed_sanity"] += 1
            leaderboard.add(score)

            # Zapisz obiecujących kandydatów
            if score.accuracy_mean >= PROMISING_MIN_ACCURACY:
                try:
                    _save_promising(tree, score, seed)
                except Exception as e:
                    logger.warning(f"Could not save promising candidate: {e}")

            if score.accuracy_mean > stats["best_accuracy"]:
                stats["best_accuracy"] = score.accuracy_mean
                stats["best_expression"] = expr_str
                logger.info(f"  ★ New best: {expr_str} → "
                            f"{score.accuracy_mean:.2f}%")
        else:
            stats["failed_sanity"] += 1

        # 5. Update progress bar
        elapsed = time.time() - stats["start_time"]
        rate = stats["total_generated"] / max(elapsed, 1)
        pbar.set_postfix({
            "best": f"{stats['best_accuracy']:.1f}%",
            "pass": f"{stats['passed_sanity']}/{stats['total_generated']}",
            "rate": f"{rate:.1f}/s" if rate >= 1 else f"{1/max(rate,0.01):.0f}s/ea",
        })

        # 6. Checkpoint
        if (i + 1) % STAGE0["checkpoint_every"] == 0:
            _save_checkpoint(i + 1, stats, leaderboard)
            logger.info(f"  Checkpoint at {i + 1}: "
                        f"{stats['passed_sanity']}/{stats['total_generated']} passed, "
                        f"best={stats['best_accuracy']:.2f}%")

    # Final report
    _save_checkpoint(start_from + total, stats, leaderboard)

    elapsed = time.time() - stats["start_time"]
    logger.info(f"\nSearch complete!")
    logger.info(f"  Total time: {elapsed / 3600:.1f} hours")
    logger.info(f"  Candidates evaluated: {stats['total_generated']}")
    logger.info(f"  Passed sanity: {stats['passed_sanity']}")
    logger.info(f"  Best accuracy: {stats['best_accuracy']:.2f}%")
    logger.info(f"  Best expression: {stats['best_expression']}")

    leaderboard.report()
    return leaderboard


def _save_checkpoint(candidate_num: int, stats: dict, leaderboard: Leaderboard):
    """Zapisz stan przeszukiwania do pliku."""
    checkpoint = {
        "candidate_num": candidate_num,
        "stats": stats,
    }
    path = SCREENING_DIR / "checkpoint.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)
