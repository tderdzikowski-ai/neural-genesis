#!/usr/bin/env python3
"""
Neural Genesis — Stage 0: Random Search

Entry point. Generuje losowe wyrażenia, testuje jako aktywacje,
rankinguje. Można przerywać i wznawiać.

Usage:
    python run_stage0.py                          # Domyślne 10 000 kandydatów
    python run_stage0.py --count 100              # Szybki test: 100 kandydatów
    python run_stage0.py --count 1000 --seed 123  # 1000 z innym seedem
    python run_stage0.py --start 500 --count 500  # Kontynuuj od #500
"""

import argparse
import logging
import sys

from search.random_search import run_random_search
from analysis.leaderboard import Leaderboard
from config import DEVICE, STAGE0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("neural_genesis.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Neural Genesis — Stage 0: Random Activation Search"
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help=f"Number of candidates to evaluate (default: {STAGE0['total_candidates']})"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Resume from candidate number N (default: 0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for expression generator (default: 42)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  NEURAL GENESIS — Stage 0: Random Search")
    logger.info("=" * 60)
    logger.info(f"  Device:     {DEVICE}")
    logger.info(f"  Candidates: {args.count or STAGE0['total_candidates']}")
    logger.info(f"  Start from: {args.start}")
    logger.info(f"  Seed:       {args.seed}")
    logger.info("=" * 60)

    leaderboard = Leaderboard()

    if not leaderboard.baselines:
        logger.warning(
            "No baselines found! Run 'python run_baselines.py' first "
            "to establish comparison points."
        )

    run_random_search(
        total=args.count,
        start_from=args.start,
        seed=args.seed,
        leaderboard=leaderboard,
    )


if __name__ == "__main__":
    main()
