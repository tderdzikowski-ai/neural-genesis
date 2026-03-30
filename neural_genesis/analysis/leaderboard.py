"""
Neural Genesis — Leaderboard

Przechowuje ranking odkrytych funkcji aktywacji.
Zapisuje do JSON po każdej aktualizacji.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path

from evaluation.metrics import ActivationScore
from config import LEADERBOARD_PATH

logger = logging.getLogger(__name__)


class Leaderboard:
    def __init__(self, save_path: Path = LEADERBOARD_PATH):
        self.save_path = save_path
        self.entries: list[ActivationScore] = []
        self.baselines: dict[str, ActivationScore] = {}
        self._seen_hashes: set[str] = set()

        # Załaduj istniejący leaderboard jeśli jest
        if self.save_path.exists():
            self.load()

    def add(self, score: ActivationScore) -> bool:
        """
        Dodaj kandydata. Zwraca True jeśli dodany (nie duplikat).
        """
        if score.tree_hash in self._seen_hashes:
            return False
        if score.composite_score < 0:
            return False

        self._seen_hashes.add(score.tree_hash)
        self.entries.append(score)
        self.entries.sort(key=lambda s: s.composite_score, reverse=True)
        self.save()
        return True

    def add_baseline(self, name: str, score: ActivationScore):
        """Dodaj wynik baseline'owej aktywacji."""
        score.expression = name
        self.baselines[name] = score
        self.save()

    def get_top(self, k: int = 20) -> list[ActivationScore]:
        """Zwróć top-k kandydatów."""
        return self.entries[:k]

    def report(self, top_k: int = 20):
        """Wydrukuj czytelny raport."""
        print(f"\n{'=' * 90}")
        print(f"  NEURAL GENESIS LEADERBOARD — Top {top_k}")
        print(f"{'=' * 90}")

        # Baselines
        if self.baselines:
            print(f"\n  --- BASELINES ---")
            sorted_bl = sorted(
                self.baselines.items(),
                key=lambda x: x[1].accuracy_mean,
                reverse=True,
            )
            for name, s in sorted_bl:
                print(f"    {name:20s}  acc={s.accuracy_mean:6.2f}%  "
                      f"score={s.composite_score:.4f}")

        # Discovered
        if self.entries:
            best_baseline_acc = max(
                (s.accuracy_mean for s in self.baselines.values()),
                default=0.0,
            )
            print(f"\n  --- DISCOVERED (total: {len(self.entries)}) ---")
            for i, entry in enumerate(self.entries[:top_k]):
                beats = ""
                if entry.accuracy_mean > best_baseline_acc:
                    beats = " ★ BEATS ALL BASELINES"
                elif self.baselines.get("GELU") and \
                     entry.accuracy_mean > self.baselines["GELU"].accuracy_mean:
                    beats = " ✓ beats GELU"

                print(f"    #{i + 1:3d}  {entry.expression:45s}  "
                      f"acc={entry.accuracy_mean:6.2f}% ± {entry.accuracy_std:4.2f}  "
                      f"nodes={entry.tree_nodes:2d}  "
                      f"score={entry.composite_score:.4f}{beats}")
        else:
            print("\n  No discovered activations yet.")

        print(f"\n{'=' * 90}\n")

    def save(self):
        """Zapisz do JSON."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "baselines": {
                name: s.to_dict() for name, s in self.baselines.items()
            },
            "discovered": [s.to_dict() for s in self.entries],
        }
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self):
        """Załaduj z JSON."""
        try:
            with open(self.save_path) as f:
                data = json.load(f)

            for name, d in data.get("baselines", {}).items():
                s = ActivationScore(**{k: v for k, v in d.items()
                                       if k in ActivationScore.__dataclass_fields__})
                self.baselines[name] = s

            for d in data.get("discovered", []):
                s = ActivationScore(**{k: v for k, v in d.items()
                                       if k in ActivationScore.__dataclass_fields__})
                self.entries.append(s)
                if s.tree_hash:
                    self._seen_hashes.add(s.tree_hash)

            self.entries.sort(key=lambda s: s.composite_score, reverse=True)
            logger.info(f"Loaded leaderboard: {len(self.baselines)} baselines, "
                        f"{len(self.entries)} discovered")
        except Exception as e:
            logger.warning(f"Could not load leaderboard: {e}")
