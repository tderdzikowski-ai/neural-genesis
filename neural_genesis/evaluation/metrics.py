"""
Neural Genesis — Metrics & Scoring

Wielowymiarowa ocena funkcji aktywacji.
ActivationScore zbiera wszystkie metryki.
compute_composite_score łączy je w jeden numer do rankingu.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

from config import SCORE_WEIGHTS


@dataclass
class ActivationScore:
    """Pełny profil oceny funkcji aktywacji."""

    # --- Identyfikacja ---
    expression: str = ""              # Czytelna forma wyrażenia
    tree_hash: str = ""               # Hash do deduplikacji
    tree_depth: int = 0               # Głębokość drzewa
    tree_nodes: int = 0               # Liczba węzłów
    num_params: int = 0               # Ile uczonych parametrów

    # --- Główne metryki ---
    accuracy_mean: float = 0.0        # Średnia test accuracy (%)
    accuracy_std: float = 0.0         # Odchylenie (stabilność)
    accuracies: list[float] = field(default_factory=list)  # Per-seed

    # --- Szybkość treningu ---
    epochs_to_90pct: Optional[int] = None  # Ile epok do 90% acc
    final_train_loss: float = 999.0   # Końcowy loss

    # --- Zdrowie gradientów ---
    grad_mean: float = 0.0            # Średnia |gradient|
    grad_std: float = 0.0             # Zmienność gradientów
    grad_vanish_ratio: float = 0.0    # % warstw z znikającym gradientem
    grad_explode_ratio: float = 0.0   # % warstw z eksplodującym gradientem

    # --- Wydajność ---
    forward_time_ms: float = 0.0      # Czas forward pass [ms]
    memory_mb: float = 0.0            # Zużycie pamięci [MB]

    # --- Flagi bezpieczeństwa ---
    produced_nan: bool = False
    produced_inf: bool = False
    training_crashed: bool = False
    crash_reason: str = ""

    # --- Composite score (obliczany) ---
    composite_score: float = -1.0

    def to_dict(self) -> dict:
        return asdict(self)


# Wynik dla kandydatów, którzy nie przeszli walidacji
FAILED_SCORE = ActivationScore(
    accuracy_mean=0.0,
    training_crashed=True,
    crash_reason="failed_validation",
    composite_score=-1.0,
)


def compute_composite_score(s: ActivationScore) -> float:
    """
    Jeden numer do rankingu. Wyższy = lepszy.
    Zwraca -1.0 dla zdyskwalifikowanych kandydatów.
    """
    if s.produced_nan or s.produced_inf or s.training_crashed:
        return -1.0

    w = SCORE_WEIGHTS
    score = 0.0

    # 1. Accuracy — najważniejsze (waga 60%)
    score += w["accuracy"] * (s.accuracy_mean / 100.0)

    # 2. Stabilność — niska wariancja między seedami (waga 15%)
    stability = max(0.0, 1.0 - s.accuracy_std / 2.0)
    score += w["stability"] * stability

    # 3. Szybkość konwergencji (waga 10%)
    if s.epochs_to_90pct is not None and s.epochs_to_90pct > 0:
        convergence = max(0.0, 1.0 - s.epochs_to_90pct / 50.0)
    else:
        convergence = 0.0
    score += w["convergence"] * convergence

    # 4. Prostota wyrażenia — Occam's Razor (waga 10%)
    simplicity = max(0.0, 1.0 - s.tree_nodes / 15.0)
    score += w["simplicity"] * simplicity

    # 5. Wydajność obliczeniowa vs ReLU (waga 5%)
    relu_time_ms = 0.05  # Typowy czas ReLU
    if s.forward_time_ms > 0:
        efficiency = min(1.0, relu_time_ms / s.forward_time_ms)
    else:
        efficiency = 1.0
    score += w["efficiency"] * efficiency

    return round(score, 6)
