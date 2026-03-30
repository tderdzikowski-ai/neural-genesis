"""
Neural Genesis — Configuration
All hyperparameters in one place. No magic numbers in code.

Changelog:
  v1.2 — Screening na CIFAR-10 zamiast FashionMNIST.
          Kandydaci z FashionMNIST nie generalizowali (93%→81% vs Mish 84%).
          CIFAR-10 screening z 5 epokami daje lepszą predykcję full eval.
  v1.1 — Rebalanced SCORE_WEIGHTS: accuracy 0.60→0.70, simplicity 0.10→0.05
          Po Stage 0 (10 kandydatów) liniowa (1+x) wygrała composite score
          mimo 3% gorszej accuracy. Prostota była zbyt nagradzana.
          Podniesiono min_depth do 2 żeby wyeliminować trywialne funkcje.
"""

import torch
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
SCREENING_DIR = RESULTS_DIR / "screening"
FULL_EVAL_DIR = RESULTS_DIR / "full_eval"
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.json"

# =============================================================================
# Device — auto-detect CUDA (Windows RTX) > MPS (Mac M2) > CPU
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# =============================================================================
# Expression Tree — Search Space
# =============================================================================

EXPRESSION = {
    "max_depth": 4,           # Maksymalna głębokość drzewa
    "min_depth": 2,           # Minimalna głębokość (było 1 → trywialne wyrażenia)
    "max_nodes": 15,          # Maksymalna liczba węzłów
    "max_params": 3,          # Maks. uczonych parametrów (α, β, γ)
    "p_leaf_base": 0.1,       # Bazowe prawdopodobieństwo liścia
    "p_leaf_depth_scale": 0.3,  # Jak szybko rośnie p_leaf z głębokością
    "p_binary": 0.55,         # Prawdopodobieństwo operacji binarnej (vs unarnej)
    "p_input": 0.50,          # Prawdopodobieństwo Input(x) wśród liści
    "p_constant": 0.25,       # Prawdopodobieństwo stałej
    "p_parameter": 0.25,      # Prawdopodobieństwo uczonego parametru
}

EPSILON = 1e-7                # Bezpieczna stała numeryczna

# =============================================================================
# Evaluation Network
# =============================================================================

EVAL_NETWORK = {
    "type": "SmallCNN",       # Typ sieci testowej
    "conv_channels": [32, 32, 64, 64],  # Kanały konwolucyjne
    "fc_hidden": 256,         # Neurony w warstwie ukrytej FC
}

# =============================================================================
# Training — Screening Phase (szybki test)
# =============================================================================

SCREENING = {
    "dataset": "CIFAR10",         # v1.2: było FashionMNIST — nie generalizował na CIFAR
    "epochs": 5,                  # v1.2: było 10 — mniej epok bo CIFAR wolniejszy
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "weight_decay": 0.0,
    "num_seeds": 1,               # 1 seed na screening (szybkość)
    "early_stop_epoch": 2,        # Po tylu epokach sprawdź czy się uczy
    "early_stop_min_acc": 12.0,   # v1.2: było 15 — CIFAR jest trudniejszy
    "max_eval_time_sec": 180,     # v1.2: było 120 — CIFAR potrzebuje więcej
}

# =============================================================================
# Training — Full Evaluation Phase (dokładny test)
# =============================================================================

FULL_EVAL = {
    "dataset": "CIFAR10",
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "num_seeds": 3,           # 3 seedy dla stabilności
    "seeds": [42, 123, 456],
}

# =============================================================================
# Stage 0 — Random Search
# =============================================================================

STAGE0 = {
    "total_candidates": 10_000,
    "top_k_for_full_eval": 100,  # Top-N z screeningu → pełna ewaluacja
    "checkpoint_every": 100,     # Zapisz co N kandydatów
    "log_every": 10,             # Loguj postęp co N
}

# =============================================================================
# Composite Score Weights
#
# v1.1: Rebalanced po Stage 0 test run.
#   Problem: (1+x) z 90.12% wygrywał z sigmoid(x^x...) z 93.03%
#   bo simplicity=0.10 zbyt nagradzało 3-node wyrażenia.
#   Fix: accuracy ↑ 0.70, simplicity ↓ 0.05
# =============================================================================

SCORE_WEIGHTS = {
    "accuracy": 0.70,         # było 0.60 — accuracy jest najważniejsze
    "stability": 0.12,        # było 0.15 — nadal ważne, lekko obniżone
    "convergence": 0.08,      # było 0.10 — szybkość uczenia się
    "simplicity": 0.05,       # było 0.10 — za bardzo nagradzało trywialne f-cje
    "efficiency": 0.05,       # bez zmian — czas forward pass
}

# =============================================================================
# Safety Limits
# =============================================================================

SAFETY = {
    "output_clamp": 100.0,        # Clamp na wyjściu aktywacji
    "max_output_value": 1000.0,   # Sanity check — max |output|
    "min_output_std": 1e-6,       # Min std outputu (wykrywa stałe funkcje)
    "max_gradient": 1000.0,       # Max |gradient| w sanity check
    "max_train_loss": 100.0,      # Przerwij trening jeśli loss > tego
}