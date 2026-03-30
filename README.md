# Neural Genesis: Automated Discovery of Activation Functions

> *Discovered on Apple MacBook M2*

A framework for automated discovery of novel activation functions using evolutionary search over a compositional expression space. Runs entirely on consumer hardware (Apple Silicon / single GPU).

**Key discovery: SoftplusErf** — a new activation function that consistently outperforms GELU and Mish across architectures and datasets.

## SoftplusErf

```
SoftplusErf(x) = softplus(x) * erf(alpha * x)
```

where `alpha` is a learnable parameter (1 per layer), initialized to 1.0.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftplusErf(nn.Module):
    def __init__(self, alpha_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        return F.softplus(x) * torch.erf(self.alpha * x)
```

Or simply:

```python
from softpluserf import SoftplusErf

model = nn.Sequential(
    nn.Linear(784, 256),
    SoftplusErf(),
    nn.Linear(256, 10),
)
```

### Results

All experiments: 50 epochs, Adam optimizer, lr=0.001, weight_decay=1e-4.

#### CIFAR-100 — SmallCNN (3 seeds)

| Activation | Accuracy | Std |
|---|---|---|
| **SoftplusErf** | **61.78%** | **0.20** |
| GELU | 60.68% | 0.20 |
| Mish | 59.75% | 0.19 |
| ReLU | 55.10% | 0.19 |

#### CIFAR-10 — ResNet-18 (3 seeds)

| Activation | Accuracy | Std |
|---|---|---|
| **SoftplusErf** | **91.43%** | **0.09** |
| GELU | 91.00% | 0.03 |
| Mish | 90.91% | 0.07 |
| ReLU | 90.86% | 0.23 |

#### CIFAR-10 — SmallCNN (10 seeds)

| Activation | Accuracy | Std |
|---|---|---|
| **SoftplusErf** | **88.09%** | **0.21** |
| GELU | 87.34% | 0.24 |
| Mish | 87.25% | 0.20 |
| ReLU | 85.42% | 0.33 |

#### Gradient Flow

| Activation | Grad Mean | Vanishing (%) |
|---|---|---|
| **SoftplusErf** | 0.5013 | **0.0** |
| Mish | 0.5033 | 0.0 |
| GELU | 0.5000 | 11.9 |
| ReLU | 0.5000 | 50.0 |

## How It Works

Neural Genesis explores a compositional search space of mathematical expressions to discover activation functions:

1. **Expression space**: Trees built from 27 unary operations (relu, sigmoid, tanh, sin, erf, softplus, gaussian, gcu, hermite, ...) and 7 binary operations (add, mul, max, ...), with up to 3 learnable parameters per expression.

2. **Screening**: Each candidate is evaluated on CIFAR-10 for 5 epochs (~75s on MPS). Candidates scoring above threshold proceed to full evaluation.

3. **Full evaluation**: Top candidates are tested across multiple architectures (SmallCNN, ResNet-18), datasets (CIFAR-10, CIFAR-100), and seeds (3-10) for statistical robustness.

4. **Composite scoring**: Candidates are ranked by a weighted combination of accuracy (70%), stability (12%), convergence speed (8%), simplicity (5%), and efficiency (5%).

The search evaluated **1,194 candidate activations** to discover SoftplusErf.

## Installation

```bash
git clone https://github.com/tomekd/neural-genesis.git
cd neural-genesis
pip install torch torchvision numpy
```

## Usage

### Use SoftplusErf in your model

```python
from softpluserf import SoftplusErf

# Drop-in replacement for nn.ReLU(), nn.GELU(), nn.Mish()
model = YourModel(activation=SoftplusErf())
```

### Run the search pipeline

```bash
# Quick test (verify setup)
python neural_genesis/run_test.py

# Measure baselines
python neural_genesis/run_baselines.py

# Random search (500 candidates)
python neural_genesis/run_stage0.py --count 500 --seed 42
```

### Reproduce SoftplusErf evaluation

```bash
python neural_genesis/run_softplus_erf_eval.py
```

## Project Structure

```
neural-genesis/
  softpluserf.py              # Standalone activation (drop-in)
  neural_genesis/
    config.py                  # Search space & training config
    expression/                # Expression tree: nodes, ops, codegen
    evaluation/                # Training, metrics, networks (SmallCNN, ResNet-18)
    search/                    # Random search & targeted search
    analysis/                  # Leaderboard & visualization
    run_*.py                   # Entry points
  results/                     # Evaluation results (JSON)
  paper/                       # Paper PDF
```

## Citation

If you use SoftplusErf or Neural Genesis in your research:

```bibtex
@software{derdzikowski2026neuralgenesis,
  author = {Derdzikowski, Tomasz},
  title = {Neural Genesis: Automated Discovery of Activation Functions},
  year = {2026},
  url = {https://github.com/tomekd/neural-genesis}
}
```

## License

MIT
