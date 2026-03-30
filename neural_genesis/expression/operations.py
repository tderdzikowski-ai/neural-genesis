"""
Neural Genesis — Safe PyTorch Operations

Każda operacja matematyczna ma "bezpieczną" wersję, która:
  - Nigdy nie produkuje NaN
  - Nigdy nie produkuje Inf
  - Jest różniczkowalna (tam gdzie to możliwe)
  - Ma ograniczony zakres wyjściowy
"""

import torch
from config import EPSILON


# =============================================================================
# Unary operations: f(x) -> y
# =============================================================================

def _identity(x: torch.Tensor) -> torch.Tensor:
    return x

def _negate(x: torch.Tensor) -> torch.Tensor:
    return -x

def _abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)

def _square(x: torch.Tensor) -> torch.Tensor:
    return x ** 2

def _cube(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x ** 3, -100.0, 100.0)

def _sqrt_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.abs(x) + EPSILON)

def _sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)

def _cos(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x)

def _exp_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, -20.0, 20.0))

def _log_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.abs(x) + EPSILON)

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.clamp(x, -20.0, 20.0))

def _tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

def _sign(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x)

def _inv_safe(x: torch.Tensor) -> torch.Tensor:
    # Dodajemy epsilon z zachowaniem znaku, żeby uniknąć dzielenia przez 0
    safe_x = x + EPSILON * torch.sign(x + EPSILON)
    return 1.0 / safe_x

def _gaussian(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-x ** 2)

def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(torch.clamp(x, -20.0, 20.0)))

def _erf_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.erf(x)

def _softsign(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.abs(x))

def _arctan(x: torch.Tensor) -> torch.Tensor:
    return torch.atan(x)

def _sinc_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) / (x + EPSILON)

def _gcu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.cos(x)

def _ncu(x: torch.Tensor) -> torch.Tensor:
    return x - torch.clamp(x ** 3, -100.0, 100.0) / 3.0

def _hermite(x: torch.Tensor) -> torch.Tensor:
    return x * torch.exp(-x ** 2)

def _mexican_hat(x: torch.Tensor) -> torch.Tensor:
    return (1.0 - x ** 2) * torch.exp(-x ** 2 / 2.0)

def _logcosh(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.cosh(torch.clamp(x, -20.0, 20.0)))

def _sig_deriv(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s)


UNARY_FN = {
    "identity":  _identity,
    "negate":    _negate,
    "abs":       _abs,
    "square":    _square,
    "cube":      _cube,
    "sqrt_abs":  _sqrt_abs,
    "sin":       _sin,
    "cos":       _cos,
    "exp_safe":  _exp_safe,
    "log_safe":  _log_safe,
    "sigmoid":   _sigmoid,
    "tanh":      _tanh,
    "relu":      _relu,
    "sign":      _sign,
    "inv_safe":  _inv_safe,
    "gaussian":  _gaussian,
    "softplus":  _softplus,
    "erf_safe":  _erf_safe,
    "softsign":  _softsign,
    "arctan":    _arctan,
    "sinc_safe": _sinc_safe,
    "gcu":       _gcu,
    "ncu":       _ncu,
    "hermite":   _hermite,
    "mexican_hat": _mexican_hat,
    "logcosh":   _logcosh,
    "sig_deriv": _sig_deriv,
}


# =============================================================================
# Binary operations: f(x, y) -> z
# =============================================================================

def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

def _sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y

def _mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y

def _div_safe(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    safe_y = y + EPSILON * torch.sign(y + EPSILON)
    return x / safe_y

def _max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.max(x, y)

def _min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.min(x, y)

def _pow_safe(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # |x|^y z zachowaniem znaku x, y ograniczone do [-5, 5]
    base = torch.abs(x) + EPSILON
    exponent = torch.clamp(y, -5.0, 5.0)
    result = torch.sign(x) * torch.pow(base, exponent)
    return torch.clamp(result, -100.0, 100.0)


BINARY_FN = {
    "add":       _add,
    "sub":       _sub,
    "mul":       _mul,
    "div_safe":  _div_safe,
    "max":       _max,
    "min":       _min,
    "pow_safe":  _pow_safe,
}
