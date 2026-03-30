#!/usr/bin/env python3
"""
Neural Genesis — Full Evaluation: SoftplusErf
    softplus(x) * erf(alpha * x)

5 tests:
  1. CIFAR-100 SmallCNN, 50ep, 3 seeds
  2. ResNet-18 CIFAR-10, 50ep, 3 seeds
  3. SmallCNN CIFAR-10, 50ep, 10 seeds
  4. Gradient flow analysis
  5. Convergence speed (acc @ 5, 10, 20, 50 epochs)

Comparisons: ShiftGate, ReLU, GELU, Mish
"""

import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluation.datasets import get_data_loaders
from config import PROJECT_ROOT, SAFETY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("neural_genesis.log"),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "results" / "softplus_erf_full_eval.json"
SEEDS_3 = [42, 123, 456]
SEEDS_10 = [42, 123, 456, 789, 1000, 2024, 3141, 4242, 5555, 6789]


# =============================================================================
# Device — prefer MPS, fallback CPU
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


# =============================================================================
# Activation functions
# =============================================================================

class SoftplusErf(nn.Module):
    """softplus(x) * erf(alpha * x)"""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return F.softplus(x) * torch.erf(self.alpha * x)


class ShiftGate(nn.Module):
    """sigma(alpha - x) * x"""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.alpha - x)


ACTIVATIONS = {
    "SoftplusErf": (lambda: SoftplusErf(), True),   # (factory, has_params)
    "ShiftGate":   (lambda: ShiftGate(), True),
    "ReLU":        (lambda: nn.ReLU(), False),
    "GELU":        (lambda: nn.GELU(), False),
    "Mish":        (lambda: nn.Mish(), False),
}


# =============================================================================
# SmallCNN (existing architecture)
# =============================================================================

class SmallCNN(nn.Module):
    def __init__(self, activation_factory, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        ch = [32, 32, 64, 64]
        fc_hidden = 256

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], 3, padding=1),
            nn.BatchNorm2d(ch[0]),
            activation_factory(),
            nn.Conv2d(ch[0], ch[1], 3, padding=1),
            nn.BatchNorm2d(ch[1]),
            activation_factory(),
            nn.MaxPool2d(2),

            nn.Conv2d(ch[1], ch[2], 3, padding=1),
            nn.BatchNorm2d(ch[2]),
            activation_factory(),
            nn.Conv2d(ch[2], ch[3], 3, padding=1),
            nn.BatchNorm2d(ch[3]),
            activation_factory(),
            nn.MaxPool2d(2),
        )

        feature_size = ch[3] * (img_size // 4) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, fc_hidden),
            activation_factory(),
            nn.Dropout(0.25),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =============================================================================
# ResNet-18 (custom, with swappable activation)
# =============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation_factory=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = activation_factory()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = activation_factory()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, activation_factory, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act0 = activation_factory()

        self.layer1 = self._make_layer(64, 2, stride=1, activation_factory=activation_factory)
        self.layer2 = self._make_layer(128, 2, stride=2, activation_factory=activation_factory)
        self.layer3 = self._make_layer(256, 2, stride=2, activation_factory=activation_factory)
        self.layer4 = self._make_layer(512, 2, stride=2, activation_factory=activation_factory)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride, activation_factory):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, activation_factory))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act0(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# =============================================================================
# Training helpers
# =============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            return None, None  # Signal crash

        loss = criterion(outputs, targets)
        if math.isnan(loss.item()) or loss.item() > 100.0:
            return None, None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total, running_loss / len(train_loader)


def evaluate_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def collect_gradient_stats(model):
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.data.abs().mean().item())
    if not grad_norms:
        return {"mean": 0, "std": 0, "vanish_ratio": 1.0, "explode_ratio": 0.0}
    arr = np.array(grad_norms)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "vanish_ratio": float((arr < 1e-7).mean()),
        "explode_ratio": float((arr > 100).mean()),
    }


def measure_forward_time(activation_factory, device, num_runs=200):
    act = activation_factory().to(device)
    x = torch.randn(1024, device=device)
    for _ in range(20):
        act(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        act(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()
    return (end - start) / num_runs * 1000  # ms


# =============================================================================
# Full training run with per-epoch tracking
# =============================================================================

def full_train(
    model_class,
    activation_factory,
    dataset_name,
    epochs,
    seed,
    device,
    lr=0.001,
    weight_decay=1e-4,
    convergence_checkpoints=None,
):
    """
    Returns dict with:
      - best_acc, final_acc, crashed
      - epoch_accs: list of (epoch, test_acc)
      - convergence: {5: acc, 10: acc, 20: acc, 50: acc}
      - grad_stats: gradient flow analysis
    """
    if convergence_checkpoints is None:
        convergence_checkpoints = [5, 10, 20, 50]

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    train_loader, test_loader = get_data_loaders(dataset_name, batch_size=128)

    # Dataset config
    ds_cfg = {
        "CIFAR10":  {"in_channels": 3, "num_classes": 10, "img_size": 32},
        "CIFAR100": {"in_channels": 3, "num_classes": 100, "img_size": 32},
    }
    cfg = ds_cfg[dataset_name]

    if model_class == SmallCNN:
        model = SmallCNN(activation_factory, cfg["in_channels"], cfg["num_classes"], cfg["img_size"]).to(device)
    elif model_class == ResNet18:
        model = ResNet18(activation_factory, cfg["num_classes"]).to(device)
    else:
        raise ValueError(f"Unknown model: {model_class}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    epoch_accs = []
    convergence = {}

    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if train_acc is None:
            logger.warning(f"  Crashed at epoch {epoch}")
            return {"crashed": True, "crash_epoch": epoch}

        test_acc = evaluate_accuracy(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc

        epoch_accs.append((epoch, test_acc))

        if epoch in convergence_checkpoints:
            convergence[epoch] = best_acc
            logger.info(f"    ep {epoch}: test={test_acc:.2f}% best={best_acc:.2f}% loss={train_loss:.3f}")

        # Log every 10 epochs
        if epoch % 10 == 0 and epoch not in convergence_checkpoints:
            logger.info(f"    ep {epoch}: test={test_acc:.2f}% best={best_acc:.2f}% loss={train_loss:.3f}")

    # Gradient stats from final state
    grad_stats = collect_gradient_stats(model)

    return {
        "crashed": False,
        "best_acc": best_acc,
        "final_acc": test_acc,
        "convergence": convergence,
        "epoch_accs": epoch_accs,
        "grad_stats": grad_stats,
    }


# =============================================================================
# Test runners
# =============================================================================

def run_test(test_name, model_class, dataset_name, epochs, seeds, activations, device):
    """Run a test across all activations and seeds."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST: {test_name}")
    logger.info(f"  Model: {model_class.__name__}, Dataset: {dataset_name}")
    logger.info(f"  Epochs: {epochs}, Seeds: {len(seeds)}, Device: {device}")
    logger.info(f"{'='*60}")

    results = {}

    for act_name, (act_factory, _) in activations.items():
        logger.info(f"\n  Testing: {act_name}")
        seed_results = []
        crashed = False

        for seed in seeds:
            logger.info(f"    Seed {seed}...")
            t0 = time.time()

            try:
                r = full_train(
                    model_class=model_class,
                    activation_factory=act_factory,
                    dataset_name=dataset_name,
                    epochs=epochs,
                    seed=seed,
                    device=device,
                )
            except RuntimeError as e:
                err_str = str(e)
                logger.error(f"    RuntimeError: {err_str[:200]}")

                # MPS crash — fallback to CPU
                if "mps" in err_str.lower() or "MPS" in err_str:
                    logger.info(f"    MPS crashed — retrying on CPU...")
                    try:
                        r = full_train(
                            model_class=model_class,
                            activation_factory=act_factory,
                            dataset_name=dataset_name,
                            epochs=epochs,
                            seed=seed,
                            device=torch.device("cpu"),
                        )
                    except Exception as e2:
                        logger.error(f"    CPU also failed: {e2}")
                        r = {"crashed": True, "crash_reason": str(e2)}
                else:
                    r = {"crashed": True, "crash_reason": err_str[:200]}
            except Exception as e:
                logger.error(f"    Exception: {e}")
                r = {"crashed": True, "crash_reason": str(e)[:200]}

            elapsed = time.time() - t0

            if r.get("crashed"):
                logger.warning(f"    {act_name} seed {seed}: CRASHED ({elapsed:.0f}s)")
                crashed = True
                break
            else:
                logger.info(f"    {act_name} seed {seed}: {r['best_acc']:.2f}% ({elapsed:.0f}s)")
                seed_results.append(r)

        if crashed:
            results[act_name] = {"crashed": True, "crash_reason": r.get("crash_reason", "unknown")}
        else:
            accs = [r["best_acc"] for r in seed_results]
            # Merge convergence across seeds
            conv_keys = seed_results[0].get("convergence", {}).keys()
            avg_convergence = {}
            for k in conv_keys:
                vals = [r["convergence"][k] for r in seed_results if k in r.get("convergence", {})]
                avg_convergence[str(k)] = round(float(np.mean(vals)), 2)

            # Merge gradient stats
            avg_grad = {}
            for gk in ["mean", "std", "vanish_ratio", "explode_ratio"]:
                vals = [r["grad_stats"][gk] for r in seed_results]
                avg_grad[gk] = round(float(np.mean(vals)), 6)

            results[act_name] = {
                "accs": [round(a, 2) for a in accs],
                "mean": round(float(np.mean(accs)), 2),
                "std": round(float(np.std(accs)), 2),
                "convergence": avg_convergence,
                "grad_stats": avg_grad,
                "crashed": False,
            }
            logger.info(f"  -> {act_name}: {results[act_name]['mean']:.2f}% +/- {results[act_name]['std']:.2f}")

    return results


def run_gradient_analysis(activations, device):
    """Detailed gradient flow analysis."""
    logger.info(f"\n{'='*60}")
    logger.info("TEST 4: Gradient Flow Analysis")
    logger.info(f"{'='*60}")

    results = {}
    for act_name, (act_factory, _) in activations.items():
        act = act_factory().to(device)
        x = torch.linspace(-5, 5, 1000, device=device, requires_grad=True)

        try:
            y = act(x)
            grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]

            # Second derivative (curvature)
            grad2 = torch.autograd.grad(grad.sum(), x)[0]

            results[act_name] = {
                "output_mean": round(float(y.mean().item()), 4),
                "output_std": round(float(y.std().item()), 4),
                "output_range": [round(float(y.min().item()), 4), round(float(y.max().item()), 4)],
                "grad_mean": round(float(grad.mean().item()), 4),
                "grad_std": round(float(grad.std().item()), 4),
                "grad_range": [round(float(grad.min().item()), 4), round(float(grad.max().item()), 4)],
                "grad_near_zero_pct": round(float((grad.abs() < 1e-3).float().mean().item()) * 100, 1),
                "grad2_mean": round(float(grad2.mean().item()), 4),
                "grad2_std": round(float(grad2.std().item()), 4),
                # Regions
                "positive_region_pct": round(float((y > 0).float().mean().item()) * 100, 1),
                "negative_region_pct": round(float((y < 0).float().mean().item()) * 100, 1),
            }
            logger.info(f"  {act_name}: grad_mean={results[act_name]['grad_mean']:.4f} "
                        f"grad_std={results[act_name]['grad_std']:.4f} "
                        f"zero_pct={results[act_name]['grad_near_zero_pct']:.1f}%")
        except Exception as e:
            logger.error(f"  {act_name}: gradient analysis failed: {e}")
            results[act_name] = {"error": str(e)}

    return results


def run_forward_time(activations, device):
    """Measure forward pass time for each activation."""
    logger.info(f"\n{'='*60}")
    logger.info("Forward Time Benchmark")
    logger.info(f"{'='*60}")

    results = {}
    for act_name, (act_factory, _) in activations.items():
        t = measure_forward_time(act_factory, device)
        results[act_name] = round(t, 4)
        logger.info(f"  {act_name}: {t:.4f} ms")
    return results


# =============================================================================
# Main
# =============================================================================

def save_results(all_results):
    """Save results incrementally."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {RESULTS_PATH}")


def main():
    device = DEVICE
    logger.info(f"Device: {device}")
    logger.info(f"SoftplusErf: softplus(x) * erf(alpha * x)")
    logger.info(f"Comparisons: ShiftGate, ReLU, GELU, Mish")

    all_results = {
        "candidate": "SoftplusErf",
        "expression": "softplus(x) * erf(alpha * x)",
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {},
    }

    # =========================================================================
    # Test 1: CIFAR-100, SmallCNN, 50ep, 3 seeds
    # =========================================================================
    try:
        r1 = run_test(
            "Test 1: CIFAR-100 SmallCNN",
            SmallCNN, "CIFAR100", 50, SEEDS_3, ACTIVATIONS, device,
        )
        all_results["tests"]["cifar100_smallcnn"] = r1
        save_results(all_results)
    except Exception as e:
        logger.error(f"Test 1 failed: {e}\n{traceback.format_exc()}")
        all_results["tests"]["cifar100_smallcnn"] = {"error": str(e)}
        save_results(all_results)

    # =========================================================================
    # Test 2: ResNet-18 CIFAR-10, 50ep, 3 seeds
    # =========================================================================
    try:
        r2 = run_test(
            "Test 2: ResNet-18 CIFAR-10",
            ResNet18, "CIFAR10", 50, SEEDS_3, ACTIVATIONS, device,
        )
        all_results["tests"]["resnet18_cifar10"] = r2
        save_results(all_results)
    except Exception as e:
        logger.error(f"Test 2 failed: {e}\n{traceback.format_exc()}")
        all_results["tests"]["resnet18_cifar10"] = {"error": str(e)}
        save_results(all_results)

    # =========================================================================
    # Test 3: SmallCNN CIFAR-10, 50ep, 10 seeds
    # =========================================================================
    try:
        r3 = run_test(
            "Test 3: SmallCNN CIFAR-10 (10 seeds)",
            SmallCNN, "CIFAR10", 50, SEEDS_10, ACTIVATIONS, device,
        )
        all_results["tests"]["smallcnn_cifar10_10seeds"] = r3
        save_results(all_results)
    except Exception as e:
        logger.error(f"Test 3 failed: {e}\n{traceback.format_exc()}")
        all_results["tests"]["smallcnn_cifar10_10seeds"] = {"error": str(e)}
        save_results(all_results)

    # =========================================================================
    # Test 4: Gradient flow analysis
    # =========================================================================
    try:
        r4 = run_gradient_analysis(ACTIVATIONS, device)
        all_results["tests"]["gradient_analysis"] = r4
        save_results(all_results)
    except Exception as e:
        logger.error(f"Test 4 failed: {e}\n{traceback.format_exc()}")
        all_results["tests"]["gradient_analysis"] = {"error": str(e)}
        save_results(all_results)

    # =========================================================================
    # Test 5: Convergence speed (already captured in tests 1-3)
    # Compile convergence comparison
    # =========================================================================
    try:
        convergence_summary = {}
        for test_key in ["cifar100_smallcnn", "resnet18_cifar10", "smallcnn_cifar10_10seeds"]:
            test_data = all_results["tests"].get(test_key, {})
            conv_for_test = {}
            for act_name in ACTIVATIONS:
                act_data = test_data.get(act_name, {})
                if not act_data.get("crashed") and "convergence" in act_data:
                    conv_for_test[act_name] = act_data["convergence"]
            if conv_for_test:
                convergence_summary[test_key] = conv_for_test

        all_results["tests"]["convergence_summary"] = convergence_summary
    except Exception as e:
        logger.error(f"Convergence summary failed: {e}")

    # Forward time benchmark
    try:
        ft = run_forward_time(ACTIVATIONS, device)
        all_results["forward_time_ms"] = ft
        save_results(all_results)
    except Exception as e:
        logger.error(f"Forward time failed: {e}")

    # =========================================================================
    # Final summary
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")

    for test_key, test_data in all_results["tests"].items():
        if test_key == "convergence_summary" or test_key == "gradient_analysis":
            continue
        if isinstance(test_data, dict) and "error" not in test_data:
            logger.info(f"\n  {test_key}:")
            for act_name in ACTIVATIONS:
                ad = test_data.get(act_name, {})
                if ad.get("crashed"):
                    logger.info(f"    {act_name}: CRASHED")
                elif "mean" in ad:
                    logger.info(f"    {act_name}: {ad['mean']:.2f}% +/- {ad['std']:.2f}")

    save_results(all_results)
    logger.info(f"\nAll results saved to {RESULTS_PATH}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
