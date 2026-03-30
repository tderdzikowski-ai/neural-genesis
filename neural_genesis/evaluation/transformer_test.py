#!/usr/bin/env python3
"""
Neural Genesis — Mini Vision Transformer (ViT) Test

Tests activation functions in Transformer FFN blocks.
Compares ShiftGate (QG_linear) vs GELU vs Mish vs ReLU vs SwiGLU.
"""

import json
import logging
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.datasets import get_data_loaders
from config import DEVICE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("neural_genesis.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Activation functions
# =============================================================================

class ShiftGate(nn.Module):
    """σ(α-x)·x — QG_linear / ShiftGate"""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.alpha - x)


class SwiGLU(nn.Module):
    """SwiGLU-style: split input, apply swish gate.
    Expects input dim to be split in half internally."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, out_features)
        self.w2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.w1(x) * F.silu(self.w2(x))


# =============================================================================
# Mini Vision Transformer
# =============================================================================

class PatchEmbedding(nn.Module):
    """Split image into patches and embed."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, d_model)
        x = self.proj(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        return x


class TransformerFFN(nn.Module):
    """Feedforward block with configurable activation."""
    def __init__(self, d_model, d_ff, activation_factory):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = activation_factory()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerFFN_SwiGLU(nn.Module):
    """SwiGLU-based FFN — different structure (gated)."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_ff)
        self.fc_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.fc_out(self.swiglu(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, activation_factory, use_swiglu=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if use_swiglu:
            self.ffn = TransformerFFN_SwiGLU(d_model, d_ff)
        else:
            self.ffn = TransformerFFN(d_model, d_ff, activation_factory)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Self-attention + residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + residual
        x = self.norm2(x + self.ffn(x))
        return x


class MiniViT(nn.Module):
    """Mini Vision Transformer for CIFAR-10 classification."""
    def __init__(self, activation_factory, num_classes=10, img_size=32,
                 patch_size=4, d_model=256, n_heads=4, n_layers=4,
                 d_ff=512, use_swiglu=False):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, d_model)
        num_patches = (img_size // patch_size) ** 2  # 64

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)
        self.dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, activation_factory, use_swiglu)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # (B, 64, 128)
        # Prepend cls token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 65, 128)
        # Add positional embedding
        x = self.dropout(x + self.pos_embed)
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        # Classification from cls token
        x = self.norm(x[:, 0])
        return self.head(x)


# =============================================================================
# Training
# =============================================================================

def train_vit(activation_name, activation_factory, device, epochs=50,
              seeds=[42, 123, 456], use_swiglu=False):
    """Train ViT and return results."""
    results = {"name": activation_name, "accs": [], "per_seed": {}}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, test_loader = get_data_loaders("CIFAR10", 128)
        model = MiniViT(
            activation_factory, num_classes=10,
            use_swiglu=use_swiglu,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        if seed == seeds[0]:
            logger.info("    Model params: %d (%.1fM)", n_params, n_params / 1e6)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
        # Warmup 5 epochs then cosine decay
        warmup_epochs = 5
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_acc = 0.0
        start = time.time()

        for epoch in range(epochs):
            # Train
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if torch.isnan(outputs).any():
                    logger.warning("    %s seed %d: NaN at epoch %d", activation_name, seed, epoch)
                    results["per_seed"][seed] = {"acc": -1, "crashed": True}
                    break
                loss = criterion(outputs, targets)
                if math.isnan(loss.item()):
                    logger.warning("    %s seed %d: NaN loss at epoch %d", activation_name, seed, epoch)
                    results["per_seed"][seed] = {"acc": -1, "crashed": True}
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += loss.item()
            else:
                # Test
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                test_acc = 100.0 * correct / total
                best_acc = max(best_acc, test_acc)

                scheduler.step()

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info("    %s seed %d ep %d: test=%.2f%% best=%.2f%% loss=%.3f",
                                activation_name, seed, epoch + 1, test_acc, best_acc,
                                running_loss / len(train_loader))
                continue
            break  # NaN escape

        elapsed = time.time() - start

        if seed not in results["per_seed"]:
            results["per_seed"][seed] = {"acc": best_acc, "crashed": False}
            results["accs"].append(best_acc)
            logger.info("    %s seed %d: %.2f%% (%ds)", activation_name, seed, best_acc, elapsed)

    if results["accs"]:
        results["mean"] = float(np.mean(results["accs"]))
        results["std"] = float(np.std(results["accs"]))
    else:
        results["mean"] = 0
        results["std"] = 0
        results["crashed"] = True

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    # Force CPU to avoid MPS segfaults on Transformers
    device = DEVICE
    logger.info("=" * 60)
    logger.info("  TRANSFORMER TEST — Mini ViT on CIFAR-10")
    logger.info("=" * 60)
    logger.info("  Device: %s", device)
    logger.info("  Architecture: 4 layers, 4 heads, d=256, ff=512")
    logger.info("  Training: 50 epochs, 3 seeds, cosine annealing")
    logger.info("=" * 60)

    activations = {
        "ShiftGate": (lambda: ShiftGate(), False),
        "GELU": (lambda: nn.GELU(), False),
        "Mish": (lambda: nn.Mish(), False),
        "ReLU": (lambda: nn.ReLU(), False),
        "SwiGLU": (None, True),  # Special case — different FFN structure
    }

    all_results = {}

    for name, (factory, is_swiglu) in activations.items():
        logger.info("\n" + "-" * 60)
        logger.info("  Testing: %s", name)

        try:
            result = train_vit(name, factory, device, epochs=50,
                               seeds=[42, 123, 456], use_swiglu=is_swiglu)
        except Exception as e:
            logger.error("  %s failed on %s: %s", name, device, e)
            if device.type != "cpu":
                logger.info("  Retrying on CPU...")
                try:
                    result = train_vit(name, factory, torch.device("cpu"), epochs=50,
                                       seeds=[42, 123, 456], use_swiglu=is_swiglu)
                except Exception as e2:
                    logger.error("  %s failed on CPU too: %s", name, e2)
                    result = {"name": name, "mean": 0, "std": 0, "crashed": True}
            else:
                result = {"name": name, "mean": 0, "std": 0, "crashed": True}

        all_results[name] = result

        if not result.get("crashed"):
            logger.info("  → %s: %.2f%% ± %.2f", name, result["mean"], result["std"])

    # Save results
    save_path = "results/transformer_eval.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", save_path)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  MINI ViT CIFAR-10 — ACTIVATION COMPARISON")
    print(f"  (4 layers, 4 heads, d=128, 50 epochs, 3 seeds)")
    print(f"{'=' * 70}")

    valid = [(name, r) for name, r in all_results.items() if not r.get("crashed")]
    valid.sort(key=lambda x: x[1]["mean"], reverse=True)

    for i, (name, r) in enumerate(valid):
        marker = " ★" if i == 0 else ""
        print(f"    {name:12s}  {r['mean']:6.2f}% ± {r['std']:.2f}{marker}")

    crashed = [name for name, r in all_results.items() if r.get("crashed")]
    if crashed:
        print(f"\n    CRASHED: {', '.join(crashed)}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
