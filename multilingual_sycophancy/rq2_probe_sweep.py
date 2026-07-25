"""
RQ2 Probe Layer Sweep
=====================
Sweeps all layers for a given (activation_set, language) pair.
Trains both MMProbe and LRProbe at each layer, evaluates on test + natural sets.
Saves results to JSON for later aggregation.

Usage:
    python rq2_probe_sweep.py --act_set avg --lang english
    python rq2_probe_sweep.py --act_set avg_final --lang german
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch as t
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor


# ── Paths ────────────────────────────────────────────────────────────────────
# Adjust BASE_DIR to your working directory on the cluster
BASE_DIR = Path(os.environ.get(
    "RQ2_BASE_DIR",
    "/NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy"
))
ACTIVATIONS_DIR = BASE_DIR / "activations"
RESULTS_DIR = BASE_DIR / "results" / "probe_sweep"

LANGUAGES = ["english", "german", "indonesian", "thai", "russian", "arabic", "italian", "spanish"]
N_LAYERS = 41
LAYER_START = 5   # skip embedding + very early layers
LAYER_END = 35    # skip very late layers (inclusive)


# ── Probe Classes ────────────────────────────────────────────────────────────

class MMProbe(t.nn.Module):
    """Mass-Mean (Difference-of-Means) probe — plain dot-product classifier."""
    def __init__(self, direction):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)

    def forward(self, x):
        return t.sigmoid(x @ self.direction)

    def pred(self, x):
        return self(x).round()

    @staticmethod
    def from_data(acts, labels, device="cpu"):
        acts, labels = acts.to(device).float(), labels.to(device)
        pos_mean = acts[labels == 1].mean(0)
        neg_mean = acts[labels == 0].mean(0)
        direction = pos_mean - neg_mean
        return MMProbe(direction).to(device)


class LRProbe(t.nn.Module):
    """Logistic Regression probe (sklearn backend, torch wrapper)."""
    def __init__(self, d_in, scaler_mean=None, scaler_scale=None):
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 1, bias=False), t.nn.Sigmoid())
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _normalize(self, x):
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x):
        return self.net(self._normalize(x)).squeeze(-1)

    def pred(self, x):
        return self(x).round()

    @staticmethod
    def from_data(acts, labels, C=0.1, device="cpu"):
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr_model = LogisticRegression(C=C, random_state=42, fit_intercept=False, max_iter=1000)
        lr_model.fit(X_scaled, y)

        scaler_mean = t.tensor(scaler.mean_, dtype=t.float32)
        scaler_scale = t.tensor(scaler.scale_, dtype=t.float32)
        probe = LRProbe(acts.shape[-1], scaler_mean=scaler_mean, scaler_scale=scaler_scale).to(device)
        probe.net[0].weight.data[0] = t.tensor(lr_model.coef_[0], dtype=t.float32).to(device)
        return probe


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_activations_and_labels(filepath: str, lang: str):
    with h5py.File(filepath, "r") as f:
        activations = f[lang]["activations"][:]  # (N, n_layers, hidden_dim)
        labels = f[lang]["labels"][:]

    activations_by_layer = {
        layer_idx: t.from_numpy(activations[:, layer_idx, :].copy()).float()
        for layer_idx in range(activations.shape[1])
    }
    return activations_by_layer, t.from_numpy(labels.copy()).float()


def get_h5_paths(act_set: str):
    """Return (train_path, test_path, nat_path) for a given activation set name."""
    suffix = f"completions_{act_set}.h5"
    return (
        ACTIVATIONS_DIR / f"train_{suffix}",
        ACTIVATIONS_DIR / f"test_{suffix}",
        ACTIVATIONS_DIR / f"nat_{suffix}",
    )


# ── Sweep Logic ──────────────────────────────────────────────────────────────

def sweep_one_language(act_set: str, lang: str):
    """Run full layer sweep for one (act_set, lang) pair. Returns list of dicts."""
    train_path, test_path, nat_path = get_h5_paths(act_set)

    print(f"Loading {lang} from {act_set}...")
    train_acts, train_labels = load_activations_and_labels(str(train_path), lang)
    test_acts, test_labels = load_activations_and_labels(str(test_path), lang)
    nat_acts, nat_labels = load_activations_and_labels(str(nat_path), lang)

    results = []

    for layer in range(LAYER_START, LAYER_END + 1):
        row = {
            "act_set": act_set,
            "lang": lang,
            "layer": layer,
        }

        for probe_name, probe_cls in [("mm", MMProbe), ("lr", LRProbe)]:
            try:
                probe = probe_cls.from_data(train_acts[layer], train_labels)

                # Train acc
                train_preds = probe.pred(train_acts[layer])
                row[f"{probe_name}_train_acc"] = (train_preds == train_labels).float().mean().item()

                # Test acc
                test_preds = probe.pred(test_acts[layer])
                row[f"{probe_name}_test_acc"] = (test_preds == test_labels).float().mean().item()

                # Natural ROC-AUC (the metric that actually matters for generalization)
                nat_probs = probe(nat_acts[layer])  # continuous scores for AUC
                nat_preds = probe.pred(nat_acts[layer])

                nat_labels_np = nat_labels.detach().numpy()
                nat_probs_np = nat_probs.detach().numpy()
                nat_preds_np = nat_preds.detach().numpy()

                # ROC-AUC on continuous scores
                try:
                    row[f"{probe_name}_nat_roc_auc"] = roc_auc_score(nat_labels_np, nat_probs_np)
                except ValueError:
                    row[f"{probe_name}_nat_roc_auc"] = 0.5  # only one class present

                # Balanced accuracy on hard predictions
                n_pos = (nat_labels_np == 1).sum()
                n_neg = (nat_labels_np == 0).sum()
                if n_pos > 0 and n_neg > 0:
                    recall_pos = nat_preds_np[nat_labels_np == 1].mean()
                    recall_neg = (1 - nat_preds_np[nat_labels_np == 0]).mean()
                    row[f"{probe_name}_nat_bal_acc"] = float((recall_pos + recall_neg) / 2)
                else:
                    row[f"{probe_name}_nat_bal_acc"] = 0.5

            except Exception as e:
                print(f"  Warning: {probe_name} failed at layer {layer}: {e}")
                row[f"{probe_name}_train_acc"] = None
                row[f"{probe_name}_test_acc"] = None
                row[f"{probe_name}_nat_roc_auc"] = None
                row[f"{probe_name}_nat_bal_acc"] = None

        results.append(row)
        print(f"  Layer {layer:2d} | MM nat_auc={row.get('mm_nat_roc_auc', 'N/A'):.3f}  LR nat_auc={row.get('lr_nat_roc_auc', 'N/A'):.3f}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RQ2 Probe Layer Sweep")
    parser.add_argument("--act_set", required=True, choices=["avg", "avg_final"],
                        help="Activation set: 'avg' or 'avg_final'")
    parser.add_argument("--lang", required=True, choices=LANGUAGES,
                        help="Language to sweep")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = sweep_one_language(args.act_set, args.lang)

    outfile = RESULTS_DIR / f"sweep_{args.act_set}_{args.lang}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} rows to {outfile}")


if __name__ == "__main__":
    main()