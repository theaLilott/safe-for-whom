"""
RQ2 Mixed-Language Probe Sweep
===============================
Builds mixed-language datasets by sampling from each language's best activation set,
then sweeps layers 5–35 with both MMProbe and LRProbe.

Run AFTER rq2_aggregate_sweep.py has produced best_config.json (or paste BEST_ACT_SET below).

Usage:
    python rq2_probe_sweep_mixed.py
    python rq2_probe_sweep_mixed.py --config results/probe_sweep/best_config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch as t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(os.environ.get(
    "RQ2_BASE_DIR",
    "/NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy"
))
ACTIVATIONS_DIR = BASE_DIR / "activations"
RESULTS_DIR = BASE_DIR / "results" / "probe_sweep"

LANGUAGES = ["english", "german", "indonesian", "thai", "russian", "arabic", "italian", "spanish"]
LAYER_START = 5
LAYER_END = 35

# ── Paste here if you don't want to use --config ─────────────────────────────
BEST_ACT_SET  = {
    "arabic": "avg_final",
    "english": "avg",
    "german": "avg",
    "indonesian": "avg_final",
    "italian": "avg",
    "russian": "avg",
    "spanish": "avg",
    "thai": "avg",
}

# ── Probe Classes (same as rq2_probe_sweep.py) ──────────────────────────────

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
        activations = f[lang]["activations"][:]
        labels = f[lang]["labels"][:]
    activations_by_layer = {
        layer_idx: t.from_numpy(activations[:, layer_idx, :].copy()).float()
        for layer_idx in range(activations.shape[1])
    }
    return activations_by_layer, t.from_numpy(labels.copy()).float()


def get_h5_paths(act_set: str):
    suffix = f"completions_{act_set}.h5"
    return (
        ACTIVATIONS_DIR / f"train_{suffix}",
        ACTIVATIONS_DIR / f"test_{suffix}",
        ACTIVATIONS_DIR / f"nat_{suffix}",
    )


# ── Mixed Dataset Construction ───────────────────────────────────────────────

def make_mixed_dataset(acts_dict, labels_dict, n_total, languages, pair_based=True, seed=42):
    """
    Sample from each language and combine into a single mixed dataset.
    
    Args:
        acts_dict:   {lang: {layer: tensor}}
        labels_dict: {lang: tensor}
        n_total:     total number of samples in the mixed set
        languages:   list of languages to mix
        pair_based:  if True, sample contrastive pairs (indices 2k, 2k+1)
        seed:        random seed
    """
    rng = np.random.default_rng(seed)
    n_langs = len(languages)

    if pair_based:
        pairs_total = n_total // 2
        base_pairs = pairs_total // n_langs
        remainder = pairs_total % n_langs
        extras = rng.choice(n_langs, size=remainder, replace=False)
        pairs_per_lang = [base_pairs + (1 if i in extras else 0) for i in range(n_langs)]
    else:
        base = n_total // n_langs
        remainder = n_total % n_langs
        extras = rng.choice(n_langs, size=remainder, replace=False)
        per_lang = [base + (1 if i in extras else 0) for i in range(n_langs)]

    layers = list(acts_dict[languages[0]].keys())

    all_acts = {layer: [] for layer in layers}
    all_labels = []

    for i, lang in enumerate(languages):
        labels = labels_dict[lang]
        n_samples = len(labels)

        if pair_based:
            n_pairs = n_samples // 2
            selected_pairs = rng.choice(n_pairs, size=pairs_per_lang[i], replace=False)
            selected_idx = np.concatenate([[2*j, 2*j+1] for j in selected_pairs])
        else:
            selected_idx = rng.choice(n_samples, size=per_lang[i], replace=False)

        for layer in layers:
            all_acts[layer].append(acts_dict[lang][layer][selected_idx])
        all_labels.append(labels[selected_idx])

    mixed_labels = t.cat(all_labels, dim=0)
    shuffle_idx = t.randperm(len(mixed_labels), generator=t.Generator().manual_seed(seed))

    mixed_acts = {}
    for layer in layers:
        mixed_acts[layer] = t.cat(all_acts[layer], dim=0)[shuffle_idx]

    return mixed_acts, mixed_labels[shuffle_idx]


# ── Sweep Logic ──────────────────────────────────────────────────────────────

def sweep_mixed(best_act_set):
    """Load per-language data from best act_sets, build mixed datasets, sweep layers."""

    # Load all languages from their respective best act_sets
    all_train_acts, all_train_labels = {}, {}
    all_test_acts, all_test_labels = {}, {}
    all_nat_acts, all_nat_labels = {}, {}

    for lang in LANGUAGES:
        act_set = best_act_set[lang]
        train_path, test_path, nat_path = get_h5_paths(act_set)

        print(f"Loading {lang} from {act_set}...")
        all_train_acts[lang], all_train_labels[lang] = load_activations_and_labels(str(train_path), lang)
        all_test_acts[lang], all_test_labels[lang] = load_activations_and_labels(str(test_path), lang)
        all_nat_acts[lang], all_nat_labels[lang] = load_activations_and_labels(str(nat_path), lang)

    # Build mixed datasets
    print("\nBuilding mixed datasets...")
    mixed_train_acts, mixed_train_labels = make_mixed_dataset(
        all_train_acts, all_train_labels, n_total=600, languages=LANGUAGES, pair_based=True, seed=42)
    mixed_test_acts, mixed_test_labels = make_mixed_dataset(
        all_test_acts, all_test_labels, n_total=600, languages=LANGUAGES, pair_based=True, seed=43)
    mixed_nat_acts, mixed_nat_labels = make_mixed_dataset(
        all_nat_acts, all_nat_labels, n_total=250, languages=LANGUAGES, pair_based=False, seed=44)

    print(f"Mixed train: {len(mixed_train_labels)} samples, "
          f"test: {len(mixed_test_labels)}, nat: {len(mixed_nat_labels)}")
    print(f"Mixed train label balance: {mixed_train_labels.mean():.3f}")
    print(f"Mixed nat label balance:   {mixed_nat_labels.mean():.3f}")

    # Sweep layers
    results = []
    for layer in range(LAYER_START, LAYER_END + 1):
        row = {
            "act_set": "mixed",
            "lang": "mixed",
            "layer": layer,
        }

        for probe_name, probe_cls in [("mm", MMProbe), ("lr", LRProbe)]:
            try:
                probe = probe_cls.from_data(mixed_train_acts[layer], mixed_train_labels)

                # Train acc
                train_preds = probe.pred(mixed_train_acts[layer])
                row[f"{probe_name}_train_acc"] = (train_preds == mixed_train_labels).float().mean().item()

                # Test acc
                test_preds = probe.pred(mixed_test_acts[layer])
                row[f"{probe_name}_test_acc"] = (test_preds == mixed_test_labels).float().mean().item()

                # Natural ROC-AUC
                nat_probs = probe(mixed_nat_acts[layer])
                nat_preds = probe.pred(mixed_nat_acts[layer])

                nat_labels_np = mixed_nat_labels.detach().numpy()
                nat_probs_np = nat_probs.detach().numpy()
                nat_preds_np = nat_preds.detach().numpy()

                try:
                    row[f"{probe_name}_nat_roc_auc"] = roc_auc_score(nat_labels_np, nat_probs_np)
                except ValueError:
                    row[f"{probe_name}_nat_roc_auc"] = 0.5

                # Balanced accuracy
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
        print(f"  Layer {layer:2d} | MM nat_auc={row.get('mm_nat_roc_auc', 'N/A'):.3f}  "
              f"LR nat_auc={row.get('lr_nat_roc_auc', 'N/A'):.3f}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RQ2 Mixed-Language Probe Sweep")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to best_config.json from rq2_aggregate_sweep.py")
    args = parser.parse_args()

    # Load BEST_ACT_SET from config file or from hardcoded dict
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path) as f:
            config = json.load(f)
        best_act_set = config["best_act_set"]
        print(f"Loaded BEST_ACT_SET from {config_path}")
    elif BEST_ACT_SET:
        best_act_set = BEST_ACT_SET
        print("Using hardcoded BEST_ACT_SET")
    else:
        # Try default path
        default_config = RESULTS_DIR / "best_config.json"
        if default_config.exists():
            with open(default_config) as f:
                config = json.load(f)
            best_act_set = config["best_act_set"]
            print(f"Loaded BEST_ACT_SET from {default_config}")
        else:
            print("No BEST_ACT_SET found. Either:")
            print("  1. Run rq2_aggregate_sweep.py first")
            print("  2. Pass --config path/to/best_config.json")
            print("  3. Paste BEST_ACT_SET dict in this script")
            sys.exit(1)

    # Validate
    for lang in LANGUAGES:
        if lang not in best_act_set:
            print(f"ERROR: missing act_set for '{lang}' in config")
            sys.exit(1)

    print(f"\nBEST_ACT_SET: {best_act_set}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = sweep_mixed(best_act_set)

    outfile = RESULTS_DIR / "sweep_mixed.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} rows to {outfile}")
    print("Now re-run rq2_aggregate_sweep.py to include mixed in plots and analysis.")


if __name__ == "__main__":
    main()
