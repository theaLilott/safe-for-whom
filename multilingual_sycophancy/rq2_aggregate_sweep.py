"""
RQ2 Probe Sweep Aggregator
===========================
Reads all sweep_*.json files and produces three outputs:

1. BEST_ACT_SET:   {lang: suffix}
   For each language, which activation set gave better results overall.

2. BEST_LAYERS:    {lang: {"mm": layer, "lr": layer}}
   For each language (using its best act_set), which layer is best per probe type.

3. BEST_UNIVERSAL: {"mm": layer, "lr": layer}
   For each probe type, which single layer is best on average across all languages
   (using each language's best act_set).

Usage:
    python rq2_aggregate_sweep.py
    python rq2_aggregate_sweep.py --metric nat_roc_auc       # default
    python rq2_aggregate_sweep.py --metric nat_bal_acc        # alternative
    python rq2_aggregate_sweep.py --min-test-acc 0.9          # stricter filter
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
import numpy as np

BASE_DIR = Path(os.environ.get(
    "RQ2_BASE_DIR",
    "/NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy"
))
RESULTS_DIR = BASE_DIR / "results" / "probe_sweep"

PROBE_TYPES = ["mm", "lr"]


def load_sweep_data():
    """Load all sweep JSON files into a single DataFrame."""
    all_rows = []
    json_files = sorted(RESULTS_DIR.glob("sweep_*.json"))

    if not json_files:
        print(f"No sweep files found in {RESULTS_DIR}")
        print("Run the sweep first: sbatch run_sweep.sh")
        return None

    for jf in json_files:
        try:
            with open(jf) as f:
                rows = json.load(f)
            all_rows.extend(rows)
            print(f"  Loaded {len(rows):3d} rows from {jf.name}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  SKIPPED {jf.name} (corrupt/empty: {e})")

    if not all_rows:
        print("No valid results found. Check if sweep jobs have finished.")
        return None

    return pd.DataFrame(all_rows)


def pick_best_act_set(df, metric_suffix, min_test_acc):
    """
    Step 1: For each language, pick the activation set that gives better results
    overall (averaged across all viable layers and both probe types).

    Returns: dict {lang: act_set_suffix}
    """
    print(f"\n{'='*80}")
    print("STEP 1: Best activation set per language")
    print(f"{'='*80}")

    scores = []
    for lang in sorted(df["lang"].unique()):
        for act_set in df["act_set"].unique():
            subset = df[(df["lang"] == lang) & (df["act_set"] == act_set)]

            # Average the metric across all layers and both probe types
            lang_scores = []
            for pt in PROBE_TYPES:
                test_col = f"{pt}_test_acc"
                metric_col = f"{pt}_{metric_suffix}"
                viable = subset[subset[test_col] >= min_test_acc]
                if not viable.empty and metric_col in viable.columns:
                    lang_scores.append(viable[metric_col].mean())

            avg_score = np.mean(lang_scores) if lang_scores else 0.0
            best_per_probe = {}
            for pt in PROBE_TYPES:
                metric_col = f"{pt}_{metric_suffix}"
                test_col = f"{pt}_test_acc"
                viable = subset[subset[test_col] >= min_test_acc]
                if not viable.empty and metric_col in viable.columns:
                    best_per_probe[pt] = viable[metric_col].max()

            scores.append({
                "lang": lang,
                "act_set": act_set,
                "avg_score": avg_score,
                **{f"best_{pt}": best_per_probe.get(pt, 0.0) for pt in PROBE_TYPES},
            })

    scores_df = pd.DataFrame(scores)
    print(scores_df.to_string(index=False))

    # Pick the act_set with highest average score per language
    best_act_set = {}
    for lang in sorted(df["lang"].unique()):
        lang_df = scores_df[scores_df["lang"] == lang]
        winner = lang_df.loc[lang_df["avg_score"].idxmax()]
        best_act_set[lang] = winner["act_set"]

    print("\nBEST_ACT_SET = {")
    for lang, act_set in sorted(best_act_set.items()):
        print(f'    "{lang}": "{act_set}",')
    print("}")

    return best_act_set


def pick_best_layers_per_lang(df, best_act_set, metric_suffix, min_test_acc):
    """
    Step 2: Given the selected act_set per language, find the best layer
    for each probe type per language.

    Returns: dict {lang: {"mm": {...}, "lr": {...}}}
    """
    print(f"\n{'='*80}")
    print("STEP 2: Best layer per language per probe type")
    print(f"{'='*80}")

    best_layers = {}

    # Include mixed if present in the data
    all_langs = sorted(best_act_set.keys())
    has_mixed = "mixed" in df["lang"].values
    if has_mixed:
        all_langs = all_langs + ["mixed"]

    for lang in all_langs:
        if lang == "mixed":
            act_set = "mixed"
        else:
            act_set = best_act_set[lang]
        subset = df[(df["lang"] == lang) & (df["act_set"] == act_set)]
        best_layers[lang] = {}

        for pt in PROBE_TYPES:
            test_col = f"{pt}_test_acc"
            metric_col = f"{pt}_{metric_suffix}"
            viable = subset[subset[test_col] >= min_test_acc]

            if viable.empty or metric_col not in viable.columns:
                best_layers[lang][pt] = None
                continue

            best_row = viable.loc[viable[metric_col].idxmax()]
            best_layers[lang][pt] = {
                "layer": int(best_row["layer"]),
                "score": float(best_row[metric_col]),
                "test_acc": float(best_row[test_col]),
            }

    # Print table
    rows = []
    for lang in sorted(best_layers.keys()):
        row = {"lang": lang, "act_set": best_act_set.get(lang, "mixed")}
        for pt in PROBE_TYPES:
            info = best_layers[lang][pt]
            if info:
                row[f"{pt}_layer"] = info["layer"]
                row[f"{pt}_score"] = f"{info['score']:.3f}"
                row[f"{pt}_test_acc"] = f"{info['test_acc']:.3f}"
            else:
                row[f"{pt}_layer"] = "N/A"
                row[f"{pt}_score"] = "N/A"
                row[f"{pt}_test_acc"] = "N/A"
        rows.append(row)

    print(pd.DataFrame(rows).to_string(index=False))

    # Print copy-paste dict
    print("\nBEST_LAYERS = {")
    for lang in sorted(best_layers.keys()):
        mm_l = best_layers[lang]["mm"]["layer"] if best_layers[lang]["mm"] else "None"
        lr_l = best_layers[lang]["lr"]["layer"] if best_layers[lang]["lr"] else "None"
        print(f'    "{lang}": {{"mm": {mm_l}, "lr": {lr_l}}},')
    print("}")

    return best_layers


def pick_best_universal_layer(df, best_act_set, metric_suffix, min_test_acc):
    """
    Step 3: For each probe type, find the single layer that performs best
    on average across all languages (using each language's best act_set).

    Returns: dict {"mm": layer, "lr": layer}
    """
    print(f"\n{'='*80}")
    print("STEP 3: Best universal layer per probe type (averaged across languages)")
    print(f"{'='*80}")

    # Build a filtered df using each language's best act_set
    filtered_parts = []
    for lang, act_set in best_act_set.items():
        filtered_parts.append(df[(df["lang"] == lang) & (df["act_set"] == act_set)])
    filtered = pd.concat(filtered_parts)

    best_universal = {}

    for pt in PROBE_TYPES:
        test_col = f"{pt}_test_acc"
        metric_col = f"{pt}_{metric_suffix}"
        viable = filtered[filtered[test_col] >= min_test_acc]

        if viable.empty:
            best_universal[pt] = None
            continue

        # Average the metric across languages for each layer
        layer_avg = viable.groupby("layer")[metric_col].mean()
        layer_std = viable.groupby("layer")[metric_col].std()
        layer_count = viable.groupby("layer")[metric_col].count()

        summary = pd.DataFrame({
            "layer": layer_avg.index,
            "mean_score": layer_avg.values,
            "std_score": layer_std.values,
            "n_langs": layer_count.values,
        }).sort_values("mean_score", ascending=False)

        print(f"\n{pt.upper()} Probe — top 5 layers by avg {metric_suffix}:")
        print(summary.head(5).to_string(index=False))

        best_layer = int(summary.iloc[0]["layer"])
        best_universal[pt] = best_layer

    print(f"\nBEST_UNIVERSAL_LAYER = {{")
    for pt in PROBE_TYPES:
        layer = best_universal[pt]
        print(f'    "{pt}": {layer},')
    print("}")

    return best_universal

"""# ─── Shared color palette (Tol's muted, colorblind-safe) ───
LANG_COLORS = {
    "english":    "#332288",  # indigo
    "german":     "#88CCEE",  # cyan
    "indonesian": "#44AA99",  # teal
    "spanish":    "#117733",  # green
    "italian":    "#999933",  # olive
    "arabic":     "#DDCC77",  # sand
    "japanese":   "#CC6677",  # rose
    "russian":    "#882255",  # wine
    "thai":       "#AA4499",  # purple
    "mixed":      "#000000",  # black (special)
}

def plot_layer_curves(df, best_act_set, metric_suffix):
    
    For each probe type, plot nat ROC-AUC vs layer with one line per language.
    Uses each language's best act_set. Includes mixed if present. Saves to PDF.

    # Filter to each language's best act_set
    filtered_parts = []
    for lang, act_set in best_act_set.items():
        filtered_parts.append(df[(df["lang"] == lang) & (df["act_set"] == act_set)])
    # Include mixed if present
    has_mixed = "mixed" in df["lang"].values
    if has_mixed:
        filtered_parts.append(df[(df["lang"] == "mixed") & (df["act_set"] == "mixed")])
    filtered = pd.concat(filtered_parts)

    lang_ordered = ["english", "german", "spanish", "italian", "russian", "arabic", "indonesian", "thai"]
    all_langs = [l for l in lang_ordered if l in best_act_set]
    if has_mixed and ("mixed" not in all_langs):
        all_langs.append("mixed")
    elif has_mixed:
        all_langs.remove("mixed")
        all_langs.append("mixed")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, pt in zip(axes, PROBE_TYPES):
        metric_col = f"{pt}_{metric_suffix}"

        for lang in all_langs:
            lang_data = filtered[filtered["lang"] == lang].sort_values("layer")
            color = LANG_COLORS[lang]
            style = {
                "color": color,
                "linestyle": "--", "linewidth": 2.5,
                "marker": "s", "markersize": 4,
            } if lang == "mixed" else {
                "color": color,
                "linewidth": 1.5,
                "marker": "o", "markersize": 3,
            }
            ax.plot(lang_data["layer"], lang_data[metric_col],
                    label=lang.capitalize(), **style)

        ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"ROC-AUC-Score")
        ax.set_title(f"{'Difference-of-Means' if pt == 'mm' else 'Logistic Regression'} Probe")
        ax.legend(fontsize=8)
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.2)

    fig.suptitle(f"ROC-AUC-Score for Probe Predictions on Activations from Natural Completions by Layer and Language ", fontsize=13)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "layer_curves.pdf"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Layer curve plot saved to {plot_path}")
 """
 


# ─── Shared color palette: sample evenly from RdBu ─────────
# ─── Shared color palette: skip pale center of RdBu ─────────
_LANG_ORDER = ["english", "german", "spanish", "italian", "russian", "arabic", "indonesian", "thai"]
_n = len(_LANG_ORDER)
# Sample from [0, 0.38] and [0.62, 1.0] — avoids the washed-out middle
_positions = [i * 0.30 / (_n // 2 - 1) for i in range(_n // 2)] + \
             [0.70 + i * 0.30 / (_n // 2 - 1) for i in range(_n // 2)]
_samples = sample_colorscale("RdBu", _positions)
LANG_COLORS = {lang: col for lang, col in zip(_LANG_ORDER, _samples)}
LANG_COLORS["mixed"] = "red"


def plot_layer_curves(df, best_act_set, metric_suffix):
    filtered_parts = []
    for lang, act_set in best_act_set.items():
        filtered_parts.append(df[(df["lang"] == lang) & (df["act_set"] == act_set)])
    has_mixed = "mixed" in df["lang"].values
    if has_mixed:
        filtered_parts.append(df[(df["lang"] == "mixed") & (df["act_set"] == "mixed")])
    filtered = pd.concat(filtered_parts)

    all_langs = [l for l in _LANG_ORDER if l in best_act_set]
    if has_mixed:
        if "mixed" in all_langs:
            all_langs.remove("mixed")
        all_langs.append("mixed")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Difference-of-Means Probe", "Logistic Regression Probe"],
        horizontal_spacing=0.1,
    )

    for idx, pt in enumerate(PROBE_TYPES):
        metric_col = f"{pt}_{metric_suffix}"

        for lang in all_langs:
            lang_data = filtered[filtered["lang"] == lang].sort_values("layer")
            color = LANG_COLORS[lang]
            is_mixed = lang == "mixed"

            fig.add_trace(
                go.Scatter(
                    x=lang_data["layer"],
                    y=lang_data[metric_col],
                    mode="lines+markers",
                    name=lang.capitalize(),
                    line=dict(
                        color=color,
                        width=2.5 if is_mixed else 1.5,
                        dash= "solid",
                    ),
                    marker=dict(
                        size=5 if is_mixed else 4,
                        symbol="square" if is_mixed else "circle",
                    ),
                    legendgroup=lang,
                    showlegend=(idx == 0),  # legend only once per language
                ),
                row=1, col=idx + 1,
            )

        # Chance-level baseline
        fig.add_hline(y=0.5, line=dict(color="grey", dash="dash", width=0.8),
                       opacity=0.5, row=1, col=idx + 1)

    # ─── Axes: grid + visible spines ─────────────────────────
    axis_common = dict(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
        gridwidth=1,
        showline=True,
        linecolor="rgba(0,0,0,0.3)",
        linewidth=1,
        zeroline=False,
    )

    fig.update_yaxes(title_text="ROC-AUC-Score", range=[0.25, 1.0], **axis_common, row=1, col=1)
    fig.update_yaxes(range=[0.25, 1.0], **axis_common, row=1, col=2)
    fig.update_xaxes(title_text="Layer", **axis_common, row=1, col=1)
    fig.update_xaxes(title_text="Layer", **axis_common, row=1, col=2)


    fig.update_layout(
        font=dict(size=11),
        height=300,
        width=900,
        plot_bgcolor="white",
        legend=dict(font=dict(size=10)),
    )
    fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.4,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
    )
)

    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12)

    plot_path = RESULTS_DIR / "layer_curves.pdf"
    fig.write_image("figures/rq2_layer_curves.png", scale= 3)
    print(f"Layer curve plot saved to {plot_path}")
    #fig.show()

def main():
    parser = argparse.ArgumentParser(description="RQ2 Probe Sweep Aggregator")
    parser.add_argument("--metric", default="nat_roc_auc",
                        help="Metric suffix to maximize, without probe prefix (default: nat_roc_auc)")
    parser.add_argument("--min-test-acc", type=float, default=0.8,
                        help="Minimum test accuracy to consider a layer viable (default: 0.8)")
    args = parser.parse_args()

    print(f"Metric: *_{args.metric}  |  Min test acc: {args.min_test_acc}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_sweep_data()
    if df is None:
        return

    print(f"\nTotal rows: {len(df)}")
    print(f"Activation sets: {sorted(df['act_set'].unique())}")
    print(f"Languages: {sorted(df['lang'].unique())}")
    print(f"Layers: {sorted(df['layer'].unique())}")

    # ── Step 1: Best act_set per language ─────────────────────────────────────
    best_act_set = pick_best_act_set(df, args.metric, args.min_test_acc)

    # ── Step 2: Best layer per language per probe type ────────────────────────
    best_layers = pick_best_layers_per_lang(df, best_act_set, args.metric, args.min_test_acc)

    # ── Step 3: Best universal layer per probe type ───────────────────────────
    best_universal = pick_best_universal_layer(df, best_act_set, args.metric, args.min_test_acc)

    # ── Plot layer curves ─────────────────────────────────────────────────────
    plot_layer_curves(df, best_act_set, args.metric)

    # ── Save everything ───────────────────────────────────────────────────────
    outfile = RESULTS_DIR / "sweep_summary.csv"
    df.to_csv(outfile, index=False)

    config = {
        "metric": args.metric,
        "min_test_acc": args.min_test_acc,
        "best_act_set": best_act_set,
        "best_layers": {
            lang: {pt: info["layer"] if info else None for pt, info in probes.items()}
            for lang, probes in best_layers.items()
        },
        "best_universal_layer": best_universal,
    }
    config_path = RESULTS_DIR / "best_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Full results saved to {outfile}")
    print(f"Config saved to {config_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()