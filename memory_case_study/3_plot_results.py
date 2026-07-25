#!/usr/bin/env python3
"""
3_plot_results.py -- plot refusal rates from the summary produced by common.summarize().

Reads out/summary.csv (columns: target_model, gender, income, refusal_rate, n)
and writes:
  figures/fig1_by_gender.pdf   grouped bars, one pair (female/male) per model
  figures/fig2_by_income.pdf   grouped bars, one triple (high/mid/low income) per model

Error bars are 95% normal-approximation binomial confidence intervals computed
from the aggregated (refusal_rate, n) of each group. Models are sorted by
overall refusal rate, ascending.

Run:  python 3_plot_results.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SUMMARY_CSV

FIG_DIR = Path("./figures")

Z95 = 1.96


def short_name(model_id):
    """openrouter id -> short label, e.g. mistralai/mistral-small-24b-instruct-2501 -> mistral-small-24b"""
    name = model_id.split("/")[-1]
    name = re.sub(r"-instruct", "", name)
    name = re.sub(r"-a\d+b", "", name)        # MoE active-params suffix (…-a22b)
    name = re.sub(r"-\d{4}$", "", name)       # trailing release date (…-2501)
    return name


def aggregate(df, by):
    """Collapse the (model, gender, income) summary onto (model, `by`) with pooled n."""
    df = df.copy()
    df["refusals"] = df["refusal_rate"] * df["n"]
    g = df.groupby(["target_model", by])[["refusals", "n"]].sum().reset_index()
    g["refusal_rate"] = g["refusals"] / g["n"]
    g["ci95"] = Z95 * np.sqrt(g["refusal_rate"] * (1 - g["refusal_rate"]) / g["n"])
    return g


def grouped_bars(agg, by, levels, colors, labels, out_path):
    order = (agg.groupby("target_model")
                .apply(lambda m: m["refusals"].sum() / m["n"].sum())
                .sort_values().index.tolist())
    x = np.arange(len(order))
    width = 0.8 / len(levels)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (level, color, label) in enumerate(zip(levels, colors, labels)):
        sub = agg[agg[by] == level].set_index("target_model").reindex(order)
        pos = x + (i - (len(levels) - 1) / 2) * width
        ax.bar(pos, sub["refusal_rate"], width * 0.95, color=color,
               edgecolor="black", linewidth=0.6, label=label,
               yerr=sub["ci95"], capsize=3,
               error_kw=dict(elinewidth=1.2, capthick=1.2))
        for xp, (rate, ci) in zip(pos, sub[["refusal_rate", "ci95"]].itertuples(index=False)):
            ax.text(xp, rate + ci + 0.015, f"{rate:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Refusal rate")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in order], rotation=30, ha="right")
    ax.yaxis.grid(True, linestyle=":", color="grey", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(levels), frameon=False)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}")


def main():
    df = pd.read_csv(SUMMARY_CSV)

    by_gender = aggregate(df, "gender")
    grouped_bars(by_gender, "gender",
                 levels=["female", "male"],
                 colors=["cornflowerblue", "coral"],
                 labels=["Female", "Male"],
                 out_path=FIG_DIR / "fig1_by_gender.pdf")

    by_income = aggregate(df, "income")
    grouped_bars(by_income, "income",
                 levels=["high_income", "mid_income", "low_income"],
                 colors=["cornflowerblue", "mediumseagreen", "coral"],
                 labels=["High income", "Mid income", "Low income"],
                 out_path=FIG_DIR / "fig2_by_income.pdf")


if __name__ == "__main__":
    main()
