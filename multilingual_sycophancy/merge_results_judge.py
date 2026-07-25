#!/usr/bin/env python3
"""
Merge per-language judge results into a single CSV and print summary statistics.

Usage:
    python merge_results.py <results_dir> <output_csv> [--lang-col lang]

Example:
    python merge_results.py results/rq1_qwen3_14b/ results/rq1_qwen3_14b_judged.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge per-language judge results")
    parser.add_argument("results_dir", type=str, help="Directory containing judge_*.csv files")
    parser.add_argument("output_csv", type=str, help="Output merged CSV path")
    parser.add_argument("--lang-col", type=str, default="lang",
                        help="Language column name for reporting")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_files = sorted(results_dir.glob("judge_*.csv"))

    if not csv_files:
        print(f"ERROR: No judge_*.csv files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} result files:")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        lang = f.stem.replace("judge_", "")
        print(f"  {f.name}: {len(df)} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged: {len(merged)} total rows")

    # ── Summary statistics ──
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    label_map = {0: "Sycophantic (NTA)", 1: "Non-sycophantic (YTA)", -1: "Unjudgeable"}
    for label_val in sorted(merged["judge_label"].unique()):
        count = (merged["judge_label"] == label_val).sum()
        pct = count / len(merged) * 100
        print(f"  {label_map.get(label_val, f'Unknown({label_val})')}: {count} ({pct:.1f}%)")

    judgeable = merged[merged["judge_label"] >= 0]
    if len(judgeable) > 0:
        syc_rate = (judgeable["judge_label"] == 0).sum() / len(judgeable) * 100
        print(f"\n  Overall sycophancy rate: {syc_rate:.1f}%")

    # ── Per-language breakdown ──
    if args.lang_col in merged.columns:
        print(f"\n{'Language':<15} {'Total':>6} {'YTA(1)':>7} {'NTA(0)':>7} {'-1':>5} {'Syc%':>7}")
        print("-" * 50)
        for lang in sorted(merged[args.lang_col].unique()):
            sub = merged[merged[args.lang_col] == lang]
            n_total = len(sub)
            n_yta = (sub["judge_label"] == 1).sum()
            n_nta = (sub["judge_label"] == 0).sum()
            n_unk = (sub["judge_label"] == -1).sum()
            n_judg = n_yta + n_nta
            syc = n_nta / n_judg * 100 if n_judg > 0 else float("nan")
            print(f"  {lang:<13} {n_total:>6} {n_yta:>7} {n_nta:>7} {n_unk:>5} {syc:>6.1f}%")

    # ── Parse quality ──
    if "judge_parse_success_pass0" in merged.columns:
        # Count parse failures across all passes
        pass_cols = [c for c in merged.columns if c.startswith("judge_parse_success_pass")]
        for col in pass_cols:
            n_fail = (~merged[col]).sum()
            if n_fail > 0:
                print(f"  {col}: {n_fail} parse failures")

    # ── Inter-pass agreement ──
    if "judge_confidence" in merged.columns:
        print(f"\n{'='*60}")
        print("INTER-PASS AGREEMENT")
        print("="*60)
        unanimous = (merged["judge_confidence"] == 1.0).sum()
        print(f"  Unanimous: {unanimous}/{len(merged)} ({unanimous/len(merged)*100:.1f}%)")
        mean_conf = merged["judge_confidence"].mean()
        print(f"  Mean confidence: {mean_conf:.3f}")

        if args.lang_col in merged.columns:
            print(f"\n  {'Language':<15} {'Unanimous%':>10} {'MeanConf':>10}")
            print(f"  {'-'*37}")
            for lang in sorted(merged[args.lang_col].unique()):
                sub = merged[merged[args.lang_col] == lang]
                u = (sub["judge_confidence"] == 1.0).sum() / len(sub) * 100
                mc = sub["judge_confidence"].mean()
                print(f"  {lang:<15} {u:>9.1f}% {mc:>10.3f}")

    # ── Save ──
    merged.to_csv(args.output_csv, index=False)
    print(f"\nSaved merged results to {args.output_csv}")


if __name__ == "__main__":
    main()
