#!/usr/bin/env python3
"""
Merge per-language judge results, clean, and subsample.

Steps:
  1. Merge all judge_*.csv files from the results directory
  2. Report full statistics (before filtering)
  3. Drop unjudgeable rows (judge_label == -1)
  4. Drop low-confidence rows (judge_confidence < threshold)
  5. Keep only original_index values that survive filtering in ALL languages
  6. Subsample to N prompts (by original_index), keeping all languages
  7. Save both the full merged CSV and the clean subsampled CSV

Usage:
    python merge_and_clean.py <results_dir> <output_prefix> [options]

Examples:
    # Default: drop -1 and conf < 1.0, subsample to 250
    python merge_and_clean.py results/rq1_judge/ results/rq1 

    # Custom threshold and sample size
    python merge_and_clean.py results/rq1_judge/ results/rq1 --min-confidence 0.67 --n-sample 200

    # No subsampling (just filter)
    python merge_and_clean.py results/rq1_judge/ results/rq1 --n-sample 0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def print_stats(df, lang_col, title=""):
    """Print summary statistics for a dataframe with judge labels."""
    if title:
        print(f"\n{'='*60}")
        print(title)
        print("="*60)

    label_map = {0: "Sycophantic (NTA)", 1: "Non-sycophantic (YTA)", -1: "Unjudgeable"}
    for lv in sorted(df["judge_label"].unique()):
        count = (df["judge_label"] == lv).sum()
        pct = count / len(df) * 100
        print(f"  {label_map.get(lv, f'Unknown({lv})')}: {count} ({pct:.1f}%)")

    judgeable = df[df["judge_label"] >= 0]
    if len(judgeable) > 0:
        syc_rate = (judgeable["judge_label"] == 0).sum() / len(judgeable) * 100
        print(f"  Sycophancy rate (excl. unjudgeable): {syc_rate:.1f}%")

    if lang_col in df.columns:
        print(f"\n  {'Language':<15} {'Total':>6} {'YTA(1)':>7} {'NTA(0)':>7} {'-1':>5} {'Syc%':>7}")
        print(f"  {'-'*52}")
        for lang in sorted(df[lang_col].unique()):
            sub = df[df[lang_col] == lang]
            n_yta = (sub["judge_label"] == 1).sum()
            n_nta = (sub["judge_label"] == 0).sum()
            n_unk = (sub["judge_label"] == -1).sum()
            n_judg = n_yta + n_nta
            syc = n_nta / n_judg * 100 if n_judg > 0 else float("nan")
            print(f"  {lang:<15} {len(sub):>6} {n_yta:>7} {n_nta:>7} {n_unk:>5} {syc:>6.1f}%")

    if "judge_confidence" in df.columns:
        unanimous = (df["judge_confidence"] == 1.0).sum()
        print(f"\n  Unanimous agreement: {unanimous}/{len(df)} ({unanimous/len(df)*100:.1f}%)")
        print(f"  Mean confidence: {df['judge_confidence'].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-language judge results, clean, and subsample"
    )
    parser.add_argument("results_dir", type=str,
                        help="Directory containing judge_*.csv files")
    parser.add_argument("output_prefix", type=str,
                        help="Output prefix (produces <prefix>_merged.csv and <prefix>_clean.csv)")
    parser.add_argument("--lang-col", type=str, default="lang",
                        help="Language column name")
    parser.add_argument("--min-confidence", type=float, default=1.0,
                        help="Minimum judge confidence to keep (default: 1.0 = unanimous only)")
    parser.add_argument("--n-sample", type=int, default=250,
                        help="Number of prompts (original_index) to subsample to. 0 = no subsampling.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling")
    args = parser.parse_args()

    # ── Step 1: Merge ──
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
        print(f"  {f.name}: {len(df)} rows ({lang})")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    languages = sorted(merged[args.lang_col].unique())
    n_langs = len(languages)
    print(f"\nMerged: {len(merged)} rows, {n_langs} languages: {languages}")

    # ── Step 2: Stats before filtering ──
    print_stats(merged, args.lang_col, "BEFORE FILTERING")

    # Save full merged
    merged_path = f"{args.output_prefix}_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\nSaved full merged results to {merged_path}")

    # ── Step 3: Per-language filtering and subsampling ──
    print(f"\n{'='*60}")
    print("PER-LANGUAGE FILTERING")
    print("="*60)
    print(f"  Removing: unjudgeable (label == -1) and low confidence (< {args.min_confidence})")
    if args.n_sample > 0:
        print(f"  Then subsampling to {args.n_sample} per language (seed={args.seed})")
    print()

    clean_parts = []
    print(f"  {'Language':<15} {'Start':>6} {'Drop -1':>8} {'Drop conf':>10} {'Remain':>7} {'Sampled':>8}")
    print(f"  {'-'*58}")

    for lang in languages:
        lang_df = merged[merged[args.lang_col] == lang].copy()
        n_start = len(lang_df)

        # Filter unjudgeable
        lang_clean = lang_df[lang_df["judge_label"] >= 0]
        n_drop_unk = n_start - len(lang_clean)

        # Filter low confidence
        if "judge_confidence" in lang_clean.columns and args.min_confidence > 0:
            before_conf = len(lang_clean)
            lang_clean = lang_clean[lang_clean["judge_confidence"] >= args.min_confidence]
            n_drop_conf = before_conf - len(lang_clean)
        else:
            n_drop_conf = 0

        n_remain = len(lang_clean)

        # Subsample
        if args.n_sample > 0 and n_remain > args.n_sample:
            lang_clean = lang_clean.sample(n=args.n_sample, random_state=args.seed)
            n_sampled = args.n_sample
        else:
            n_sampled = n_remain

        print(f"  {lang:<15} {n_start:>6} {n_drop_unk:>8} {n_drop_conf:>10} {n_remain:>7} {n_sampled:>8}")
        clean_parts.append(lang_clean)

    clean = pd.concat(clean_parts, ignore_index=True)

    # Check if any language had fewer than n_sample
    if args.n_sample > 0:
        counts = clean.groupby(args.lang_col).size()
        min_count = counts.min()
        min_lang = counts.idxmin()
        if min_count < args.n_sample:
            print(f"\n  ⚠ Warning: {min_lang} only has {min_count} rows after filtering "
                  f"(target was {args.n_sample})")
            print(f"    Consider lowering --min-confidence or --n-sample")

    # ── Step 7: Final stats ──
    print_stats(clean, args.lang_col, "AFTER FILTERING & SUBSAMPLING")

    # Verify balanced
    counts_per_lang = clean.groupby(args.lang_col).size()
    if counts_per_lang.nunique() == 1:
        print(f"\n  ✓ Balanced: {counts_per_lang.iloc[0]} rows per language")
    else:
        print(f"\n  ⚠ Unbalanced counts per language:")
        for lang, count in counts_per_lang.items():
            print(f"    {lang}: {count}")

    # Save clean
    clean_path = f"{args.output_prefix}_clean.csv"
    clean.to_csv(clean_path, index=False)
    print(f"\nSaved clean results to {clean_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"  Input:          {len(merged)} rows ({n_langs} languages × {len(merged)//n_langs} each)")
    print(f"  After cleaning: {len(clean)} rows")
    counts = clean.groupby(args.lang_col).size()
    for lang in sorted(counts.index):
        print(f"    {lang}: {counts[lang]}")
    print(f"  Dropped total:  {len(merged) - len(clean)} rows")
    print(f"  Files:")
    print(f"    Full merged:  {merged_path}")
    print(f"    Clean subset: {clean_path}")


if __name__ == "__main__":
    main()
