import pandas as pd
import os
import glob
import numpy as np
from scipy.stats import kendalltau

EXPECTED_COLS = [
    "First most important factor",
    "Second most important factor",
    "Third most important factor",
    "Fourth most important factor",
    "Fifth most important factor",
]

def normalize_col(c):
    return str(c).strip()

def compute_kendall_w(rankings):
    all_factors = sorted({item for sublist in rankings for item in sublist if isinstance(item, str) and item.strip()})
    matrix = []
    for judge_ranking in rankings:
        judge_rank = []
        for factor in all_factors:
            if factor in judge_ranking:
                judge_rank.append(judge_ranking.index(factor) + 1)
            else:
                judge_rank.append(np.nan)
        matrix.append(judge_rank)
    matrix = np.array(matrix).T  # shape: (factors, judges)
    mask = ~np.isnan(matrix).any(axis=1)
    matrix_clean = matrix[mask, :]
    n_judges = matrix_clean.shape[1]
    ws = []
    for i in range(n_judges):
        for j in range(i + 1, n_judges):
            tau, _ = kendalltau(matrix_clean[:, i], matrix_clean[:, j])
            ws.append(tau)
    kendall_w = np.nanmean(ws) if ws else np.nan
    return kendall_w

def borda_count_from_file(file_path, output_path=None):
    df = pd.read_excel(file_path)
    df.columns = [normalize_col(c) for c in df.columns]
    for col in EXPECTED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in file {file_path}")
    rank_cols = EXPECTED_COLS

    # Collect rankings for Kendall's W
    all_rankings = []
    long_rows = []
    for _, row in df.iterrows():
        ranking = []
        for i, col in enumerate(rank_cols, start=1):
            factor = row[col]
            if pd.isna(factor) or str(factor).strip() == "":
                continue
            ranking.append(str(factor).strip())
            points = 6 - i
            long_rows.append({"factor": str(factor).strip(), "rank": i, "points": points})
        all_rankings.append(ranking)

    long_df = pd.DataFrame(long_rows)
    results = (
        long_df.groupby("factor", as_index=False)["points"]
        .sum()
        .sort_values(["points", "factor"], ascending=[False, True])
        .reset_index(drop=True)
    )
    results["final_rank"] = results["points"].rank(method="dense", ascending=False).astype(int)

    # Compute Kendall's W
    kendall_w = compute_kendall_w(all_rankings)
    results["kendall_w"] = kendall_w  # Add to every row

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Saved Borda results to {output_path} (with kendall_w={kendall_w:.3f})")

    print(f"Kendall's W for {os.path.basename(file_path)}: {kendall_w:.3f}")

    return results, kendall_w

if __name__ == "__main__":
    files = glob.glob("survey_results/relevance_*.xlsx")
    kendall_results = []
    for file in files:
        basename = os.path.splitext(os.path.basename(file))[0]
        out = f"survey_results/{basename}_borda.csv"
        _, kendall_w = borda_count_from_file(file, output_path=out)
        kendall_results.append({"file": basename, "kendall_w": kendall_w})
    # Save all Kendall's W results for tracking
    pd.DataFrame(kendall_results).to_csv("survey_results/kendall_w_summary.csv", index=False)
    print("Saved Kendall's W summary to survey_results/kendall_w_summary.csv")
