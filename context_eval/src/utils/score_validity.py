import pandas as pd

CSV = "eval_dataset/safety_scores_o3.csv"  # update path if needed
df = pd.read_csv(CSV)

def score_status(series):
    allowed = {0, 1, 2, 3, 4, 5}
    n_nan = series.isna().sum()
    not_allowed = series[~series.isin(allowed) & ~series.isna()]
    n_invalid = not_allowed.shape[0]
    n_valid = series.isin(allowed).sum()
    return n_nan, n_invalid, n_valid, not_allowed

for col in ["apparent_safety_score", "contextualized_safety_score"]:
    print(f"\n=== {col} ===")
    n_nan, n_invalid, n_valid, not_allowed = score_status(df[col])
    print(f"NaN: {n_nan}")
    print(f"Invalid (not 0-5): {n_invalid}")
    if n_invalid > 0:
        print("  Offending values:")
        print(not_allowed.values)
    print(f"Valid (0-5): {n_valid}")
