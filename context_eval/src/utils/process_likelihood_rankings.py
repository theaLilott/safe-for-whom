import pandas as pd
import glob
import os
import numpy as np
import re
from collections import Counter

# --- Canonical theme/topic mapping ---
THEME_TO_TOPIC = {
    'Small Investments': 'finance',
    'Savings Strategy': 'finance',
    'Debt Management': 'finance',
    'Home Ownership & Major Purchases': 'finance',
    'Diet & Weight Management': 'health',
    'Mental Health & Stress Management': 'health',
    'Managing Substance Use or High-Risk Habits': 'health',
    'Family Planning & Health': 'health',
    'Handling Identity-Based Workplace Conflict': 'career',
    'Negotiation & Promotion': 'career',
    'Navigating Systemic Burnout & a Toxic Work Environment': 'career',
    'Career Planning': 'career'
}

def clean_theme_from_col(col):
    c = col.replace('\xa0', ' ').strip()
    c = re.sub(r"( - First most important factor)+", " - First most important factor", c)
    if c.endswith(" - First most important factor"):
        c = c[:-len(" - First most important factor")]
    return c.strip()

def find_best_column(df, expected):
    expected_clean = expected.replace('\xa0', ' ').strip()
    for c in df.columns:
        c_clean = c.replace('\xa0', ' ').strip()
        if c_clean == expected_clean:
            return c
    expected_nospace = expected_clean.replace(' ', '').lower()
    for c in df.columns:
        c_nospace = c.replace('\xa0', '').replace(' ', '').lower()
        if expected_nospace in c_nospace:
            return c
    for c in df.columns:
        if expected_clean.split(' - ')[0] in c:
            return c
    return None

def compute_kendalls_w(rankings, valid_factors):
    if not rankings or len(rankings) < 2:
        return None
    m = len(valid_factors)
    if m < 2:
        return None
    n = len(rankings)
    factor_index = {f: j for j, f in enumerate(valid_factors)}
    rank_matrix = [[0.0] * m for _ in range(n)]
    tie_correction_total = 0.0
    for i, ranking in enumerate(rankings):
        seen = set()
        cleaned = []
        for f in ranking:
            if isinstance(f, str):
                s = f.strip()
                if s in factor_index and s not in seen:
                    cleaned.append(s)
                    seen.add(s)
        k = len(cleaned)
        for pos, f in enumerate(cleaned):
            j = factor_index[f]
            rank_matrix[i][j] = float(pos + 1)
        t = m - k
        if t > 0:
            avg_tied_rank = (k + 1 + m) / 2.0
            for f, j in factor_index.items():
                if f not in seen:
                    rank_matrix[i][j] = avg_tied_rank
            tie_correction_total += (t ** 3 - t)
    R = [sum(rank_matrix[i][j] for i in range(n)) for j in range(m)]
    R_bar = n * (m + 1) / 2.0
    S = sum((Rj - R_bar) ** 2 for Rj in R)
    denom = (n ** 2) * (m ** 3 - m) - n * tie_correction_total
    if denom <= 0:
        return None
    W = 12.0 * S / denom
    if W < 0:
        W = 0.0
    if W > 1:
        W = 1.0
    return float(W)

def collect_all_context_factors(files):
    factor_set = set()
    whiches = ["First", "Second", "Third", "Fourth", "Fifth"]
    for file in files:
        xl = pd.ExcelFile(file)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            for col in df.columns:
                if any(w in col for w in whiches):
                    for v in df[col].dropna():
                        s = str(v).strip()
                        if s and s.lower() != "nan":
                            factor_set.add(s)
    return sorted(factor_set)

def process_theme(df, theme, valid_factors):
    whiches = ["First", "Second", "Third", "Fourth", "Fifth"]
    ranking_cols = []
    for which in whiches:
        expected_col = f"{theme} - {which} most important factor"
        best_col = find_best_column(df, expected_col)
        if best_col is not None:
            ranking_cols.append(best_col)
        else:
            raise RuntimeError(f"ERROR: Could not find column for '{expected_col}' in theme '{theme}'")
    rankings = []
    for _, row in df.iterrows():
        r = [str(row[c]).strip() for c in ranking_cols if pd.notnull(row[c]) and str(row[c]).strip()]
        if r:
            rankings.append(r)
    # Borda count (using valid_factors to ensure top5 are present in order)
    factor_scores = Counter()
    max_rank = len(valid_factors)
    for r in rankings:
        for i, f in enumerate(r):
            factor_scores[f] += max_rank - i
    final_borda = [f for f, _ in factor_scores.most_common()]
    top5 = []
    for f in final_borda:
        if f not in top5:
            top5.append(f)
        if len(top5) == 5:
            break
    # Pad to 5 if needed (rare, if everyone omits a factor)
    for f in valid_factors:
        if f not in top5 and len(top5) < 5:
            top5.append(f)
    w = compute_kendalls_w(rankings, valid_factors)
    return top5, w

def main():
    DATA_DIR = 'survey_results'
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, 'likelihood_survey_*.xlsx')) +
        glob.glob(os.path.join(DATA_DIR, 'likelihood_surveys_*.xlsx')) +
        glob.glob(os.path.join(DATA_DIR, 'linkelihood_survey_*.xlsx'))
    )
    if not files:
        raise RuntimeError(f"ERROR: No survey Excel files found in {DATA_DIR}")
    # --- Collect the union of all context factors ---
    valid_factors = collect_all_context_factors(files)
    print(f"Using {len(valid_factors)} unique context factors across all themes: {valid_factors}")
    # --- Canonical output order for rows and ids ---
    theme_topic_order = [
        ('Small Investments', 'finance'),
        ('Savings Strategy', 'finance'),
        ('Debt Management', 'finance'),
        ('Home Ownership & Major Purchases', 'finance'),
        ('Diet & Weight Management', 'health'),
        ('Mental Health & Stress Management', 'health'),
        ('Managing Substance Use or High-Risk Habits', 'health'),
        ('Family Planning & Health', 'health'),
        ('Handling Identity-Based Workplace Conflict', 'career'),
        ('Negotiation & Promotion', 'career'),
        ('Navigating Systemic Burnout & a Toxic Work Environment', 'career'),
        ('Career Planning', 'career')
    ]
    theme_to_rownum = {}
    for ix, (theme, topic) in enumerate(theme_topic_order):
        prefix = topic[0]
        localnum = sum(1 for t, top in theme_topic_order[:ix] if top == topic)
        theme_to_rownum[theme] = (prefix, localnum)
    theme_results = {}
    for file in files:
        xl = pd.ExcelFile(file)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            themes = set()
            for col in df.columns:
                if "First most important factor" in col:
                    theme = clean_theme_from_col(col)
                    themes.add(theme)
            for theme in themes:
                theme_clean = theme.strip()
                if theme_clean not in THEME_TO_TOPIC:
                    raise RuntimeError(f"ERROR: Theme in file ({theme_clean}) not found in canonical mapping. Please check THEME_TO_TOPIC and input files.")
                if theme_clean not in theme_to_rownum:
                    raise RuntimeError(f"ERROR: Theme ({theme_clean}) not in canonical output order list.")
                topic = THEME_TO_TOPIC[theme_clean]
                top5, w = process_theme(df, theme_clean, valid_factors)
                prefix, num = theme_to_rownum[theme_clean]
                id_val = f"{prefix}{num}"
                theme_results[theme_clean] = {
                    'id': id_val,
                    'topic': topic,
                    'theme': theme_clean,
                    '1': top5[0] if len(top5) > 0 else '',
                    '2': top5[1] if len(top5) > 1 else '',
                    '3': top5[2] if len(top5) > 2 else '',
                    '4': top5[3] if len(top5) > 3 else '',
                    '5': top5[4] if len(top5) > 4 else '',
                    'kendalls_w': round(w, 2) if w is not None else ''
                }
    # --- Check for missing themes ---
    missing = []
    for theme, _ in theme_topic_order:
        if theme not in theme_results:
            missing.append(theme)
    if missing:
        print("ERROR: The following canonical themes were not found in your input files or were not processed:")
        for m in missing:
            print(f"  - {m}")
        print("Please check your input survey files and column headers.")
        raise RuntimeError("Missing required theme(s).")
    # --- Output in canonical order ---
    out_rows = []
    for theme, topic in theme_topic_order:
        row = theme_results[theme]
        out_rows.append(row)
    out_df = pd.DataFrame(out_rows)
    cols = ['id', 'topic', 'theme', '1', '2', '3', '4', '5', 'kendalls_w']
    out_df = out_df[cols]
    out_df.to_csv("eval_dataset/likelihood_rankings.csv", index=False)
    print("Done! Saved to eval_dataset/likelihood_rankings.csv (all canonical themes processed)")

if __name__ == '__main__':
    main()
