import re
import pandas as pd
from typing import Dict, List, Tuple, Optional

def _norm(s: str) -> str:
    s = re.sub(r"[\s_/\\\-]+", " ", str(s).lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _find_col_contains(cols, must_have_all=None, must_have_any=None) -> Optional[str]:
    normed = {c: _norm(c) for c in cols}
    for c, lc in normed.items():
        ok_all = all(tok in lc for tok in (must_have_all or []))
        ok_any = True if not must_have_any else any(tok in lc for tok in must_have_any)
        if ok_all and ok_any:
            return c
    return None

def load_profiles_from_raw(
    prof_path: str,
    factors: List[str],
    id_to_topic_theme: Dict[str, Tuple[str, str]],
) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    """
    Load the vulnerability profiles file and return:
      - df: filtered to known IDs, with columns [topic, theme, __level_norm__, profile_id] added
      - level_col: resolved vulnerability 'level' column name (exact or fuzzy)
      - factor_col_map: mapping canonical factor -> actual column name in the file (fuzzy-tolerant)
    """
    df = pd.read_excel(prof_path)

    if "ID" not in df.columns:
        raise ValueError("Profiles must have an 'ID' column like c0/h2/f3.")

    # Filter to rows with known IDs and attach topic/theme
    topics, themes, keep_idx = [], [], []
    for idx, x in enumerate(df["ID"].astype(str).str.strip()):
        if x not in id_to_topic_theme:
            continue
        t, th = id_to_topic_theme[x]
        topics.append(t)
        themes.append(th)
        keep_idx.append(idx)
    df = df.iloc[keep_idx].reset_index(drop=True)
    df["topic"] = topics
    df["theme"] = themes

    # Resolve the vulnerability level column
    if "level" in df.columns:
        level_col = "level"
    else:
        level_col = _find_col_contains(df.columns, must_have_all=["vulner"])
    if not level_col:
        raise ValueError("Need a 'level' column OR a header containing 'vulner' for vulnerability level.")

    # Special-case mapping for Health Status/Disabilities
    health_exact = "Health Status/Disabilities"
    if health_exact in df.columns:
        health_col = health_exact
    else:
        health_col = _find_col_contains(
            df.columns,
            must_have_all=["health"],
            must_have_any=["status/disabilities", "status", "disabil"]
        )

    # Build factor
    factor_col_map: Dict[str, str] = {}
    for f in factors:
        if f == health_exact and health_col:
            factor_col_map[f] = health_col
            continue
        if f in df.columns:
            factor_col_map[f] = f
            continue
        tokens = _norm(f).split()
        guess = _find_col_contains(df.columns, must_have_all=tokens)
        factor_col_map[f] = guess if guess else f  # may be missing; caller handles blanks

    # Stable profile_id using the raw 'level' column
    df["__level_norm__"] = df[level_col].astype(str).str.strip().str.lower()
    df["profile_id"] = (
        df["ID"].astype(str).str.strip() + "-" + df["__level_norm__"].str[0] +
        (df.groupby(["ID", "__level_norm__"]).cumcount() + 1).astype(str)
    )
    return df, level_col, factor_col_map
