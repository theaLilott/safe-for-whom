import os, json, time, logging
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from src.utils.constants import (
    FACTORS, ID_TO_TOPIC_THEME, OUTPUT_COLUMNS,
)
from src.utils.building_variants import (
    build_level3_variants_with_llm, build_level5_variants_with_llm, append_variant_usage_row, VARIANT_USAGE, VARIANT_USAGE_CSV,
)
from src.utils.profile_utils import load_profiles_from_raw

# ---------------- Paths ----------------
DATA_DIR  = "eval_dataset"
PLAIN_PATH = os.path.join(DATA_DIR, "eval_dataset.csv")
REL_PATH   = os.path.join(DATA_DIR, "relevance_rankings.csv")
LIK_PATH   = os.path.join(DATA_DIR, "likelihood_rankings.csv")
AI_PATH    = os.path.join(DATA_DIR, "ai_rankings.csv")
PROF_XLSX  = os.path.join(DATA_DIR, "vulnerability_profiles_raw.xlsx")
OUT_PATH   = os.path.join(DATA_DIR, "evaluation_prompts.csv")

# clause cache
CLAUSE_CSV = os.path.join(DATA_DIR, "clause_cache.csv")

# ------- Logging -------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"context_eval_{run_id}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# ------- Globals resolved via shared loader -------
LEVEL_COL: Optional[str] = None
FACTOR_COL_MAP: Dict[str, str] = {}

def sanitize_value(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()

def profile_lookup(prof: dict, factor: str) -> str:
    col = FACTOR_COL_MAP.get(factor, factor)
    if col in prof and pd.notna(prof[col]):
        return sanitize_value(prof[col])
    return ""

def join_clauses(clauses: List[str]) -> str:
    if not clauses:
        return ""
    if len(clauses) == 1:
        return f"{clauses[0]}."
    return (", ".join(clauses[:-1]) + f", and {clauses[-1]}.")

# --------- Stage 1:CLAUSE LOOKUP from cache CSV ----------
if not os.path.exists(CLAUSE_CSV):
    raise FileNotFoundError(
        f"Missing clause cache at {CLAUSE_CSV}. Run generate_clauses.py first."
    )
_clause_df = pd.read_csv(CLAUSE_CSV)
CLAUSE_LOOKUP = {
    (row["factor"].strip(), str(row["value"]).strip()): row["clause"]
    for _, row in _clause_df.iterrows()
}

def get_clause(factor, value):
    k = (str(factor).strip(), str(value).strip())
    if k not in CLAUSE_LOOKUP:
        logging.warning(f"Missing clause for: {k} (raw value: {repr(value)})")
    else:
        logging.debug(f"Clause found for: {k}")
    return CLAUSE_LOOKUP.get(k, f"I have {factor.lower()} = {value}")

# ---------------- Rankings loaders ----------------
def load_single_ranking_map(df: pd.DataFrame, kind: str) -> Dict[str, Dict]:
    out = {}
    needed = {"id","topic","theme","1","2","3","4","5"}
    if df is None or df.empty or not set(needed).issubset(df.columns):
        raise ValueError(f"{kind} rankings CSV is missing or malformed. Need columns {sorted(needed)}.")
    for _, r in df.iterrows():
        out[str(r["theme"])] = {
            "id": r["id"], "topic": r["topic"], "theme": r["theme"],
            "1": r["1"], "2": r["2"], "3": r["3"], "4": r["4"], "5": r["5"],
        }
    return out

# ---------------- Main ----------------
def main():
    # Load core data
    plain = pd.read_csv(PLAIN_PATH)
    ai_df  = pd.read_csv(AI_PATH)
    rel_df = pd.read_csv(REL_PATH)
    lik_df = pd.read_csv(LIK_PATH)

    # Use the SAME loader/mapping as in generate_clauses.py
    profiles_df, level_col, factor_col_map = load_profiles_from_raw(
        PROF_XLSX, FACTORS, ID_TO_TOPIC_THEME
    )
    global LEVEL_COL, FACTOR_COL_MAP
    LEVEL_COL = level_col
    FACTOR_COL_MAP = factor_col_map

    # Build ranking maps
    rel_map = load_single_ranking_map(rel_df, "Relevance")
    lik_map = load_single_ranking_map(lik_df, "Likelihood")

    # AI rankings
    ai_by_model: Dict[str, Dict[str, Dict]] = {}
    models = ai_df["model"].dropna().unique().tolist()
    for m in models:
        ai_by_model[m] = {}
        for _, r in ai_df[ai_df["model"] == m].iterrows():
            ai_by_model[m][str(r["theme"])] = {
                "id": r["id"], "topic": r["topic"], "theme": r["theme"],
                "1": r["1"], "2": r["2"], "3": r["3"], "4": r["4"], "5": r["5"],
            }

    # Group profiles by theme
    profs_by_theme: Dict[str, List[Dict]] = {}
    for _, r in profiles_df.iterrows():
        profs_by_theme.setdefault(r["theme"], []).append(r.to_dict())

    rows = []
    VARIANT_CACHE: Dict[Tuple[str, str, str, str], Dict[str, List[Dict[str, object]]]] = {}
    DROP_PROFILE_COLS = {
    "ID", "vulnerability", "Column1", "Column2", "Column3", "Column4",
    "Column5", "Column6", "topic", "theme", "__level_norm__", "profile_id"
    }

    # ---- Level 0 
    for _, base in plain.iterrows():
        request_id, topic, theme, question = base["id"], base["topic"], base["theme"], base["question"]
        theme_profiles = profs_by_theme.get(theme, [])
        if not theme_profiles:
            logging.warning(f"No profiles found for theme '{theme}'.")
            continue
        for prof in theme_profiles:
            level_val = str(prof.get(LEVEL_COL, "")).strip().lower()
            prof_uid  = prof.get("profile_id") or prof.get("ID")
            vprof = {k: v for k, v in prof.items() if k not in DROP_PROFILE_COLS}
            prof_json = json.dumps(vprof, ensure_ascii=False)
            pid0 = f"{request_id}_none_{level_val}_{prof_uid}_0_v0"
            rows.append({
                "prompt_id": pid0,
                "request_id": request_id,
                "topic": topic,
                "theme": theme,
                "vulnerability_profile_level": level_val,
                "vulnerability_profile": prof_json,
                "ranking_type": "Baseline",
                "ai_model": "All",
                "context_level": 0,
                "context_variant": 0,
                "context_factors_used": "[]",
                "final_prompt": question.strip(),
            })

    # ---- Contexted prompts
    for _, base in plain.iterrows():
        request_id, topic, theme, question = base["id"], base["topic"], base["theme"], base["question"]

        # common rankings for this theme (Relevance & Likelihood)
        if theme not in rel_map or theme not in lik_map:
            logging.warning(f"Missing Relevance/Likelihood ranking for theme='{theme}'. Skipping those.")
            common_rankings: List[Tuple[str, Dict]] = []
        else:
            common_rankings = [("Relevance", rel_map[theme]), ("Likelihood", lik_map[theme])]

        theme_profiles = profs_by_theme.get(theme, [])
        if not theme_profiles:
            logging.warning(f"No profiles found for theme '{theme}'.")
            continue

        # --- Relevance & Likelihood 
        for kind, rk in common_rankings:
            top5 = [rk["1"], rk["2"], rk["3"], rk["4"], rk["5"]]
            for prof in theme_profiles:
                level_val = str(prof.get(LEVEL_COL, "")).strip().lower()
                prof_uid  = prof.get("profile_id") or prof.get("ID")
                logging.info(f"--- Processing Theme(Relevance & Likelihood): '{theme}', Ranking: '{kind}', Profile: '{prof_uid}', Top5 Factors: {top5}")
                vprof = {k: v for k, v in prof.items() if k not in DROP_PROFILE_COLS}
                prof_json = json.dumps(vprof, ensure_ascii=False)

                # Build clauses via cache
                for f in top5:
                    raw_val = prof.get(f, "")
                    norm_val = profile_lookup(prof, f)
                    logging.debug(f"   Factor '{f}': Raw='{raw_val}' | Normalized='{norm_val}'")
                factor_to_clause = {f: get_clause(f, profile_lookup(prof, f)) for f in top5}
                logging.info(f"   factor_to_clause: {factor_to_clause}")

                # L1
                l1_clause = factor_to_clause[top5[0]].rstrip(".")
                final_prompt = f"{l1_clause}. {question}".strip()
                pid = f"{request_id}_{kind.lower()}_{level_val}_{prof_uid}_1_v0"
                rows.append({
                    "prompt_id": pid,
                    "request_id": request_id,
                    "topic": topic,
                    "theme": theme,
                    "vulnerability_profile_level": level_val,
                    "vulnerability_profile": prof_json,
                    "ranking_type": kind,
                    "ai_model": "All",
                    "context_level": 1,
                    "context_variant": 0,
                    "context_factors_used": json.dumps([top5[0]], ensure_ascii=False),
                    "final_prompt": final_prompt,
                })

                # L3/L5 
                rank_key = f"{kind}:NA:{tuple(top5)}"

                # L3
                cache_key_l3 = ("L3", theme, rank_key, prof_uid)
                if cache_key_l3 not in VARIANT_CACHE:
                    VARIANT_CACHE[cache_key_l3] = build_level3_variants_with_llm(theme, rank_key, top5[:3], factor_to_clause)
                varset_l3 = VARIANT_CACHE[cache_key_l3]
                lvl3_items = varset_l3.get("level3", [])

                for vi, item in enumerate(lvl3_items, start=1):
                    context_text = item["context_text"] or join_clauses([c.rstrip(".") for c in item["clauses"]])
                    pid = f"{request_id}_{kind.lower()}_{level_val}_{prof_uid}_3_v{vi}"
                    rows.append({
                        "prompt_id": pid,
                        "request_id": request_id,
                        "topic": topic,
                        "theme": theme,
                        "vulnerability_profile_level": level_val,
                        "vulnerability_profile": prof_json,
                        "ranking_type": kind,
                        "ai_model": "All",
                        "context_level": 3,
                        "context_variant": vi,
                        "context_factors_used": json.dumps(top5[:3], ensure_ascii=False),
                        "final_prompt": f"{context_text} {question}".strip(),
                    })

                cache_key_l5 = ("L5", theme, rank_key, prof_uid)
                if cache_key_l5 not in VARIANT_CACHE:
                    VARIANT_CACHE[cache_key_l5] = build_level5_variants_with_llm(theme, rank_key, top5, factor_to_clause)
                varset_l5 = VARIANT_CACHE[cache_key_l5]
                lvl5_items = varset_l5.get("level5", [])

                for vi, item in enumerate(lvl5_items, start=1):
                    context_text = item["context_text"] or join_clauses([c.rstrip(".") for c in item["clauses"]])
                    pid = f"{request_id}_{kind.lower()}_{level_val}_{prof_uid}_5_v{vi}"
                    rows.append({
                        "prompt_id": pid,
                        "request_id": request_id,
                        "topic": topic,
                        "theme": theme,
                        "vulnerability_profile_level": level_val,
                        "vulnerability_profile": prof_json,
                        "ranking_type": kind,
                        "ai_model": "All",
                        "context_level": 5,
                        "context_variant": vi,
                        "context_factors_used": json.dumps(top5, ensure_ascii=False),
                        "final_prompt": f"{context_text} {question}".strip(),
                    })

        # --- AI (per model)
        for model, theme_map in ai_by_model.items():
            if theme not in theme_map:
                logging.warning(f"Missing AI ranking for theme='{theme}' (model={model}). Skipping.")
                continue
            rk = theme_map[theme]
            top5 = [rk["1"], rk["2"], rk["3"], rk["4"], rk["5"]]

            for prof in theme_profiles:
                level_val = str(prof.get(LEVEL_COL, "")).strip().lower()
                prof_uid  = prof.get("profile_id") or prof.get("ID")
                logging.info(f"--- Processing Theme(AI): '{theme}', Model: '{model}', Profile: '{prof_uid}', Top5 Factors: {top5}")
                vprof = {k: v for k, v in prof.items() if k not in DROP_PROFILE_COLS}
                prof_json = json.dumps(vprof, ensure_ascii=False)

                for f in top5:
                    raw_val = prof.get(f, "")
                    norm_val = profile_lookup(prof, f)
                    logging.debug(f"   Factor '{f}': Raw='{raw_val}' | Normalized='{norm_val}'")

                factor_to_clause = {f: get_clause(f, profile_lookup(prof, f)) for f in top5}

                logging.info(f"   factor_to_clause: {factor_to_clause}")

                # L1
                l1_clause = factor_to_clause[top5[0]].rstrip(".")
                pid = f"{request_id}_ai-{model}_{level_val}_{prof_uid}_1_v0"
                rows.append({
                    "prompt_id": pid,
                    "request_id": request_id,
                    "topic": topic,
                    "theme": theme,
                    "vulnerability_profile_level": level_val,
                    "vulnerability_profile": prof_json,
                    "ranking_type": "AI",
                    "ai_model": model,
                    "context_level": 1,
                    "context_variant": 0,
                    "context_factors_used": json.dumps([top5[0]], ensure_ascii=False),
                    "final_prompt": f"{l1_clause}. {question}".strip(),
                })

                # L3/L5
                rank_key = f"AI:{model}:{tuple(top5)}"

                # L3
                cache_key_l3 = ("L3", theme, rank_key, prof_uid)
                if cache_key_l3 not in VARIANT_CACHE:
                    VARIANT_CACHE[cache_key_l3] = build_level3_variants_with_llm(theme, rank_key, top5[:3], factor_to_clause)
                varset_l3 = VARIANT_CACHE[cache_key_l3]
                lvl3_items = varset_l3.get("level3", [])

                for vi, item in enumerate(lvl3_items, start=1):
                    context_text = item["context_text"] or join_clauses([c.rstrip(".") for c in item["clauses"]])
                    pid = f"{request_id}_ai-{model}_{level_val}_{prof_uid}_3_v{vi}"
                    rows.append({
                        "prompt_id": pid,
                        "request_id": request_id,
                        "topic": topic,
                        "theme": theme,
                        "vulnerability_profile_level": level_val,
                        "vulnerability_profile": prof_json,
                        "ranking_type": "AI",
                        "ai_model": model,
                        "context_level": 3,
                        "context_variant": vi,
                        "context_factors_used": json.dumps(top5[:3], ensure_ascii=False),
                        "final_prompt": f"{context_text} {question}".strip(),
                    })

                # L5
                cache_key_l5 = ("L5", theme, rank_key, prof_uid)
                if cache_key_l5 not in VARIANT_CACHE:
                    VARIANT_CACHE[cache_key_l5] = build_level5_variants_with_llm(theme, rank_key, top5, factor_to_clause)
                varset_l5 = VARIANT_CACHE[cache_key_l5]
                lvl5_items = varset_l5.get("level5", [])
                for vi, item in enumerate(lvl5_items, start=1):
                    context_text = item["context_text"] or join_clauses([c.rstrip(".") for c in item["clauses"]])
                    pid = f"{request_id}_ai-{model}_{level_val}_{prof_uid}_5_v{vi}"
                    rows.append({
                        "prompt_id": pid,
                        "request_id": request_id,
                        "topic": topic,
                        "theme": theme,
                        "vulnerability_profile_level": level_val,
                        "vulnerability_profile": prof_json,
                        "ranking_type": "AI",
                        "ai_model": model,
                        "context_level": 5,
                        "context_variant": vi,
                        "context_factors_used": json.dumps(top5, ensure_ascii=False),
                        "final_prompt": f"{context_text} {question}".strip(),
                    })

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Sanity logs
    n_questions = len(plain)
    n_models = len(models)
    logging.info(f"Themes: {plain['theme'].nunique()} | Questions: {n_questions} | Models: {n_models}")
    logging.info(f"Profiles total: {sum(len(v) for v in (profs_by_theme.values()))}")
    logging.info(f"Rows built: {len(out_df)}  (expected ≈35,712 if inputs complete)")

    out_df.to_csv(OUT_PATH, index=False)
    logging.info(f"Wrote {len(out_df)} prompts to {OUT_PATH}")

    # Add usage summary 
    extra = {
        "themes": plain["theme"].nunique(),
        "questions": len(plain),
        "models": len(models),
        "variant_sets_built": len(VARIANT_CACHE),
    }
    append_variant_usage_row(extra)
    logging.info(f"Variant token usage saved to {VARIANT_USAGE_CSV}: {VARIANT_USAGE} | extra={extra}")

if __name__ == "__main__":
    main()
