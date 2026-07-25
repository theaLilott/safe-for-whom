import os, time, logging
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime
from openai import OpenAI

from src.utils.constants import FACTORS, ID_TO_TOPIC_THEME, CLAUSE_NORMALIZER_PROMPT
from src.utils.profile_utils import load_profiles_from_raw  

# ---------- Paths ----------
DATA_DIR    = "eval_dataset"
PROF_XLSX   = os.path.join(DATA_DIR, "vulnerability_profiles_raw.xlsx")
CLAUSE_CSV  = os.path.join(DATA_DIR, "clause_cache.csv")
USAGE_CSV   = os.path.join(DATA_DIR, "clause_cache_usage.csv")

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------- OpenAI ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY.")
client = OpenAI(api_key=OPENAI_API_KEY)
CLAUSE_MODEL = "gpt-4.1-nano"

# ---------- clause cache ----------
CLAUSE_CACHE: Dict[Tuple[str, str], str] = {}

# Token usage accumulator (successful calls only)
USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "successful_api_calls": 0,
}

def accumulate_usage(resp) -> None:
    try:
        u = getattr(resp, "usage", None) or {}
        pt = int(getattr(u, "prompt_tokens", 0) or u.get("prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or u.get("completion_tokens", 0) or 0)
        tt = int(getattr(u, "total_tokens", 0) or u.get("total_tokens", 0) or (pt + ct))
        USAGE["prompt_tokens"] += pt
        USAGE["completion_tokens"] += ct
        USAGE["total_tokens"] += tt
        USAGE["successful_api_calls"] += 1
    except Exception:
        pass

def sanitize_value(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()

def profile_lookup(prof: dict, factor: str, factor_col_map: Dict[str, str]) -> str:
    col = factor_col_map.get(factor, factor)
    if col in prof and pd.notna(prof[col]):
        return sanitize_value(prof[col])
    return ""

def normalize_clause(factor: str, value: str) -> str:
    """(factor, value) -> minimal first-person clause (no trailing period)."""
    key = (factor, value)
    # cache check 
    if key in CLAUSE_CACHE:
        return CLAUSE_CACHE[key]

    prompt = CLAUSE_NORMALIZER_PROMPT.format(factor=factor, value=value)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=CLAUSE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            accumulate_usage(resp)
            text = (resp.choices[0].message.content or "").strip().rstrip(".")
            CLAUSE_CACHE[key] = text
            return text
        except Exception as e:
            logging.warning(f"[{CLAUSE_MODEL}] normalize error {factor}={value} (attempt {attempt+1}/3): {e}")
            time.sleep(1.0 * (attempt + 1))

    # Fallback if API keeps failing
    fallback = f"I have {factor.lower()}" + (f" = {value}" if value else "")
    CLAUSE_CACHE[key] = fallback
    return fallback

def append_usage_row(usage_row: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    row_df = pd.DataFrame([usage_row])
    if not os.path.exists(USAGE_CSV):
        row_df.to_csv(USAGE_CSV, index=False)
    else:
        row_df.to_csv(USAGE_CSV, mode="a", index=False, header=False)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 0) Optional smoke test (not counted in usage)
    try:
        _ = client.chat.completions.create(
            model=CLAUSE_MODEL,
            messages=[{"role": "user", "content": "ok"}],
            temperature=0.0,
        )
        logging.info("OpenAI smoke test OK.")
    except Exception as e:
        logging.warning(f"Smoke test failed (continuing anyway): {e}")

    # 1) Load profiles with the shared loader/mapping
    profiles_df, level_col, factor_col_map = load_profiles_from_raw(
        PROF_XLSX, FACTORS, ID_TO_TOPIC_THEME
    )

    # 2) Collect all unique (factor, value) pairs across every profile/theme
    pairs = set()
    for _, row in profiles_df.iterrows():
        prof = row.to_dict()
        for factor in FACTORS:
            value = profile_lookup(prof, factor, factor_col_map)
            if value:
                pairs.add((factor, value))
    pairs_total = len(pairs)
    logging.info(f"Discovered {pairs_total} unique (factor, value) pairs in profiles.")

    # 3) Normalize every pair 
    for i, (factor, value) in enumerate(sorted(pairs), start=1):
        if i % 50 == 0:
            logging.info(f"...normalized {i}/{pairs_total}")
        _ = normalize_clause(factor, value)

    # 4) Write final CSV 
    rows = [{"factor": f, "value": v, "clause": CLAUSE_CACHE[(f, v)]} for (f, v) in sorted(CLAUSE_CACHE.keys())]
    out_df = pd.DataFrame(rows, columns=["factor", "value", "clause"]).drop_duplicates(subset=["factor", "value"], keep="first")
    out_df.to_csv(CLAUSE_CSV, index=False)
    logging.info(f"Wrote {len(out_df)} unique (factor, value) clauses to {CLAUSE_CSV}")

    # 5) Add usage summary csv
    usage_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": CLAUSE_MODEL,
        "pairs_total_seen": pairs_total,
        "pairs_normalized": pairs_total,              # all are normalized this run
        "successful_api_calls": USAGE["successful_api_calls"],
        "prompt_tokens": USAGE["prompt_tokens"],
        "completion_tokens": USAGE["completion_tokens"],
        "total_tokens": USAGE["total_tokens"],
    }
    append_usage_row(usage_row)
    logging.info(f"Token usage saved to {USAGE_CSV}: {usage_row}")

if __name__ == "__main__":
    main()
