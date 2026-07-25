import os, json, time, logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

from src.utils.constants import (
    VARIANT_BUILDER_PROMPT_L3, VARIANT_BUILDER_PROMPT_L5,
)
# ------- Auth / client -------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY.")
client = OpenAI(api_key=OPENAI_API_KEY)
VARIANT_MODEL = "gpt-4o-mini"

# --- Usage tracking for variant calls ---
DATA_DIR  = "eval_dataset"
VARIANT_USAGE_CSV = os.path.join(DATA_DIR, "variant_usage.csv")
VARIANT_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "successful_api_calls": 0,
}

def accumulate_variant_usage(resp) -> None:
    try:
        u = getattr(resp, "usage", None) or {}
        pt = int(getattr(u, "prompt_tokens", 0) or u.get("prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or u.get("completion_tokens", 0) or 0)
        tt = int(getattr(u, "total_tokens", 0) or u.get("total_tokens", 0) or (pt + ct))
        VARIANT_USAGE["prompt_tokens"] += pt
        VARIANT_USAGE["completion_tokens"] += ct
        VARIANT_USAGE["total_tokens"] += tt
        VARIANT_USAGE["successful_api_calls"] += 1
    except Exception:
        pass

def append_variant_usage_row(extra: dict):
    import pandas as pd
    from datetime import datetime
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": VARIANT_MODEL,
        "successful_api_calls": VARIANT_USAGE["successful_api_calls"],
        "prompt_tokens": VARIANT_USAGE["prompt_tokens"],
        "completion_tokens": VARIANT_USAGE["completion_tokens"],
        "total_tokens": VARIANT_USAGE["total_tokens"],
    }
    row.update(extra or {})
    df = pd.DataFrame([row])
    if not os.path.exists(VARIANT_USAGE_CSV):
        df.to_csv(VARIANT_USAGE_CSV, index=False)
    else:
        df.to_csv(VARIANT_USAGE_CSV, mode="a", index=False, header=False)

def parse_variant_response(raw: str, expect_key: str) -> List[Dict[str, object]]:
    data = json.loads(raw)
    out = data.get(expect_key, data) if isinstance(data, dict) else data
    if not isinstance(out, list):
        raise ValueError("Invalid JSON shape for variants")

    norm: List[Dict[str, object]] = []
    for item in out:
        if isinstance(item, dict) and "clauses" in item:
            clauses = item.get("clauses", [])
            sent = item.get("context_text", "")
            if isinstance(clauses, list) and all(isinstance(x, str) for x in clauses):
                norm.append({"clauses": clauses, "context_text": str(sent or "").strip()})
        elif isinstance(item, list) and all(isinstance(x, str) for x in item):
            norm.append({"clauses": item, "context_text": ""})
    return norm

def build_level3_variants_with_llm(theme, ranking_key, top3, factor_to_clause):
    clauses_json = json.dumps(
    {"clauses_to_use": [factor_to_clause[f] for f in top3 if factor_to_clause[f].strip()]},
    ensure_ascii=False
    )
    prompt = f"{VARIANT_BUILDER_PROMPT_L3}\n\nClauses to use (JSON): {clauses_json}"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=VARIANT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            accumulate_variant_usage(resp)
            raw = (resp.choices[0].message.content or "").strip()
            logging.info(f"[VARIANT L3][{theme}][{ranking_key}] LLM raw output:\n{raw}")
            return {"level3": parse_variant_response(raw, expect_key="level3")}
        except Exception as e:
            logging.warning(f"[{VARIANT_MODEL}] L3 variant builder error (attempt {attempt+1}/3): {e}")
            time.sleep(1.5 * (attempt + 1))
    # Hard fallback for L3 
    from itertools import permutations
    c = [factor_to_clause[f] for f in top3]
    unique_orders = list(permutations(c, 3))
    fallback = [
        {"clauses": list(order), "context_text": ""}
        for order in unique_orders[:5]
    ]
    return {"level3": fallback}

def build_level5_variants_with_llm(
    theme: str,
    ranking_key: str,
    top5: List[str],
    factor_to_clause: Dict[str, str],
) -> Dict[str, List[Dict[str, object]]]:
    clauses_json = json.dumps(
    {"clauses_to_use": [factor_to_clause[f] for f in top5 if factor_to_clause[f].strip()]},
    ensure_ascii=False
    )
    prompt = f"{VARIANT_BUILDER_PROMPT_L5}\n\nClauses to use (JSON): {clauses_json}"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=VARIANT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            accumulate_variant_usage(resp)
            raw = (resp.choices[0].message.content or "").strip()
            logging.info(f"[VARIANT L5][{theme}][{ranking_key}] LLM raw output:\n{raw}")
            return {"level5": parse_variant_response(raw, expect_key="level5")}
        except Exception as e:
            logging.warning(f"[{VARIANT_MODEL}] L5 variant builder error (attempt {attempt+1}/3): {e}")
            time.sleep(1.5 * (attempt + 1))
    # Hard fallback for L5 
    c = [factor_to_clause[f] for f in top5]
    fallback = [
        {"clauses": c, "context_text": ""},
        {"clauses": c[::-1], "context_text": ""},
        {"clauses": [c[1], c[3], c[0], c[4], c[2]], "context_text": ""},
        {"clauses": [c[2], c[0], c[4], c[1], c[3]], "context_text": ""},
        {"clauses": [c[3], c[4], c[1], c[2], c[0]], "context_text": ""},
    ]
    return {"level5": fallback}
