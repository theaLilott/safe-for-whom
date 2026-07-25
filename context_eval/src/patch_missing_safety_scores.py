import os
import pandas as pd
import time
import glob
from dotenv import load_dotenv
from openai import OpenAI

from utils_pipeline import (
    build_apparent_prompt,
    build_contextualized_prompt,
    extract_overall_safety_score,
    call_judge_o3  # Make sure this uses the OpenAI Python client, NOT requests!
)

MAGIC_ROW_COUNT = 621

def get_llm_response_row(prompt_id, used_llm, input_csv):
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Missing {input_csv}")
    llm_df = pd.read_csv(input_csv)
    # Ensure string comparison
    llm_df["prompt_id"] = llm_df["prompt_id"].astype(str).str.strip()
    llm_df["used_llm"] = llm_df["used_llm"].astype(str).str.strip()
    match = llm_df[
        (llm_df['prompt_id'] == str(prompt_id).strip()) &
        (llm_df['used_llm'] == str(used_llm).strip())
    ]
    if match.empty:
        raise ValueError(f"No matching row in {input_csv} for prompt_id={prompt_id}, used_llm={used_llm}")
    return match.iloc[0]

def main():
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set!")
    client = OpenAI(api_key=OPENAI_API_KEY)
    JUDGE_MODEL = "o3"
    TEMPERATURE = 1.0  # Must be 1.0 for OpenAI o3

    score_files = sorted(glob.glob("scores/safety_scores_h*.csv"))

    for SCORES_CSV in score_files:
        print(f"\n--- Patching {SCORES_CSV} ---")
        df = pd.read_csv(SCORES_CSV)
        # Find the correct input file
        base = os.path.basename(SCORES_CSV).replace("safety_scores_", "").replace(".csv", "")
        input_csv = f"output/llm_response_{base}.csv"
        if not os.path.exists(input_csv):
            print(f"  [SKIP] Missing input: {input_csv}")
            continue
        df_in = pd.read_csv(input_csv)
        # Ensure string keys
        for d in [df, df_in]:
            d["prompt_id"] = d["prompt_id"].astype(str).str.strip()
            d["used_llm"] = d["used_llm"].astype(str).str.strip()
        # Merge to get ranking_type
        merged = df.merge(
            df_in[["prompt_id", "used_llm", "ranking_type"]],
            on=["prompt_id", "used_llm"],
            how="left"
        )
        # Robust: find correct ranking_type column name after merge
        if "ranking_type" in merged.columns:
            ranking_col = "ranking_type"
        elif "ranking_type_y" in merged.columns:
            ranking_col = "ranking_type_y"
        else:
            print(f"  [ERROR] After merge, no 'ranking_type' column found in {SCORES_CSV}, skipping.")
            continue

        # Only patch non-AI rows that have NaN in any safety score
        nan_rows = merged[
            (merged[ranking_col] != "AI") &
            (merged[["apparent_safety_score", "contextualized_safety_score"]].isna().any(axis=1))
        ]

        print(f"  Found {len(nan_rows)} non-AI rows with missing safety scores.")

        for idx in nan_rows.index:
            prompt_id = merged.at[idx, "prompt_id"]
            used_llm = merged.at[idx, "used_llm"]
            try:
                row = get_llm_response_row(prompt_id, used_llm, input_csv)
            except Exception as e:
                print(f"    Skipping: {e}")
                continue
            user_prompt = row["final_prompt"]
            ai_response = row["model_response"]
            user_profile = row.get("vulnerability_profile", "")

            # Patch only missing fields
            if pd.isna(merged.at[idx, "apparent_safety_score"]):
                apparent_prompt = build_apparent_prompt(user_prompt, ai_response)
                apparent_response = call_judge_o3(
                    apparent_prompt, model=JUDGE_MODEL, client=client, temperature=TEMPERATURE)
                apparent_score = extract_overall_safety_score(apparent_response)
                df.at[idx, "apparent_safety_score"] = apparent_score
                df.at[idx, "apparent_safety_judge_response"] = apparent_response
                print(f"    [Patched] apparent={apparent_score}")
            if pd.isna(merged.at[idx, "contextualized_safety_score"]):
                contextualized_prompt = build_contextualized_prompt(user_profile, user_prompt, ai_response)
                contextualized_response = call_judge_o3(
                    contextualized_prompt, model=JUDGE_MODEL, client=client, temperature=TEMPERATURE)
                contextualized_score = extract_overall_safety_score(contextualized_response)
                df.at[idx, "contextualized_safety_score"] = contextualized_score
                df.at[idx, "contextualized_safety_judge_response"] = contextualized_response
                print(f"    [Patched] contextualized={contextualized_score}")
            time.sleep(1.2)
        df.to_csv(SCORES_CSV, index=False)
        print(f"  ✅ Patched missing scores in {SCORES_CSV}")

if __name__ == "__main__":
    main()
