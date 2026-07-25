import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from utils_pipeline import (
    build_apparent_prompt,
    build_contextualized_prompt,
    extract_overall_safety_score,
    call_judge_o3
)

CHECK_INTERVAL = 20 * 60  # 20 minutes (in seconds)
MAGIC_ROW_COUNT = 621     # Only non-AI rows!

def load_valid_input(input_csv):
    df = pd.read_csv(input_csv)
    # Only keep rows with non-NaN model_response
    df = df[~df["model_response"].isna()].reset_index(drop=True)
    # Filter out AI ranking type
    if "ranking_type" in df.columns:
        df = df[df["ranking_type"] != "AI"].reset_index(drop=True)
    # Standardize string columns
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["used_llm"] = df["used_llm"].astype(str)
    return df

def load_scores(output_csv):
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        out_df = pd.read_csv(output_csv)
        out_df["prompt_id"] = out_df["prompt_id"].astype(str)
        out_df["used_llm"] = out_df["used_llm"].astype(str)
        # Filter out AI ranking type (may already exist from earlier runs)
        if "ranking_type" in out_df.columns:
            out_df = out_df[out_df["ranking_type"] != "AI"].reset_index(drop=True)
    else:
        out_df = pd.DataFrame(columns=[
            "prompt_id", "used_llm", "apparent_safety_score", "contextualized_safety_score",
            "apparent_safety_judge_response", "contextualized_safety_judge_response"
        ])
    return out_df

def jobs_to_score(df, out_df):
    # Map (prompt_id, used_llm) => row index in output
    done_dict = {(row["prompt_id"], row["used_llm"]): i for i, row in out_df.iterrows()}
    jobs = []
    for idx, row in df.iterrows():
        key = (row["prompt_id"], row["used_llm"])
        if key not in done_dict:
            jobs.append((idx, row))
        else:
            i = done_dict[key]
            appr_score = out_df.loc[i, "apparent_safety_score"]
            ctx_score = out_df.loc[i, "contextualized_safety_score"]
            if pd.isna(appr_score) or pd.isna(ctx_score):
                jobs.append((idx, row))
    return jobs

def main():
    if len(sys.argv) < 2:
        print("Usage: python score_pipeline.py <request_id>", flush=True)
        sys.exit(1)
    request_id = sys.argv[1]
    INPUT_CSV = f"output/llm_response_{request_id}.csv"
    OUTPUT_CSV = f"scores/safety_scores_{request_id}.csv"
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set!")
    client = OpenAI(api_key=OPENAI_API_KEY)
    JUDGE_MODEL = "o3"    # Or "gpt-4o" if you want
    TEMPERATURE = 1.0              # For gpt-4o-mini, only 1.0 is allowed

    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Reloading data...", flush=True)
        df = load_valid_input(INPUT_CSV)
        out_df = load_scores(OUTPUT_CSV)
        total = len(df)
        print(f"Loaded {total} valid non-AI rows from {INPUT_CSV}", flush=True)
        print(f"Output currently has {len(out_df)} non-AI rows.", flush=True)

        jobs = jobs_to_score(df, out_df)
        print(f"Need to score {len(jobs)} rows this round.", flush=True)

        for job_idx, (idx, row) in enumerate(jobs, 1):
            prompt_id = row["prompt_id"]
            used_llm = row["used_llm"]
            user_prompt = row["final_prompt"]
            ai_response = row["model_response"]
            user_profile = row["vulnerability_profile"]

            apparent_prompt = build_apparent_prompt(user_prompt, ai_response)
            contextualized_prompt = build_contextualized_prompt(user_profile, user_prompt, ai_response)

            print(f"[{job_idx}/{len(jobs)}][{idx+1}/{total}] Judging prompt_id={prompt_id}, used_llm={used_llm}...", flush=True)
            apparent_response = call_judge_o3(
                apparent_prompt, model=JUDGE_MODEL, client=client, temperature=TEMPERATURE)
            contextualized_response = call_judge_o3(
                contextualized_prompt, model=JUDGE_MODEL, client=client, temperature=TEMPERATURE)

            apparent_score = extract_overall_safety_score(apparent_response)
            contextualized_score = extract_overall_safety_score(contextualized_response)

            print(f"    => apparent={apparent_score}, contextualized={contextualized_score}", flush=True)

            # Write/update output
            output_row = {
                "prompt_id": prompt_id,
                "used_llm": used_llm,
                "apparent_safety_score": apparent_score,
                "contextualized_safety_score": contextualized_score,
                "apparent_safety_judge_response": apparent_response,
                "contextualized_safety_judge_response": contextualized_response,
            }
            # If ranking_type is present, preserve it (for writing back)
            if "ranking_type" in row:
                output_row["ranking_type"] = row["ranking_type"]
            # Update or append in out_df
            key = (prompt_id, used_llm)
            match = ((out_df["prompt_id"] == prompt_id) & (out_df["used_llm"] == used_llm))
            if match.any():
                idx_match = out_df[match].index[0]
                for k, v in output_row.items():
                    out_df.loc[idx_match, k] = v
            else:
                out_df = pd.concat([out_df, pd.DataFrame([output_row])], ignore_index=True)

            # Save after each row
            out_df.to_csv(OUTPUT_CSV, index=False)

            time.sleep(1.2)

        print(f"[Status] Output file now has {len(out_df)} non-AI rows.", flush=True)
        if len(out_df) >= MAGIC_ROW_COUNT:
            print(f"\n✅ All {MAGIC_ROW_COUNT} safety scores saved to {OUTPUT_CSV}", flush=True)
            break

        # After sweep, check for new or NaN rows before sleeping
        print("Rechecking for new or NaN rows before sleep...", flush=True)
        df = load_valid_input(INPUT_CSV)
        out_df = load_scores(OUTPUT_CSV)
        jobs = jobs_to_score(df, out_df)
        if len(jobs) == 0:
            print(f"Nothing new to score. Sleeping for {CHECK_INTERVAL // 60} minutes...", flush=True)
            time.sleep(CHECK_INTERVAL)
        else:
            print(f"More work detected (need to score {len(jobs)} rows). Continuing immediately.", flush=True)

if __name__ == "__main__":
    main()
