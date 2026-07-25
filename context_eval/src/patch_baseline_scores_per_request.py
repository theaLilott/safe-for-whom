# patch_baseline_scores_per_request.py

import os
import sys
import pandas as pd
from dotenv import load_dotenv

from utils_pipeline import (
    build_apparent_prompt,
    build_contextualized_prompt,
    extract_overall_safety_score,
    call_judge
)

EXPECTED = 9  # Baseline prompts per model
JUDGE_MODEL = "openai/o3"
TEMPERATURE = 0.2

def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_baseline_scores_per_request.py <request_id>   # e.g., f00")
        sys.exit(1)

    request_id = sys.argv[1]
    response_file = f"output/llm_response_{request_id}.csv"
    score_file = f"scores/safety_scores_{request_id}.csv"

    if not os.path.exists(response_file):
        print(f"Missing {response_file}, skipping.")
        sys.exit(0)

    load_dotenv()
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set!")

    print(f"\n--- Checking {score_file} ---")
    responses = pd.read_csv(response_file)

    # Only keep Baseline rows
    baseline_rows = responses[responses["ranking_type"].str.lower() == "baseline"]

    # If score file missing, create empty one with all baseline rows (no scores yet)
    if not os.path.exists(score_file):
        print(f"{score_file} not found. Creating new baseline score file...")
        scores = baseline_rows[["prompt_id", "used_llm"]].copy()
        scores["apparent_safety_score"] = None
        scores["contextualized_safety_score"] = None
        scores["apparent_safety_judge_response"] = None
        scores["contextualized_safety_judge_response"] = None
        scores.to_csv(score_file, index=False)
    else:
        scores = pd.read_csv(score_file)

    # Merge to get Baseline prompt info for those missing scores
    merged = pd.merge(
        scores,
        baseline_rows[["prompt_id", "used_llm", "final_prompt", "model_response", "vulnerability_profile"]],
        on=["prompt_id", "used_llm"],
        how="left"
    )

    # Find Baseline rows with missing safety scores
    mask = (
        merged["apparent_safety_score"].isna() | merged["contextualized_safety_score"].isna()
    )
    to_patch = merged[mask]

    if to_patch.empty:
        print(f"All Baseline rows complete for {request_id}.")
        return

    print(f"Need to patch {len(to_patch)} Baseline rows in {score_file}")

    for i, row in to_patch.iterrows():
        prompt_id = row["prompt_id"]
        used_llm = row["used_llm"]
        user_prompt = row["final_prompt"]
        ai_response = row["model_response"]
        user_profile = row["vulnerability_profile"]

        print(f"  [Patch] prompt_id={prompt_id}, used_llm={used_llm}")

        # Build judge prompts
        apparent_prompt = build_apparent_prompt(user_prompt, ai_response)
        contextualized_prompt = build_contextualized_prompt(user_profile, user_prompt, ai_response)

        # Call judge model for both scores
        apparent_response = call_judge(apparent_prompt, model=JUDGE_MODEL, apikey=OPENROUTER_API_KEY, temperature=TEMPERATURE)
        contextualized_response = call_judge(contextualized_prompt, model=JUDGE_MODEL, apikey=OPENROUTER_API_KEY, temperature=TEMPERATURE)
        apparent_score = extract_overall_safety_score(apparent_response)
        contextualized_score = extract_overall_safety_score(contextualized_response)

        print(f"    => apparent={apparent_score}, contextualized={contextualized_score}")

        # Patch in the scores directly in the scores DataFrame (not merged, use original index)
        score_idx = scores[(scores["prompt_id"] == prompt_id) & (scores["used_llm"] == used_llm)].index
        if len(score_idx) != 1:
            print(f"    [Warn] Found {len(score_idx)} score rows for prompt_id={prompt_id}, used_llm={used_llm}, skipping!")
            continue
        idx = score_idx[0]
        scores.loc[idx, "apparent_safety_score"] = apparent_score
        scores.loc[idx, "contextualized_safety_score"] = contextualized_score
        scores.loc[idx, "apparent_safety_judge_response"] = apparent_response
        scores.loc[idx, "contextualized_safety_judge_response"] = contextualized_response

        # Save after each patch (to be safe)
        scores.to_csv(score_file, index=False)

    print(f"\n✅ Patch complete! All incomplete Baseline rows processed for {request_id}.")

if __name__ == "__main__":
    main()
