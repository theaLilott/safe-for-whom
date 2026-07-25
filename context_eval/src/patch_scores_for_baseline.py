# patch_incomplete_baseline_scores.py

import os
import glob
import pandas as pd
from dotenv import load_dotenv

from utils_pipeline import (
    build_apparent_prompt,
    build_contextualized_prompt,
    extract_overall_safety_score,
    call_judge
)

EXPECTED = 9
JUDGE_MODEL = "openai/o3"
TEMPERATURE = 0.2

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set!")

score_files = sorted(glob.glob("scores/safety_scores_h*.csv"))

for score_file in score_files:
    request_id = score_file.split('_')[-1].replace('.csv', '')  # e.g., h00
    llm_response_file = f"output/llm_response_{request_id}.csv"
    if not os.path.exists(llm_response_file):
        print(f"Missing {llm_response_file}, skipping {score_file}")
        continue

    print(f"\n--- Checking {score_file} ---")
    scores = pd.read_csv(score_file)
    responses = pd.read_csv(llm_response_file)

    # Merge just to get baseline prompt info for those missing scores
    merged = pd.merge(
        scores,
        responses[["prompt_id", "used_llm", "ranking_type", "final_prompt", "model_response", "vulnerability_profile"]],
        on=["prompt_id", "used_llm"],
        how="left"
    )

    # Find baseline rows with missing safety scores
    mask = (
        (merged["ranking_type"].str.lower() == "baseline") &
        (merged["apparent_safety_score"].isna() | merged["contextualized_safety_score"].isna())
    )
    to_patch = merged[mask]

    if to_patch.empty:
        print(f"All baseline rows complete for {request_id}.")
        continue

    print(f"Need to patch {len(to_patch)} baseline rows in {score_file}")

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

print("\n✅ Patch complete! All incomplete baseline rows processed.")
