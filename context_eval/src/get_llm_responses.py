import os
from dotenv import load_dotenv
import pandas as pd
import requests
import time

# Load environment variables from .env file
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

INPUT_CSV = "eval_dataset/stratified_sampled_12_with_contextdiv_modelrot_ctxspread.csv"
OUTPUT_CSV = "eval_dataset/llm_responses.csv"

AI_MODELS_FOR_ALL = [
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro",
    "openai/gpt-5"
]

def assign_used_llm(df, ai_models_for_all=AI_MODELS_FOR_ALL):
    for (topic, theme), group in df.groupby(["topic", "theme"], sort=False):
        all_mask = (group["ai_model"] == "All")
        all_indices = group[all_mask].index.tolist()
        for i, idx in enumerate(all_indices):
            used_model = ai_models_for_all[i % 3]
            df.at[idx, "used_llm"] = used_model
        df.loc[group[~all_mask].index, "used_llm"] = group[~all_mask]["ai_model"]
    return df

def get_model_response(prompt, model, apikey=OPENROUTER_API_KEY):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {apikey}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    if model in ["google/gemini-2.5-pro", "anthropic/claude-sonnet-4"]:
        payload["temperature"] = 1.0
    try:
        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error with model {model}: {e} | Response: {getattr(r, 'text', '')}")
        return f"ERROR: {e}"

def main():
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set in environment or .env!")
    df = pd.read_csv(INPUT_CSV)
    df = assign_used_llm(df)

    # Get already-completed prompt_ids (if any)
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        done_ids = set(pd.read_csv(OUTPUT_CSV)["prompt_id"])
        print(f"Resuming: {len(done_ids)} prompt_ids already done, will skip.")
    else:
        done_ids = set()

    for i, row in df.iterrows():
        pid = row.get("prompt_id", row.get("request_id", i))
        if pid in done_ids:
            print(f"Skipping already completed prompt_id {pid}")
            continue
        prompt = row["final_prompt"]
        model = row["used_llm"]
        print(f"[{i+1}/{len(df)}] Calling model: {model} | theme: {row['theme']} ...")
        response = get_model_response(prompt, model)
        time.sleep(1.5)
        out_row = {
            "prompt_id": pid,
            "topic": row["topic"],
            "theme": row["theme"],
            "vulnerability_profile_level": row["vulnerability_profile_level"],
            "vulnerability_profile": row["vulnerability_profile"],
            "ranking_type": row["ranking_type"],
            "used_llm": model,
            "final_prompt": prompt,
            "model_response": response
        }
        # Append to CSV immediately
        write_header = not os.path.isfile(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
        pd.DataFrame([out_row]).to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False
        )

    print(f"\n✅ Done! All responses saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
