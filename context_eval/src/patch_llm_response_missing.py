import os
from dotenv import load_dotenv
import pandas as pd
import requests
import time
import glob

load_dotenv()
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def get_model_response(prompt, model, apikey=OPENROUTER_API_KEY, max_retries=3):
    headers = {"Authorization": f"Bearer {apikey}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0 if model in [
            "openai/gpt-5", 
            "google/gemini-2.5-pro", 
            "anthropic/claude-sonnet-4"
        ] else None
    }
    if payload["temperature"] is None:
        del payload["temperature"]
    r = None
    e = None
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            e = exc
            print(f"Attempt {attempt+1} failed with model {model}: {e}")
            time.sleep(2)
    print(f"Giving up on model {model}. Last error: {e}")
    return f"ERROR: {e}"

def patch_missing_responses(csv_path):
    df = pd.read_csv(csv_path)
    nan_mask = df[["used_llm", "model_response"]].isna().any(axis=1)
    nan_indices = df[nan_mask].index.tolist()
    print(f"\nFile: {csv_path} — Found {len(nan_indices)} rows with missing model_response or used_llm.")
    patch_count = 0

    for idx in nan_indices:
        prompt_id = df.at[idx, 'prompt_id'] if 'prompt_id' in df.columns else idx
        used_llm = df.at[idx, 'used_llm']
        prompt = df.at[idx, 'final_prompt']
        if pd.isna(used_llm) or pd.isna(prompt):
            print(f"  Skipping prompt_id={prompt_id} due to missing used_llm or final_prompt")
            continue
        print(f"  Patching: prompt_id={prompt_id}, used_llm={used_llm}")
        response = get_model_response(prompt, used_llm)
        df.at[idx, 'model_response'] = response
        patch_count += 1
        time.sleep(1.5)
    # Overwrite CSV only if we patched anything
    if patch_count > 0:
        df.to_csv(csv_path, index=False)
        print(f"  Patched {patch_count} rows in-place in {csv_path}")

def main():
    csv_files = glob.glob("output/llm_response_*.csv")
    print(f"Found {len(csv_files)} CSV files to process.")
    for csv_path in csv_files:
        patch_missing_responses(csv_path)

if __name__ == "__main__":
    main()
