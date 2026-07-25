import os
import pandas as pd
import requests
import time
from dotenv import load_dotenv
import sys

# =============== CONFIGURATION ===============
INPUT_CSV = "eval_dataset/evaluation_prompts.csv"
OUTPUT_CSV = None  # Will set it below, based on request_id
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# List of models to run when ai_model is "All"
MODELS_FOR_ALL = [
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro",
    "openai/gpt-5"
]

API_URL = "https://openrouter.ai/api/v1/chat/completions"


import requests

def get_model_response(prompt, model, apikey=OPENROUTER_API_KEY, max_retries=3):
    headers = {"Authorization": f"Bearer {apikey}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0 if model in ["openai/gpt-5", "google/gemini-2.5-pro", "anthropic/claude-sonnet-4"] else None
    }
    if payload["temperature"] is None:
        del payload["temperature"]

    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, headers=headers)
            if r.status_code == 403:
                # Try to get error code and reason from body
                try:
                    error_info = r.json().get('error', {})
                    message = error_info.get('message', '').lower()
                    metadata = error_info.get('metadata', {})
                    if 'moderation' in message or 'flagged' in message or 'moderation' in metadata.get('provider_name', ''):
                        print(f"🚩 Moderation error (content flagged) for model {model}, prompt: {prompt[:50]}...")
                    else:
                        print(f"🚫 Access/credits error for model {model}. Message: {message}")
                except Exception as parse_exc:
                    print(f"403 error but could not parse JSON: {parse_exc}")
                return None  # or handle as you wish
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Attempt {attempt+1} failed with model {model}: {e}")
            time.sleep(2)
    print(f"❌ All attempts failed for model {model}. Skipping and returning NaN.")
    return None





def main():

    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set in environment or .env!")


    request_id_arg = None
    if len(sys.argv) > 1:
        request_id_arg = sys.argv[1]

    df = pd.read_csv(INPUT_CSV)
    if request_id_arg is not None:
        df = df[df["request_id"] == request_id_arg]
        print(f"Loaded {len(df)} rows for request_id '{request_id_arg}' from {INPUT_CSV}", flush=True)
        OUTPUT_CSV = f"output/llm_response_{request_id_arg}.csv"
    else:
        print(f"Loaded {len(df)} rows from {INPUT_CSV}", flush=True)
        OUTPUT_CSV = "output/llm_responses_all.csv"

    # Setup output file and resume (skip prompt_id + used_llm combos already done)
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        done = pd.read_csv(OUTPUT_CSV)
        done_keys = set(zip(done["prompt_id"], done["used_llm"]))
        print(f"Resuming: {len(done_keys)} prompt_id+used_llm combos already done, will skip.")
    else:
        done_keys = set()

    # === Build jobs list: [(row_idx, row, model), ...] ===
    jobs = []
    for i, row in df.iterrows():
        prompt_id = row.get("prompt_id", row.get("request_id", i))
        ai_model = row["ai_model"]
        if ai_model.strip().lower() == "all":
            models_to_run = MODELS_FOR_ALL
        else:
            models_to_run = [ai_model.strip()]
        for model in models_to_run:
            key = (prompt_id, model)
            if key in done_keys:
                continue
            jobs.append((i, row, model))

    total_jobs = len(jobs)
    print(f"Will run {total_jobs} model-prompt jobs.")

    # === Main jobs loop ===
    for job_num, (i, row, model) in enumerate(jobs, 1):
        prompt_id = row.get("prompt_id", row.get("request_id", i))
        prompt = row["final_prompt"]
        print(f"[{job_num}/{total_jobs}] Running: prompt_id={prompt_id}, model={model}, theme={row.get('theme', '')}", flush=True)

        response = get_model_response(prompt, model)
        time.sleep(1.5)

        out_row = row.to_dict()
        out_row["used_llm"] = model
        out_row["model_response"] = response

        # Only keep required columns
        output_columns = [
            "prompt_id", "topic", "theme",
            "vulnerability_profile_level", "vulnerability_profile",
            "ranking_type", "used_llm", "final_prompt", "model_response"
        ]
        out_row = {k: out_row.get(k, "") for k in output_columns}

        # Write to output (append, header only if new file)
        write_header = not os.path.isfile(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
        pd.DataFrame([out_row]).to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False
        )
        done_keys.add((prompt_id, model))

    print(f"\n✅ Done! All responses saved to {OUTPUT_CSV}", flush=True)

if __name__ == "__main__":
    main()
