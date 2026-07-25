# src/main.py

import pandas as pd
from src.logger import log, log_file
from utils_pipeline import get_models_for_row, call_model_api
from src.config import DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_CSV

def main():
    log.info(f"Logging to {log_file}")
    log.info(f"Loading prompts from {DEFAULT_INPUT_CSV}")

    df = pd.read_csv(DEFAULT_INPUT_CSV)
    results = []

    for idx, row in df.iterrows():
        prompt_id = row['prompt_id']
        prompt = row['final_prompt']
        ranking_type = str(row['ranking_type']) if pd.notna(row['ranking_type']) else ""
        ai_model = row['ai_model'] if pd.notna(row['ai_model']) else ""

        models = get_models_for_row(ranking_type, ai_model)

        for model_name in models:
            log.info(f"[{prompt_id}] Calling model: {model_name}")
            response = call_model_api(model_name, prompt)

            if response is None:
                log.error(f"[{prompt_id}] No response from {model_name}, skipping.")
                continue

            # Save to result list
            results.append({
                "prompt_id": prompt_id,
                "model_name": model_name,
                "final_prompt": prompt,
                "model_response": response,
                "ranking_type": ranking_type,
                "ai_model": ai_model,
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(DEFAULT_OUTPUT_CSV, index=False)
    log.info(f"Saved model responses to {DEFAULT_OUTPUT_CSV}")

if __name__ == "__main__":
    main()
