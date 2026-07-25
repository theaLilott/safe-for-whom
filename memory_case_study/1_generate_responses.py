#!/usr/bin/env python3
"""
1_generate_responses.py -- PASS 1 (run locally)

Query each target model over the persona x advice-prompt grid via OpenRouter and
save the raw responses to out/responses.jsonl. No judging happens here, so this
pass is cheap to keep and never needs re-running once complete.

    pip install openai datasets pandas
    export OPENROUTER_API_KEY="sk-or-..."
    huggingface-cli login          # SORRY-Bench dataset is gated

    python 1_generate_responses.py --limit 5     # smoke test
    python 1_generate_responses.py               # full run (resumable)

Resumable: already-generated (model, persona, question, style) rows are skipped.
"""

import argparse
import os
import time

from openai import OpenAI

import common as C

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

TARGET_MODELS = [
    'meta-llama/llama-3.1-8b-instruct', 
    'meta-llama/llama-3.3-70b-instruct', 
    'deepseek/deepseek-v4-flash', 
    'google/gemma-4-26b-a4b-it',
    'mistralai/mistral-small-24b-instruct-2501', 
    'qwen/qwen3-235b-a22b-2507',
    "microsoft/phi-4",
    "qwen/qwen-2.5-7b-instruct"
]

TARGET_TEMPERATURE = 0.7
TARGET_MAX_TOKENS = 1024

SLEEP_BETWEEN_CALLS = 0      # :free tiers are tightly rate-limited
MAX_RETRIES = 6
BACKOFF_BASE = 4.0


def generate(model, system_prompt, user_question):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question},
                ],
                temperature=TARGET_TEMPERATURE,
                max_tokens=TARGET_MAX_TOKENS,
                extra_headers={
                    "HTTP-Referer": "https://example.org/thesis-pilot",
                    "X-Title": "memory-refusal-pilot",
                },
            )
            return completion.choices[0].message.content
        except Exception as e:  # noqa: BLE001 (RateLimitError, APIError, timeouts)
            last_err = e
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {e}; sleeping {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"OpenRouter call failed after {MAX_RETRIES} retries: {last_err}")


def run(limit=None):
    rows = C.load_advice_prompts()
    if not rows:
        raise SystemExit("No advice prompts loaded -- check dataset access / config.")
    if limit:
        rows = rows[:limit]

    personas = list(C.all_personas())
    done = C.load_done_keys(C.RESPONSES_PATH)
    total = len(TARGET_MODELS) * len(personas) * len(rows)
    print(f"Models: {len(TARGET_MODELS)} | personas: {len(personas)} | prompts: {len(rows)} "
          f"| total: {total} (already done: {len(done)})")

    n = 0
    for model in TARGET_MODELS:
        for persona in personas:
            for q in rows:
                n += 1
                rec = {
                    "target_model": model,
                    "persona_id": persona["persona_id"],
                    "question_id": q["question_id"],
                    "prompt_style": q["prompt_style"],
                }
                if C.key_of(rec) in done:
                    continue
                print(f"[{n}/{total}] {model} | {persona['persona_id']} | "
                      f"q={q['question_id']} | {q['prompt_style']}")
                try:
                    answer = generate(model, persona["system_prompt"], q["question"])
                    time.sleep(SLEEP_BETWEEN_CALLS)
                except Exception as e:  # noqa: BLE001
                    print(f"  !! skipping after errors: {e}")
                    continue

                rec.update({
                    "gender": persona["gender"],
                    "income": persona["income"],
                    "category": q["category"],
                    "question": q["question"],
                    "response": answer,
                })
                C.append_jsonl(C.RESPONSES_PATH, rec)

    print(f"\nDone. Responses in {C.RESPONSES_PATH}. Next: run 2_judge_responses.py on a GPU.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N prompts (cheap smoke test)")
    args = ap.parse_args()
    run(limit=args.limit)
