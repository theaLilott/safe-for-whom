# Memory-Based Contextualisation Case Study (Chapter 5, Case Study 3)

Code for the memory-conditioned refusal experiment in Chapter 5 of the thesis
*Safe for Whom? Evaluating AI Safety Evaluations for Demographic Blindspots*.

The experiment tests whether models refuse advice requests at different rates
depending on demographic information injected via a simulated assistant
"memory": six personas spanning 2 genders x 3 income contexts (signalled
through university and home location), crossed with the advice-domain
categories (40-44) of [SORRY-Bench](https://huggingface.co/sorry-bench)
in four prompt styles (`base`, `slang`, `misspellings`, `question`).

## Pipeline

| Step | Script | Where | Output |
|------|--------|-------|--------|
| 1 | `1_generate_responses.py` | local, via [OpenRouter](https://openrouter.ai) | `out/responses.jsonl` |
| 2 | `2_judge_responses.py` | GPU (vLLM), official SORRY-Bench fine-tuned judge | `out/judgments.jsonl`, `out/judgments.csv`, `out/summary.csv` |
| 3 | `3_plot_results.py` | local | `figures/fig1_by_gender.pdf`, `figures/fig2_by_income.pdf` |

`common.py` holds the shared configuration: persona grid, memory/system-prompt
templates, SORRY-Bench loading, the verbatim judge prompt, and the
`summarize()` aggregation. Both passes are resumable and join on the key
`(target_model, persona_id, question_id, prompt_style)`.

The judging pass mirrors the official SORRY-Bench
`gen_judgment_safety_vllm.py` protocol for the
`ft-mistral-7b-instruct-v0.2-sorry-bench-202406` judge (temperature 0,
"scoreonly" extraction; 1 = compliance, 0 = refusal).

## Setup

```bash
pip install -r requirements.txt        # pass 1 + plotting
pip install vllm transformers          # pass 2 only (needs a GPU)

cp .env.example .env                   # then fill in your key
huggingface-cli login                  # SORRY-Bench dataset and judge are gated
```

Run:

```bash
python 1_generate_responses.py --limit 5    # smoke test
python 1_generate_responses.py              # full run (resumable)
python 2_judge_responses.py                 # on the GPU machine
python 2_judge_responses.py --summarize     # tables/summary only, no GPU needed
python 3_plot_results.py                    # figures from out/summary.csv
```

## Data availability

Only the aggregated `out/summary.csv` is committed. Raw model responses are
archived outside this repository and available from the author on request.

**Note on runs:** the committed `out/summary.csv` stems from an earlier pilot
over six API models. The thesis figures in `figures/` come from the final run
over eight open-weight models (the `TARGET_MODELS` list in
`1_generate_responses.py`); its raw responses are archived, but the
per-response judgments from the GPU judging pass were not retained.
`3_plot_results.py` reproduces the figures' format from any
`summary.csv` written by `common.summarize()`.
