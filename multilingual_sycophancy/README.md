# Multilingual Sycophancy: Evaluation & Cross-Lingual Probes (Chapters 4 and 6)

Code for two thesis components that share one codebase, dataset, and cluster
pipeline:

- **RQ1 → Chapter 4 (Lost in Translation):** does social sycophancy on
  AITA-style moral dilemmas differ across languages, and can an LLM judge
  evaluate it reliably across languages?
- **RQ2 + RQ3 → Chapter 6 (Beyond Evaluations):** can linear probes trained on
  one language detect sycophancy in others, and does steering along the probe
  direction causally reduce sycophancy?

Subject model: Qwen3-14B. Judge/translator: Qwen3-32B. Languages: English,
German, Spanish, Italian, Russian, Arabic, Indonesian, Thai (+ Japanese in the
judging stage).

## RQ1 pipeline (Chapter 4)

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `prepare_datasets.py` | Filters the AITA-YTA prompts (`datasets/AITA-YTA.csv`) once on English, then LLM-translates that fixed subset into each target language (index-aligned; register-preserving). Outputs the `datasets/*_dataset` HF datasets. |
| 2 | `rq1_generate.py` | Generates subject-model responses per language, with checkpointing/resume. |
| 3 | `judge.py` | LLM-as-judge sycophancy labeling: multi-pass majority vote (Qwen3-32B, thinking mode), per-pass labels and reasoning preserved. Launched per language via `jobs/launch_rq1_judge.sh`. |
| 4 | `merge_and_clean.py` | Merges per-language `judge_*.csv`, drops unjudgeable/low-confidence rows, keeps prompts surviving in ALL languages, subsamples. |
| 5 | `rq1_analysis_plot.ipynb` | Sycophancy rate by language with Wilson CIs (thesis figure; see `figures/rq1_sycophancy_rates.png`). |

Supporting scripts:

- `rq1_evaluate.py`: alternative double-judgement evaluation following the
  ELEPHANT framework (Cheng et al., 2025): judges each response in the original
  language AND in English (translated response + original English prompt) to
  measure cross-lingual judge reliability.
- `translate_responses.py`: standalone response-to-English translation used by
  the double-judgement pass.
- `merge_results_judge.py`: simple merge + summary statistics of judge outputs.
- `rq1_behavioural_eval.py`: standalone behavioural generation script for the
  cluster (hardcoded HF cache path).

## RQ2 pipeline (Chapter 6, probes)

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `rq2_collect_activations.py` | Extracts residual-stream activations of the subject model per language into HDF5 files (several GB; not committed). |
| 2 | `rq2_probe_sweep.py` | Sweeps all layers for one (activation set, language) pair; trains mass-mean (MMProbe) and logistic-regression (LRProbe) probes. `jobs/rq2_run_sweep.sh`. |
| 3 | `rq2_aggregate_sweep.py` | Aggregates sweeps into `results/probe_sweep/best_config.json`: best activation set / layer per language and the best universal layer. |
| 4 | `rq2_probe_sweep_mixed.py` | Mixed-language probe training using each language's best activation set. |
| 5 | `rq2_probe_analysis.ipynb` | Final cross-lingual transfer analysis: heatmaps at the per-language best layer and the universal layer, layer curves (thesis figures; see `figures/rq2_*.png`). |

`rq2_probes.ipynb` is the exploratory notebook (PCA visualisations, label
construction checks) kept for transparency.

Committed aggregated results: `results/probe_sweep/` (per-language sweep JSONs,
`sweep_summary.csv`, `best_config.json`, `layer_curves.pdf`).

## RQ3 pipeline (Chapter 6, causal intervention)

`rq3_causal.py` steers the subject model along the probe direction during
generation and re-evaluates sycophancy (`jobs/rq3_intervene.sh`,
`jobs/rq3_evaluate.sh`, orchestrated by `jobs/submit_rq3_pipeline.sh`).
Committed result: `results/rq3_intervention_qwen3_14b.csv`.

## Running it

```bash
pip install -r requirements.txt
```

The `jobs/*.sh` scripts are the SLURM wrappers used on the university cluster;
they contain cluster-specific paths (`/NS/MAS-llms01/...`) that need adapting.
GPU stages (generation, judging, activations, steering) assume a single A100
class GPU; `HF_TOKEN` is read from the environment / `.env`.

## Data availability

Raw per-response generations and judge outputs (~170 MB of CSVs, including
per-pass judge checkpoints) are archived outside this repository and available
from the author on request. The committed `results/` files are the aggregated
outputs the thesis figures and tables are based on; the notebooks' raw-CSV
inputs (`results/rq1_clean.csv`, `results/rq1_generations_*.csv`) are part of
that archive.
