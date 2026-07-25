# Safe for Whom? Thesis Code

Code accompanying the Bachelor's thesis

> **Safe for Whom? Evaluating AI Safety Evaluations for Demographic Blindspots**
> Manon Kempermann, Saarland University

The thesis introduces the notion of *safety discrimination* (the deprivation
of safeguards by demographic background) and examines it along two axes:
*methodological* (are our evaluation methods demographically inclusive?) and
*behavioural* (do models act differently-safe depending on who the user is?).
It has four empirical components: a mapping of 23 safety benchmarks along six
demographic axes, a multilingual/multidialectal sycophancy evaluation, three
case studies on demographic contextualisation, and cross-lingual linear-probe
transfer for sycophancy detection.

## Chapter → code map

| Thesis chapter | Component | Code |
|---|---|---|
| Ch. 3: Benchmark mapping | C1: 23 safety benchmarks × 6 demographic axes | No code (manual literature analysis; tables in the thesis) |
| Ch. 4: Lost in Translation | C2: multilingual sycophancy evaluation (RQ1) | [`multilingual_sycophancy/`](multilingual_sycophancy/) |
| Ch. 5: Contextualising, case studies 1 + 2 | C3: user-welfare safety with disclosed context (health/finance) | [`context_eval/`](context_eval/) |
| Ch. 5: Contextualising, case study 3 | C3: memory-based demographic contextualisation | [`memory_case_study/`](memory_case_study/) |
| Ch. 6: Beyond Evaluations | C4: cross-lingual probes + causal intervention (RQ2, RQ3) | [`multilingual_sycophancy/`](multilingual_sycophancy/) |

## Contents

- **`multilingual_sycophancy/`**: Does social sycophancy differ across
  languages, can LLM judges evaluate it cross-lingually (Ch. 4), and do linear
  probes transfer across languages / causally steer sycophancy (Ch. 6)?
  Subject model Qwen3-14B on translated AITA-YTA dilemmas in 8 languages.
- **`context_eval/`**: Snapshot of the public repository for the IASEAI 2026
  paper *Challenges of Evaluating LLM Safety for User Welfare*
  (canonical version: <https://github.com/theaLilott/context_eval>), which
  forms case studies 1 and 2 of Chapter 5: context-blind vs. context-aware
  safety of health/financial advice for vulnerable users. Bulky generated
  artifacts are omitted here; see `context_eval/SNAPSHOT_NOTE.md`.
- **`memory_case_study/`**: Case study 3 of Chapter 5: do refusal rates on
  SORRY-Bench advice prompts change when demographic context is injected via a
  simulated assistant "memory"?

Each folder has its own README with setup and pipeline instructions.

## Data availability

Aggregated results (summary CSVs, sweep JSONs, figures) are committed. Raw
per-response model outputs and judge scores (several hundred MB) are archived
outside this repository and available from the author on request; for case
studies 1 and 2 they are in the canonical `context_eval` repository.

## Submission archive

A self-contained zip of this repository:

```bash
git archive --format=zip HEAD -o safe-for-whom.zip
```
