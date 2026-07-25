# Snapshot note

This folder is a code snapshot of the canonical public repository
**https://github.com/theaLilott/context_eval**, which accompanies the paper
*Challenges of Evaluating LLM Safety for User Welfare* (IASEAI 2026).

The code and small data files are unchanged. To keep this thesis repository
lightweight, the following bulky generated artifacts were omitted and can be
found in the canonical repository:

- `scores/`: per-response LLM-as-judge safety scores
- `output/`: raw LLM responses per profile/domain
- `intermediate_data/`: intermediate dataset construction files
- `eval_dataset/evaluation_prompts.csv` / `.json`: the fully assembled prompt sets (regenerable with the code here)
