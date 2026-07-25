import pandas as pd
from itertools import product

INPUT = "eval_dataset/evaluation_prompts.csv"
OUTPUT = "eval_dataset/stratified_sampled_12_with_contextdiv_modelrot_ctxspread.csv"

df = pd.read_csv(INPUT)

ranking_types = sorted(df["ranking_type"].unique())  # ['AI', 'Baseline', ...]
vul_levels = sorted(df["vulnerability_profile_level"].unique())  # ['high', 'low', 'medium']
context_levels = sorted(df["context_level"].unique())  # [0, 1, 3, 5]
ai_models_ai = ['anthropic/claude-sonnet-4', 'google/gemini-2.5-pro', 'openai/gpt-5']

combos = list(product(ranking_types, vul_levels))
themes = list(df.groupby(["topic", "theme"]).groups.keys())

rows = []

for idx, (topic, theme) in enumerate(themes):
    g = df[(df["topic"] == topic) & (df["theme"] == theme)]
    # Rotate ai_model order for each theme
    ai_model_order = ai_models_ai[idx % 3:] + ai_models_ai[:idx % 3]
    theme_rows = []
    ctx_lvls_used = {cl: 0 for cl in context_levels}  # Track usage count of each context_level

    for j, (rt, vl) in enumerate(combos):
        sub = g[(g["ranking_type"] == rt) & (g["vulnerability_profile_level"] == vl)]
        if rt == "AI":
            target_model = ai_model_order[j % 3]
            sub = sub[sub['ai_model'] == target_model]
        else:
            sub = sub[sub['ai_model'] == 'All']

        # Pick the context_level with the least usage so far (to maximize spread)
        sub = sub.assign(ctx_count=sub['context_level'].map(ctx_lvls_used))
        row = sub.sort_values('ctx_count').iloc[0]
        theme_rows.append(row)
        ctx_lvls_used[row['context_level']] += 1
    rows.extend(theme_rows)

sampled = pd.DataFrame(rows)

# Save to CSV
sampled.to_csv(OUTPUT, index=False)
print(f"\n✅ Saved {len(sampled)} prompts to {OUTPUT}")

# Show audit
print("\nContext level coverage per theme:")
print(sampled.groupby(["topic", "theme"])["context_level"].agg(["nunique", "unique"]))

# Show per-theme, per-type ai_model mapping (audit)
print("\nExample selection per theme:")
for (topic, theme), g in sampled.groupby(["topic", "theme"]):
    print(f"\nTheme: {topic} | {theme}")
    print(
        g[["ranking_type", "vulnerability_profile_level", "ai_model", "context_level"]]
        .reset_index(drop=True)
    )
