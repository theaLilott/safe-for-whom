import pandas as pd

CSV = "eval_dataset\stratified_sampled_12_with_contextdiv_modelrot_ctxspread.csv"
df = pd.read_csv(CSV)

# 1. For each theme, print the mapping of (ranking_type, vul_level) → ai_model, context_level
for (topic, theme), g in df.groupby(["topic", "theme"]):
    print(f"\n--- {topic} | {theme} ---")
    print(g[["ranking_type", "vulnerability_profile_level", "ai_model", "context_level"]]
            .reset_index(drop=True))

# 2. Show how many unique context_levels per theme
print("\nContext level coverage per theme:")
print(df.groupby(["topic", "theme"])["context_level"].agg(["nunique", "unique"]))

# 3. Check that AI ranking_type is cycling models
print("\nAI model assignment for ranking_type=AI (should cycle anthropic/gemini/gpt-5):")
for (topic, theme), g in df[df["ranking_type"] == "AI"].groupby(["topic", "theme"]):
    print(f"{topic} | {theme}:")
    print(g[["vulnerability_profile_level", "ai_model", "context_level"]].reset_index(drop=True))
