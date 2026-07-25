import pandas as pd
df = pd.read_csv("eval_dataset/llm_responses_final.csv")
print("Unique prompt_id:", df["prompt_id"].nunique())
