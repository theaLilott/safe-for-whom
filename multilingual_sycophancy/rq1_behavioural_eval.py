#%%
import os
import torch

# 1. Force the cache path to your large storage
# This overrides everything else.
os.environ["HF_HOME"] = "/NS/MAS-llms01/nobackup/mkemperm/huggingface"

# 2. Verify it worked
print(f"Hugging Face will save to: {os.environ['HF_HOME']}")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
import pandas as pd
from tqdm import tqdm
import gc
import huggingface_hub 
from dotenv import load_dotenv

load_dotenv() 

hf_token = os.getenv("HF_TOKEN") 

# 3. Login
if hf_token:
    huggingface_hub.login(token=hf_token)
    print("Logged in to Hugging Face successfully.")
else:
    print("Warning: HF_TOKEN not found in .env environment.")
#%%
# --- CONFIG ---
SUBJECT_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"
LANGUAGES = ["english", "german", "indonesian", "swahili"]
OUTPUT_FILE = "results/rq1_raw_generations.csv"

# --- LOAD MODEL (4-BIT) ---
# Critical for A100 80GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading Subject Model: {SUBJECT_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    SUBJECT_MODEL,
    quantization_config=bnb_config,
    device_map="cuda:1"
)

results = []

#%%
# --- GENERATION LOOP ---
for lang in LANGUAGES:
    print(f"\nGenerating for Language: {lang}")
    # We use the TEST set to be scientifically rigorous (behavior on unseen data)
    dataset = load_from_disk(f"datasets/{lang}_dataset")["test"]
    
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = example["prompt"]
        
        # Chat Template
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to("cuda:1")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                do_sample=True, 
                max_new_tokens=128,
                temperature=1,
                top_p=0.9
            )
        
        # Extract only the response
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        results.append({
            "original_index": i, 
            "lang": lang,
            "prompt_used": prompt,
            "response_raw": response
        })

# Save
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print(f"Saved generations to {OUTPUT_FILE}")
# %%
#%%
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- CONFIG ---
INPUT_FILE = "results/rq1_raw_generations.csv"
OUTPUT_FILE = "results/rq1_translated_generations.csv"
TRANS_MODEL = "facebook/nllb-200-3.3B"
CHECKPOINT_INTERVAL = 100  # Save every N rows

# NLLB Codes for Source Languages
CODE_MAP = {
    "german": "deu_Latn",
    "indonesian": "ind_Latn",
}

# --- LOAD DATA ---
df = pd.read_csv(INPUT_FILE)

# Check for resume capability
if os.path.exists(OUTPUT_FILE):
    df_checkpoint = pd.read_csv(OUTPUT_FILE)
    if len(df_checkpoint) == len(df) and "response_english" in df_checkpoint.columns:
        start_idx = df_checkpoint["response_english"].notna().sum()
        df = df_checkpoint
        print(f"✓ Resuming from row {start_idx}/{len(df)}")
    else:
        df["response_english"] = None
        start_idx = 0
else:
    df["response_english"] = None
    start_idx = 0

print(f"Total rows: {len(df)} | Starting from: {start_idx}")

# --- LOAD TRANSLATOR ---
print(f"Loading Translator: {TRANS_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANS_MODEL, 
    torch_dtype=torch.float16
).to("cuda:1")  

def translate_to_english(text, src_lang):
    """Translate text to English, or return as-is if already English"""
    if src_lang == "english": 
        return text
    
    tokenizer.src_lang = CODE_MAP[src_lang]
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512  
    ).to("cuda:1")
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=256
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- EXECUTE WITH CHECKPOINTING ---
print("Starting translation...")

for idx in tqdm(range(start_idx, len(df)), desc="Translating responses"):
    row = df.iloc[idx]
    
    try:
        df.at[idx, "response_english"] = translate_to_english(
            row["response_raw"], 
            row["lang"]
        )
    except Exception as e:
        print(f"\n⚠️ Error at row {idx} ({row['lang']}): {e}")
        df.at[idx, "response_english"] = row["response_raw"]  # Fallback
    
    # Incremental save
    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        tqdm.write(f"✓ Checkpoint: {idx + 1}/{len(df)} rows saved")

# Final save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Translation complete! Saved to {OUTPUT_FILE}")

# Statistics
print(f"\nBreakdown:")
print(df.groupby("lang")["response_english"].count())