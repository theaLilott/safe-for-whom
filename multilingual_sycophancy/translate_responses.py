#!/usr/bin/env python3
"""
Translate Responses to English
==============================
Standalone script to translate a response column in any CSV to English
using a HuggingFace causal LM (e.g., Qwen3-32B).

Usage:
    # Basic usage
    python translate_responses.py \
        --input results/rq1_generations.csv \
        --response_col response_raw \
        --lang_col lang

    # Custom output column name & model
    python translate_responses.py \
        --input results/rq1_generations.csv \
        --response_col response_raw \
        --lang_col lang \
        --output_col response_english \
        --model Qwen/Qwen3-32B \
        --output results/rq1_generations_translated.csv

    # Test on a few rows
    python translate_responses.py \
        --input results/rq1_generations.csv \
        --response_col response_raw \
        --lang_col lang \
        --n_samples 5
"""

import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ── Language display names (extend as needed) ─────────────────────────────────
LANGUAGE_NAMES = {
    "german": "German",
    "indonesian": "Indonesian",
    "swahili": "Swahili",
    "spanish": "Spanish",
    "french": "French",
    "portuguese": "Portuguese",
    "italian": "Italian",
    "arabic": "Arabic",
    "japanese": "Japanese",
    "korean": "Korean",
    "russian": "Russian",
    "vietnamese": "Vietnamese",
    "thai": "Thai",
    "chinese": "Chinese",
    "hindi": "Hindi",
    "turkish": "Turkish",
    "dutch": "Dutch",
    "polish": "Polish",
    "english": "English",
}

TRANSLATION_SYSTEM_PROMPT = """You are a professional translator. Translate the following {source_language} text into English.

CRITICAL INSTRUCTIONS:
- Preserve the informal, colloquial tone — this is a Reddit-style response.
- Keep emotional expressions, hedging, and slang — translate their SPIRIT, not just the words.
- Do NOT add explanations, notes, or commentary. Output ONLY the translation.
- Preserve paragraph structure and formatting."""


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Translate a response column in a CSV to English using an LLM."
    )
    p.add_argument(
        "--input", type=str, required=True,
        help="Path to the input CSV file.",
    )
    p.add_argument(
        "--response_col", type=str, required=True,
        help="Name of the column containing text to translate.",
    )
    p.add_argument(
        "--lang_col", type=str, required=True,
        help="Name of the column containing the source language (e.g., 'german', 'swahili').",
    )
    p.add_argument(
        "--output_col", type=str, default=None,
        help="Name of the new column for translated text. Default: '<response_col>_english'.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to the output CSV. Default: '<input_stem>_translated.csv' in same dir.",
    )
    p.add_argument(
        "--model", type=str, default="Qwen/Qwen3-32B",
        help="HuggingFace model for translation (default: Qwen/Qwen3-32B).",
    )
    p.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index (default: 0).",
    )
    p.add_argument(
        "--n_samples", type=int, default=None,
        help="Limit to N samples per language (for testing).",
    )
    p.add_argument(
        "--checkpoint_every", type=int, default=25,
        help="Save checkpoint every N translations (default: 25).",
    )
    p.add_argument(
        "--english_label", type=str, default="english",
        help="Value in lang_col that marks English rows (skipped). Default: 'english'.",
    )
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_name: str, device: str):
    """Load model with 4-bit quantization."""
    print(f"Loading model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map=device,
    )
    model.eval()
    print(f"  ✓ Model loaded")
    return tokenizer, model


# ── Translation ───────────────────────────────────────────────────────────────
def translate_to_english(
    text: str,
    src_lang: str,
    tokenizer,
    model,
    device: str,
) -> str:
    """Translate a single text to English. Returns as-is if already English."""
    lang_display = LANGUAGE_NAMES.get(src_lang.lower(), src_lang.title())
    system_msg = TRANSLATION_SYSTEM_PROMPT.format(source_language=lang_display)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text},
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            max_new_tokens=1024,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Resolve paths & column names ──
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    output_col = args.output_col or f"{args.response_col}_english"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_translated.csv")

    # ── Load CSV ──
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Validate columns exist
    for col_name, col_val in [("response_col", args.response_col), ("lang_col", args.lang_col)]:
        if col_val not in df.columns:
            print(f"Error: column '{col_val}' not found. Available columns: {df.columns.tolist()}")
            return

    print(f"Languages found: {df[args.lang_col].unique().tolist()}")

    # ── Subsample if testing ──
    if args.n_samples is not None:
        df = df.groupby(args.lang_col).head(args.n_samples).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows ({args.n_samples} per language)")

    # ── Initialize output column & handle resume ──
    if output_col not in df.columns:
        df[output_col] = None

    # Resume from existing checkpoint
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        if len(df_existing) == len(df) and output_col in df_existing.columns:
            df[output_col] = df_existing[output_col]
            already_done = df[output_col].notna().sum()
            print(f"✓ Resuming from checkpoint: {already_done} already translated")

    # ── English rows: copy directly ──
    english_mask = (df[args.lang_col].str.lower() == args.english_label.lower()) & df[output_col].isna()
    df.loc[english_mask, output_col] = df.loc[english_mask, args.response_col]
    n_english = english_mask.sum()
    if n_english > 0:
        print(f"  Copied {n_english} English rows directly (no translation needed)")

    # ── Identify rows needing translation ──
    needs_translation = (
        (df[args.lang_col].str.lower() != args.english_label.lower())
        & df[output_col].isna()
        & df[args.response_col].notna()
    )
    translate_indices = df.index[needs_translation].tolist()
    print(f"\n{len(translate_indices)} rows to translate")

    if len(translate_indices) == 0:
        print("Nothing to do — all rows already translated.")
        df.to_csv(output_path, index=False)
        print(f"Output saved to {output_path}")
        return

    # ── Load model ──
    device = f"cuda:{args.gpu}"
    tokenizer, model = load_model(args.model, device)

    # ── Translate ──
    for count, idx in enumerate(tqdm(translate_indices, desc="Translating")):
        row = df.iloc[idx]
        try:
            df.at[idx, output_col] = translate_to_english(
                text=row[args.response_col],
                src_lang=row[args.lang_col],
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
        except Exception as e:
            tqdm.write(f"  ⚠️ Error at idx {idx} ({row[args.lang_col]}): {e}")
            df.at[idx, output_col] = None  # Leave as None so resume can retry

        # Checkpoint
        if (count + 1) % args.checkpoint_every == 0:
            df.to_csv(output_path, index=False)
            tqdm.write(f"  ✓ Checkpoint saved ({count + 1}/{len(translate_indices)})")

    # ── Save final ──
    df.to_csv(output_path, index=False)

    # ── Summary ──
    n_translated = df[output_col].notna().sum()
    n_failed = df[output_col].isna().sum()
    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"{'='*50}")
    print(f"  Output:      {output_path}")
    print(f"  Translated:  {n_translated}/{len(df)}")
    if n_failed > 0:
        print(f"  Failed:      {n_failed} (re-run to retry)")
    print(f"  New column:  '{output_col}'")


if __name__ == "__main__":
    main()
