#!/usr/bin/env python3
"""
Dataset Preparation: Translate AITA-YTA prompts into target languages
=====================================================================
Uses Qwen3-32B (or any instruction-tuned LLM) for high-quality translation
of colloquial Reddit AITA posts. Preserves pragmatic nuance better than NLLB.

Key design choices:
  - Filter ONCE on English, then translate only that fixed subset → index alignment guaranteed
  - Each language dataset stores both the translated prompt AND the original English prompt
  - LLM translation with explicit instructions to preserve register and tone

Usage:
    # Test with 10 samples
    python prepare_datasets.py --n_samples 10

    # Full run
    python prepare_datasets.py

    # Use a different model or GPU
    python prepare_datasets.py --model Qwen/Qwen3-14B --gpu 0
"""

import os
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, Dataset, DatasetDict
import huggingface_hub


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Prepare multilingual AITA datasets")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--input_csv", type=str, default="datasets/AITA-YTA.csv")
    p.add_argument("--output_dir", type=str, default="datasets")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=1000, help="Limit total samples for testing")
    p.add_argument("--max_token_length", type=int, default=500, help="Max tokens for filtering")
    p.add_argument("--test_size", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_checkpoint_every", type=int, default=50)
    p.add_argument(
        "--languages",
        nargs="+",
        default=["german", "indonesian"],
        help="Target languages (English is always included)",
    )
    return p.parse_args()


# ── Language config ───────────────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "german": "German",
    "indonesian": "Indonesian",
    "swahili": "Swahili",
    "spanish": "Spanish",
    "french" : "French",
    "portuguese" : "Portuguese",
    "italian" : "Italian", 
    "arabic": "Arabic",
    "japanese": "Japanese", 
    "korean" : "Korean", 
    "russian" : "Russian", 
    "vietnamese" : "Vietnamese",
    "thai" : "Thai"
}

TRANSLATION_SYSTEM_PROMPT = """You are a professional translator. Translate the following English text into {language}.

CRITICAL INSTRUCTIONS:
- Preserve the informal, colloquial tone. This is a Reddit post, not a formal document.
- Keep emotional expressions, hedging, and slang — translate their SPIRIT, not just the words.
- Do NOT add explanations, notes, or commentary. Output ONLY the translation.
- Do NOT translate the ACRONYM "AITA" — keep it as-is since it is universally understood. Make sure though to translate ALL actual words that are not ACRONYMS.
- Preserve paragraph structure and formatting."""


# ── Translation function ──────────────────────────────────────────────────────
def translate_with_llm(
    text: str,
    target_language: str,
    tokenizer,
    model,
    device: str,
) -> str:
    """Translate text using an instruction-tuned LLM."""
    system_msg = TRANSLATION_SYSTEM_PROMPT.format(language=target_language)
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
            temperature=0.3,  # Low temp for faithful translation
            top_p=0.9,
            max_new_tokens=1024,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    #print(response)
    return response


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"

    # ── Step 1: Load and filter English data ONCE ──
    print("Step 1: Loading and filtering English data")
    raw = load_dataset("csv", data_files=args.input_csv)["train"]
    raw = raw.select_columns(["prompt"])
    print(f"  Raw size: {len(raw)}")

    # Load a lightweight tokenizer just for length filtering
    # (We use the translation model's tokenizer for consistency)
    print(f"  Loading tokenizer for filtering: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def filter_by_length(example):
        tokens = tokenizer(example["prompt"], truncation=False)
        return len(tokens["input_ids"]) <= args.max_token_length

    filtered = raw.filter(filter_by_length)
    print(f"  Filtered size: {len(filtered)}")

    # Optional subsample for testing
    if args.n_samples is not None:
        filtered = filtered.select(range(min(args.n_samples, len(filtered))))
        print(f"  Subsampled to: {len(filtered)}")

    # Split
    split = filtered.train_test_split(test_size=args.test_size, seed=args.seed)
    train_prompts = split["train"]["prompt"]
    test_prompts = split["test"]["prompt"]
    print(f"  Train: {len(train_prompts)}, Test: {len(test_prompts)}")

    # ── Step 2: Save English dataset (with prompt_english = prompt for consistency) ──
    print("\nStep 2: Saving English dataset")
    en_train = Dataset.from_dict({"prompt": train_prompts, "prompt_english": train_prompts})
    en_test = Dataset.from_dict({"prompt": test_prompts, "prompt_english": test_prompts})
    en_dataset = DatasetDict({"train": en_train, "test": en_test})

    en_path = Path(args.output_dir) / "english_dataset"
    en_dataset.save_to_disk(str(en_path))
    print(f"  ✓ Saved to {en_path}")

    # ── Step 3: Load translation model ──
    if args.languages:
        print(f"\nStep 3: Loading translation model: {args.model}")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map=device,
        )
        model.eval()
        print("  ✓ Model loaded")

    # ── Step 4: Translate each language ──
    for lang in args.languages:
        lang_display = LANGUAGE_NAMES.get(lang, lang.title())
        print(f"\n{'='*60}")
        print(f"Step 4: Translating to {lang_display}")
        print(f"{'='*60}")

        lang_path = Path(args.output_dir) / f"{lang}_dataset"

        # Check for existing checkpoint
        translated_train = []
        translated_test = []
        start_train = 0
        start_test = 0

        checkpoint_path = Path(args.output_dir) / f".{lang}_checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            translated_train = ckpt.get("train", [])
            translated_test = ckpt.get("test", [])
            start_train = len(translated_train)
            start_test = len(translated_test)
            print(f"  Resuming: train={start_train}/{len(train_prompts)}, test={start_test}/{len(test_prompts)}")

        # Translate train split
        if start_train < len(train_prompts):
            print(f"\n  Translating train split ({start_train} → {len(train_prompts)})")
            for i in tqdm(range(start_train, len(train_prompts)), desc=f"  Train ({lang})"):
                try:
                    translated = translate_with_llm(
                        train_prompts[i], lang_display, tokenizer, model, device
                    )
                except Exception as e:
                    tqdm.write(f"  ⚠️ Error at train[{i}]: {e}")
                    translated = train_prompts[i]  # Fallback to English

                translated_train.append(translated)

                if (i + 1) % args.batch_checkpoint_every == 0:
                    with open(checkpoint_path, "w") as f:
                        json.dump({"train": translated_train, "test": translated_test}, f)
                    tqdm.write(f"  ✓ Checkpoint: {len(translated_train)} train translations")

        # Translate test split
        if start_test < len(test_prompts):
            print(f"\n  Translating test split ({start_test} → {len(test_prompts)})")
            for i in tqdm(range(start_test, len(test_prompts)), desc=f"  Test ({lang})"):
                try:
                    translated = translate_with_llm(
                        test_prompts[i], lang_display, tokenizer, model, device
                    )
                except Exception as e:
                    tqdm.write(f"  ⚠️ Error at test[{i}]: {e}")
                    translated = test_prompts[i]

                translated_test.append(translated)

                if (i + 1) % args.batch_checkpoint_every == 0:
                    with open(checkpoint_path, "w") as f:
                        json.dump({"train": translated_train, "test": translated_test}, f)
                    tqdm.write(f"  ✓ Checkpoint: {len(translated_test)} test translations")

        # Save as HF dataset — includes BOTH translated prompt and original English
        lang_train = Dataset.from_dict({
            "prompt": translated_train,
            "prompt_english": train_prompts,
        })
        lang_test = Dataset.from_dict({
            "prompt": translated_test,
            "prompt_english": test_prompts,
        })
        lang_dataset = DatasetDict({"train": lang_train, "test": lang_test})
        lang_dataset.save_to_disk(str(lang_path))
        print(f"  ✓ Saved {lang} dataset to {lang_path}")

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
    # ── Step 5: Translate verdict anchors (for RQ2/3 probes) ──
    if args.languages:
        print(f"\n{'='*60}")
        print("Step 5: Translating verdict anchors")
        print(f"{'='*60}")

        anchors_en = {
            "sycophantic": "You are not the asshole.",
            "honest": "You are the asshole.",
        }
        verdicts = {"english": anchors_en}

        for lang in args.languages:
            lang_display = LANGUAGE_NAMES.get(lang, lang.title())
            syc = translate_with_llm(anchors_en["sycophantic"], lang_display, tokenizer, model, device)
            hon = translate_with_llm(anchors_en["honest"], lang_display, tokenizer, model, device)

            verdicts[lang] = {"sycophantic": syc, "honest": hon}
            print(f"  {lang}: sycophantic='{syc}' | honest='{hon}'")

        verdict_path = Path(args.output_dir) / "verdicts_new_langs.json"
        with open(verdict_path, "w") as f:
            json.dump(verdicts, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved to {verdict_path}")

    print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    main()