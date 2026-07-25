#!/usr/bin/env python3
"""
RQ1: Multilingual Sycophancy Benchmark — Generation Pipeline
=============================================================
Generates responses from a subject model across multiple languages,
with incremental checkpointing and configurable parameters.

Usage:
    # Test run (5 samples per language)
    python rq1_generate.py --n_samples 3

    # Full run
    python rq1_generate.py

    # Resume interrupted run
    python rq1_generate.py  # automatically resumes from checkpoint

    # Custom model / languages
    python rq1_generate.py --model Qwen/Qwen3-32B --languages english german --gpu 0
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
import huggingface_hub
import h5py

# ── CLI Arguments ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RQ1 Multilingual Sycophancy Generation")
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="HuggingFace model ID (default: Qwen/Qwen3-14B)",
    )
    p.add_argument(
        "--languages",
        nargs="+",
        default=["english", "german", "indonesian"],
        help="Languages to generate for",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples per language (default: all). Use e.g. 5 for testing.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate (default: 256)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)"
    )
    p.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling (default: 0.8)")
    p.add_argument("--top_k", type=int, default=20, help="Top-k sampling (default: 20)")
    p.add_argument(
        "--gpu", type=int, default=0, help="CUDA device index (default: 0)"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets",
        help="Directory containing {lang}_dataset folders",
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Save checkpoint every N examples (default: 25)",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (default: True)",
    )
    p.add_argument(
        "--no_4bit",
        action="store_true",
        default=False,
        help="Disable 4-bit quantization (load in fp16/bf16)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)",
    )
    p.add_argument(
        "--acts_collection_mode",
        type=str,
        default="average",
        help="Mode to collect activations (default: average (across generated tokens))",
    )
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_output_path(output_dir: str, model_name: str) -> Path:
    """Construct output filename from model name."""
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    return Path(output_dir) / f"rq1_generations_{model_short}_new_langs.csv"


def load_checkpoint(output_path: Path) -> tuple[pd.DataFrame | None, set]:
    """Load existing checkpoint and return (dataframe, set of completed (lang, idx) pairs)."""
    if not output_path.exists():
        return None, set()

    df = pd.read_csv(output_path)
    completed_languages = set(df["lang"].unique()) if df is not None else set()
    print(f"✓ Loaded checkpoint: {completed_languages} already done")
    return df, completed_languages


def save_checkpoint(results: list[dict], output_path: Path):
    """Save results to CSV, merging with any existing checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame(results)

    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Deduplicate on (lang, original_index), keeping last
        df_combined = df_combined.drop_duplicates(
            subset=["lang", "original_index"], keep="last"
        )
    else:
        df_combined = df_new

    df_combined.to_csv(output_path, index=False)
    return len(df_combined)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"

    # ── Output path ──
    output_path = get_output_path(args.output_dir, args.model)
    print(f"Output: {output_path}")

    # ── Load checkpoint ──
    _, completed_lang = load_checkpoint(output_path)

    # ── Load model ──
    print(f"\nLoading model: {args.model}")
    print(f"  Device: {device}")
    print(f"  4-bit: {args.load_in_4bit and not args.no_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": device}

    if args.load_in_4bit and not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    print("✓ Model loaded\n")

    # ── Log config ──
    config_summary = {
        "model": args.model,
        "languages": args.languages,
        "n_samples": args.n_samples or "all",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "split": args.split,
        "timestamp": datetime.now().isoformat(),
    }
    print("Config:", config_summary)

    # ── Generation loop ──
    buffer = []  # Accumulate results between checkpoints
    Path("activations").mkdir(parents=True, exist_ok=True)
    with h5py.File("activations/natural_completions.h5", "a") as f:
    
        for lang in args.languages:
            print(f"\n{'='*60}")
            print(f"Language: {lang}")
            print(f"{'='*60}")

            dataset_path = Path(args.dataset_dir) / f"{lang}_dataset"
            if not dataset_path.exists():
                print(f"⚠️  Dataset not found: {dataset_path} — skipping")
                continue

            dataset = load_from_disk(str(dataset_path))[args.split]
            n_total = len(dataset) if args.n_samples is None else min(args.n_samples, len(dataset))

            activations_per_lang = []
            indices_per_lang = []
            
            if (lang in f) and (lang in completed_lang):
                continue
            
            for i in tqdm(range(n_total), desc=f"  {lang}"):

                example = dataset[i]
                prompt = example["prompt"]
                # prompt_english is stored in every language dataset by prepare_datasets.py
                prompt_english = example.get("prompt_english", prompt)

                # ── Build input ──
                messages = [{"role": "user", "content": prompt}]

                # Qwen3: disable thinking mode for clean sycophancy measurement
                # For non-Qwen models this kwarg is silently ignored by most tokenizers
                try:
                    input_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    # Fallback for tokenizers that don't support enable_thinking
                    input_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                inputs = tokenizer(input_text, return_tensors="pt").to(device)

                # ── Generate ──
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        do_sample=True,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                
                """if args.acts_collection_mode == "first":
                    # collect activation at last input token
                    with torch.no_grad():
                        fwd = model(inputs["input_ids"], output_hidden_states=True)
                        activations = torch.stack([fwd.hidden_states[layer][0, -1, :] for layer in range(len(fwd.hidden_states))]).float()
                
                elif args.acts_collection_mode =="average":
                    full_sequence = outputs[0].unsqueeze(0)        
                    fwd = model(full_sequence, output_hidden_states=True)
                    # fwd.hidden_states[layer] shape: (1, seq_len, hidden_dim)
                    first_gen = inputs["input_ids"].shape[1]
                    activations = torch.stack([
                        fwd.hidden_states[layer][0, first_gen:, :].mean(dim=0)
                    for layer in range(len(fwd.hidden_states))
                ])  # shape: (n_layers+1, hidden_dim)

                
                
                
                activations_per_lang.append(activations.cpu().numpy())
                indices_per_lang.append(i)"""
                
                
                
                # Extract only the new tokens
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )

                buffer.append(
                    {
                        "original_index": i,
                        "lang": lang,
                        "prompt_used": prompt,
                        "prompt_english": prompt_english,
                        "response_raw": response,
                        "model": args.model,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_new_tokens": args.max_new_tokens,
                    }
                )

                # ── Incremental checkpoint ──
                if len(buffer) >= args.checkpoint_every:
                    total_saved = save_checkpoint(buffer, output_path)
                    tqdm.write(f"  ✓ Checkpoint: {total_saved} total rows saved")
                    buffer = []

            
            """grp = f.create_group(lang)
            data=np.stack(activations_per_lang)
            grp.create_dataset("activations", data=data)
            grp.create_dataset("indices", data=indices_per_lang)"""


    # ── Final save ──
    if buffer:
        total_saved = save_checkpoint(buffer, output_path)
        print(f"\n✓ Final save: {total_saved} total rows → {output_path}")

    # ── Summary ──
    if output_path.exists():
        df_final = pd.read_csv(output_path)
        print(f"\n{'='*60}")
        print("Generation Summary")
        print(f"{'='*60}")
        print(df_final.groupby("lang").size().to_string())
        print(f"\nTotal: {len(df_final)} generations")
    else:
        print("\n⚠️  No generations were produced.")


if __name__ == "__main__":
    main()