from pathlib import Path
import argparse
import h5py
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
import huggingface_hub
import re
import pandas as pd

#import circuitsvis as cv
import numpy as np
import torch as t
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser(description="RQ2 Multilingual Sycophancy Activation Collection")
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="HuggingFace model ID (default: Qwen/Qwen3-14B)",
    )
    p.add_argument(
        "--languages",
        nargs="+",
        default=["english", "german", "indonesian", "thai", "russian", "arabic", "japanese", "italian", "spanish"],
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
        "--gpu", type=int, default=1, help="CUDA device index (default: 0)"
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
        "--average_acts",
        type=bool,
        default=True,
        help="average activations over whole sequence (default: True)",
    )
    
    return p.parse_args()


def extract_activations(
    dataset,
    lang,
    verdicts,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args
):
    """
    Extract last-token hidden state activations from specified layers for a list of statements.

    Args:
        statements: List of text statements to process.
        model: A HuggingFace causal language model.
        tokenizer: The corresponding tokenizer.
        layers: List of layer indices (0-indexed) to extract activations from.
        batch_size: Number of statements to process at once.

    Returns:
        Dictionary mapping layer index to tensor of activations, shape [n_statements, d_model].
    """
    activations = []
    labels = []
    
    for i in tqdm(range(len(dataset)), desc=f"  {lang}"):
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
        
        verdict_syc = verdicts[lang]["sycophantic"]
        verdict_nosyc = verdicts[lang]["honest"]

        prefix_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        
        forced_ids_syc = tokenizer(verdict_syc, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        
        forced_ids_nosyc = tokenizer(verdict_nosyc, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        

        full_ids_syc = t.cat([prefix_ids, forced_ids_syc], dim=1)
        
        full_ids_nosyc = t.cat([prefix_ids, forced_ids_nosyc], dim=1)
        
        with t.no_grad():
            fwd_syc = model(full_ids_syc, output_hidden_states=True)
            
            fwd_nosyc = model(full_ids_nosyc, output_hidden_states=True)
            
        if args.average_acts:
            first_gen = prefix_ids.shape[1]
            
            activations_syc = t.stack([
                        fwd_syc.hidden_states[layer][0, first_gen:, :].mean(dim=0)
                    for layer in range(len(fwd_syc.hidden_states))
                ]).float()
            activations_nosyc = t.stack([
                        fwd_nosyc.hidden_states[layer][0, first_gen:, :].mean(dim=0)
                    for layer in range(len(fwd_nosyc.hidden_states))
                ]).float()
        else:
            activations_syc = t.stack([fwd_syc.hidden_states[layer][0, -1, :] for layer in range(len(fwd_syc.hidden_states))]).float()
            
            activations_nosyc = t.stack([fwd_nosyc.hidden_states[layer][0, -1, :] for layer in range(len(fwd_syc.hidden_states))]).float()
         
        
        activations.extend([activations_syc.cpu().numpy(), activations_nosyc.cpu().numpy()])
        
        labels.extend([0, 1]) # 0 = sycophantic, 1 = non sycophantic
    
    activations_tensor = np.stack(activations)   
         
    return activations_tensor, labels

def extract_activations_nat(
    dataset,
    lang,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args
): 
    activations = []
    labels = []
    
    for i, row in tqdm(dataset.iterrows(), desc=f"  {lang}"):
        messages = [{"role": "user", "content": row["prompt_used"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

        response_ids = tokenizer(
            row["response_raw"], add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

        full_ids = t.cat([prompt_ids, response_ids], dim=1)
        
        with t.no_grad():
            fwd = model(full_ids, output_hidden_states=True)

        first_gen = prompt_ids.shape[1]
        if args.average_acts:
            acts = t.stack([fwd.hidden_states[layer][0, first_gen:, :].mean(dim=0) for layer in range(len(fwd.hidden_states))]).float().cpu().numpy()
        else: 
            acts = t.stack([fwd.hidden_states[layer][0, first_gen-1, :] for layer in range(len(fwd.hidden_states))]).float().cpu().numpy()
            
            
            
        
        label = row["judge_label"]
        
        activations.append(acts)
        
        labels.append(label) # 0 = sycophantic, 1 = non sycophantic
    
    activations_tensor = np.stack(activations)   
         
    return activations_tensor, labels
    
    
def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"


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
            bnb_4bit_compute_dtype=t.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = t.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    print("✓ Model loaded\n")
    
    # ── load verdicts ──
    
    with open('datasets/verdicts_improved.json', 'r') as v:
        verdicts = json.load(v)

    # ── Log config ──
    config_summary = {
        "model": args.model,
        "languages": args.languages,
        "n_samples": args.n_samples or "all",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "timestamp": datetime.now().isoformat(),
    }
    print("Config:", config_summary)
    
    
    Path("activations").mkdir(parents=True, exist_ok=True)
    
    for lang in args.languages:
        print(f"\n{'='*60}")
        print(f"Language: {lang}")
        print(f"{'='*60}")

        dataset_path = Path(args.dataset_dir) / f"{lang}_dataset"
        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset_path} — skipping")
            continue

        train_dataset = load_from_disk(str(dataset_path))["train"]
        
        test_dataset = load_from_disk(str(dataset_path))["test"]
        
        df =  pd.read_csv("results/rq1_clean.csv")
        nat_dataset = df[df["lang"] == lang]
        
        if args.n_samples:
            train_dataset = train_dataset.select(range(args.n_samples))
            test_dataset  = test_dataset.select(range(args.n_samples))
            nat_dataset = nat_dataset[:args.n_samples]
        
        train_acts, train_labels = extract_activations(train_dataset, lang, verdicts, model, tokenizer, args)
        
        test_acts, test_labels = extract_activations(test_dataset, lang, verdicts, model, tokenizer, args)
        
        nat_acts, nat_labels = extract_activations_nat(nat_dataset, lang, model, tokenizer, args)
        
        suffix = "_avg_final" if args.average_acts else ""

        with h5py.File(f"activations/train_completions{suffix}.h5", "a") as f:
            if lang in f:
                print(f"  Skipping {lang}, already in HDF5")
                continue
            grp = f.create_group(lang)
            grp.create_dataset("activations", data=train_acts)
            grp.create_dataset("labels", data=train_labels)
            
        with h5py.File(f"activations/test_completions{suffix}.h5", "a") as f:
            if lang in f:
                print(f"  Skipping {lang}, already in HDF5")
                continue
            grp = f.create_group(lang)
            grp.create_dataset("activations", data=test_acts)
            grp.create_dataset("labels", data=test_labels)
        
        with h5py.File(f"activations/nat_completions{suffix}.h5", "a") as f:
            if lang in f:
                print(f"  Skipping {lang}, already in HDF5")
                continue
            grp = f.create_group(lang)
            grp.create_dataset("activations", data=nat_acts)
            grp.create_dataset("labels", data=nat_labels)
            
        print(f"Competed {lang} activation collection and saved.")
            

if __name__ == "__main__":
    main()
            
            
            
            
            
            
            
            
            
    
    
    
    