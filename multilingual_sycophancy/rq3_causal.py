
from pathlib import Path
import h5py


import pandas as pd
import numpy as np


from pathlib import Path
import argparse





import torch as t
from datasets import load_from_disk

from jaxtyping import Bool, Float

from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
        "--output_dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of samples per language (default: 50). Use e.g. 5 for testing.",
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
        "--scale",
        type=float,
        default=-1.0,
        help="scaling paramenter (default: -1.0)",
    )
    p.add_argument(
        "--intervention_layer",
        type=int,
        default=24,
        help="layer to apply intervention to (default: 24)",
    )
    
    return p.parse_args()

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_output_path(output_dir: str, model_name: str) -> Path:
    """Construct output filename from model name."""
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    return Path(output_dir) / f"rq3_intervention_{model_short}.csv"


def load_checkpoint(output_path: Path) -> tuple[pd.DataFrame | None, set]:
    """Load existing checkpoint and return (dataframe, set of completed (lang, idx) pairs)."""
    if not output_path.exists():
        return None, set()

    df = pd.read_csv(output_path)
    completed_pairs = set(zip(df["probe_lang"], df["dataset_lang"])) if df is not None else set()
    print(f"✓ Loaded checkpoint: {completed_pairs} already done")
    return df, completed_pairs


def save_checkpoint(results: list[dict], output_path: Path):
    """Save results to CSV, merging with any existing checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame(results)

    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Deduplicate on (lang, original_index), keeping last
        df_combined = df_combined.drop_duplicates(
        subset=["probe_lang", "dataset_lang", "original_index"], keep="last"
        )
    else:
        df_combined = df_new

    df_combined.to_csv(output_path, index=False)
    return len(df_combined)


def load_activations_and_labels(filepath : str, lang):
    """loads activations and corresponding labels for specified language from h5py file and returns tuple of dictionary of activations per layer and corresponding labels as torch tensor

    Args:
        filepath (str): path to h5py file
        lang (str): any supported language (english, german, indonesian)
    """
    with h5py.File(filepath, "r") as f:
        activations = f[lang]["activations"][:]  # shape: (N, n_layers, hidden_dim)
        if "nat" in filepath:
            #indices = f[lang]["indices"][:]
            indices = [i for i in range(300)]
            df = pd.read_csv("results/rq1_generations_qwen3_14b_sycophancy_labeled.csv")
            df_lang = df[df["lang"] == lang]
                
            df_indexed = df_lang.set_index("original_index")
            assert df_lang["original_index"].is_unique, f"Duplicate original_index found for lang={lang}"
            
            labels = df_indexed.loc[indices, "sycophancy_label"].to_numpy()
        else:   
            labels = f[lang]["labels"][:]
            
    activations_by_layer = {
    layer_idx: t.from_numpy(activations[:, layer_idx, :].copy())
    for layer_idx in range(activations.shape[1])}
    
    return activations_by_layer, t.from_numpy(labels.copy())


def get_mm_direction(args,acts, labels):
    acts, labels = acts[24][:600, :].to(args.gpu), labels[:600].to(args.gpu)
        
    acts_nosyc = acts[labels == 1]
    acts_syc = acts[labels == 0]
    
    mean_nosyc = acts_nosyc.mean(dim=0)
    mean_syc = acts_syc.mean(dim=0)
    
    direction =  mean_syc - mean_nosyc

    return direction



    
    



def main():
    
    args = parse_args()
    device = f"cuda:{args.gpu}"
    PROBE_LAYER = args.intervention_layer
    
    # ── Output path ──
    output_path = get_output_path(args.output_dir, args.model)
    print(f"Output: {output_path}")

    # ── Load checkpoint ──
    _, completed_pairs = load_checkpoint(output_path)
    
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

    # ── Log config ──
    config_summary = {
        "model": args.model,
        "languages": args.languages,
        "n_samples": args.n_samples or "all",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    print("Config:", config_summary)
    
    buffer = []
    for probe_lang in args.languages:
        
        # get diff-in-means direction and scale
        acts, labels = load_activations_and_labels("activations/train_completions_avg.h5", probe_lang)
        
        direction = get_mm_direction(args, acts, labels)
        direction = direction.to(device=model.device, dtype=t.bfloat16)
        
        def make_intervention_hook(
            direction: Float[Tensor, " d_model"],
            scale: float,
        ) -> callable:
            """
            Create a forward hook that adds scale * direction to hidden states at fixed positions.
            This handles both plain-tensor and tuple outputs from transformer layers.
            """

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                if hidden_states.shape[1] == 1:
                    hidden_states[:, 0, :] += scale * direction

                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states

            return hook_fn
                
        hook = model.model.layers[PROBE_LAYER].register_forward_hook(make_intervention_hook(direction, args.scale))
        
        for data_lang in args.languages:
            
            if (probe_lang, data_lang) in completed_pairs:
                continue
            
            test_dataset = load_from_disk(f"datasets/{data_lang}_dataset")["test"]
            
            if args.n_samples:
                test_dataset  = test_dataset.select(range(args.n_samples))
                
            try:
                for i in tqdm(range(len(test_dataset)), desc=f"  {probe_lang} x {data_lang}"):
                    
                    example = test_dataset[i]
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

                    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                    
                    

                    # ── Generate ──
                    with t.no_grad():
                            outputs = model.generate(
                                **inputs,
                                do_sample=True,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                            )
                        
                            
                            
                            
                    # Extract only the new tokens
                    response = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                    )

                    buffer.append(
                        {
                            "original_index": i,
                            "probe_lang": probe_lang,
                            "dataset_lang": data_lang,
                            "prompt_used": prompt,
                            "prompt_english": prompt_english,
                            "response_raw": response,
                            "model": args.model,
                            "scale": args.scale,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                            "max_new_tokens": args.max_new_tokens,
                        }
                    )
                    if len(buffer) >= args.checkpoint_every:
                                total_saved = save_checkpoint(buffer, output_path)
                                tqdm.write(f"  ✓ Checkpoint: {total_saved} total rows saved")
                                buffer = []
            finally:
                hook.remove()
            
    if buffer:
        total_saved = save_checkpoint(buffer, output_path)
        print(f"\n✓ Final save: {total_saved} total rows → {output_path}")


if __name__ == "__main__":
    main()    
                        
                        
                    
                        
                        
                        

                

                
                
                
                
                
                
                
                
                
                
            
            
            