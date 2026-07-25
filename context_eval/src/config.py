# src/config.py

import os

# List of model identifiers for your evaluation pipeline.
ALL_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4",
    "mistral/mistral-large-latest",
    # Add more as needed, using OpenRouter/your provider's IDs
]

# Load API keys from environment variables (set via .env or shell)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Centralized access for dispatching in utils
MODEL_API_KEYS = {
    "openai": OPENAI_API_KEY,
    "anthropic": ANTHROPIC_API_KEY,
    "mistral": MISTRAL_API_KEY,
}

# Optionally: Default settings
DEFAULT_INPUT_CSV = "eval_dataset/evaluation_prompts.csv"
DEFAULT_OUTPUT_CSV = "eval_dataset/model_responses.csv"
