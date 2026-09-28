import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model_and_tok(model_name: str, device: int):
    """
    Load the model in float32 together with the tokenizer used for editing.

    The editing tokenizer pads on the right. For Qwen2 models, eos/pad/unk are set to
    ``<|endoftext|>``; for other models pad is set to eos.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    if "qwen" in model_name.lower():
        tok = AutoTokenizer.from_pretrained(
            model_name,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            unk_token="<|endoftext|>",
            trust_remote_code=True,
        )
    else:
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model.to(f"cuda:{device}")
    return model, tok


def load_generation_tok(model_name: str):
    """Tokenizer for batched generation (left padding, pad = eos)."""
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tok.pad_token_id = tok.eos_token_id
    return tok
