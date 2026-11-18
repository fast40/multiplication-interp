"""Shared helpers: model loading and prompt construction."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B"
FEWSHOT = "3*4=12\n15*6=90\n21*13=273\n8*7=56\n"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def prompt(a, b):
    return f"{FEWSHOT}{a}*{b}="


def problems(rng, n, max_prod=999):
    """Random (a, b) with a in 10..99, b in 2..99, single-token product."""
    out = []
    while len(out) < n:
        a = int(rng.integers(10, 100)); b = int(rng.integers(2, 100))
        if a * b <= max_prod:
            out.append((a, b))
    return out


def answer_id(tok, n):
    ids = tok(str(n), add_special_tokens=False).input_ids
    assert len(ids) == 1, n
    return ids[0]


def layer_out(o):
    return o[0] if isinstance(o, tuple) else o
