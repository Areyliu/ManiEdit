"""
Pivot Localization: split an answer at its highest-leverage semantic pivot.

The leverage score of a key k is k^T (Sigma + eps I)^{-1} k. The answer is split once,
at the content-word position with the highest leverage, into [0, pos) and [pos, n).
"""
import string
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

from .ManiEdit_hparams import ManiEditHyperParams
from .stats import get_cov

PUNCT_AND_COMBOS = {
    ".", ",", "!", "?", "'", '"', "-", ";", ":", "(", ")", "[", "]", "{", "}",
    "'s", "''", "``", "...", "–", "—", "/", "\\", "*", "@", "#", "$", "%", "&",
    "!,", ".,", ",.", ".\"", "!\"", ",\"", "'.\"", "\".", "!.", "?\"", "\"?", ",'", "'.", "-.", ",-", "!.", ".,", "?,",
    "<|im_end|>", "<|endoftext|>",
}
BPE_SUBWORDS = {
    "_M", "_B", "_N", "_E", "_S", "_T", "_P", "_K", "_L", "_R", "_C", "_A",
    "-N", "-m", "-n", "-M", "-bar", "-factor", "-sized", "-edge", "-after",
    "/c", "/T", "/n", "/m", "/M", "/N", "/C", "(n", "(m", "(M", "(N", "(c", "(T",
    "²", "³", "μ", "±", "·", "−", "×", "÷",
}
UNIT_TOKENS = {
    "kg", "g", "m", "s", "A", "K", "mol", "cd", "Hz", "W", "J", "V", "Pa", "N", "F", "H", "Ω", "T",
    "cm", "mm", "nm", "μm", "km", "mg", "mL", "L", "amu", "eV", "MeV", "GeV", "Js",
    "rad", "sr", "C", "Bq", "Gy", "Sv", "lm", "lx",
}
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "have", "has", "had", "do", "does", "did", "that", "which", "who", "it", "its", "as", "by",
}


def is_content_word(token_str: str) -> bool:
    """Filter out stopwords, punctuation, BPE subwords and units."""
    t = token_str.strip()
    if not t or len(t) < 2:
        return False
    if t in PUNCT_AND_COMBOS or t in BPE_SUBWORDS or t in UNIT_TOKENS:
        return False
    if all(c in string.punctuation + " \t\n\"'" for c in t):
        return False
    return t.lower() not in STOPWORDS


@torch.no_grad()
def compute_leverage_per_position(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    answer: str,
    cov: torch.Tensor,
    rewrite_layer: str,
    device: str,
    eps: float = 1e-4,
) -> List[Tuple[int, float, str]]:
    """
    For every answer position i in 1..n-1, the leverage of the key that predicts token i
    (the input of ``rewrite_layer`` at the previous position). Returns [(i, leverage, token)].
    """
    input_tok = tok([question], return_tensors="pt", padding=True).to(device)
    target_ids = tok(answer, return_tensors="pt").to(device)["input_ids"][0]
    if target_ids[0] in (tok.bos_token_id, tok.unk_token_id):
        target_ids = target_ids[1:]

    q_len = input_tok["input_ids"].size(1)
    full_ids = torch.cat([input_tok["input_ids"], target_ids.unsqueeze(0)], dim=1).to(device)

    with nethook.Trace(
        module=model,
        layer=rewrite_layer,
        retain_input=True,
        retain_output=False,
        detach=True,
        clone=True,
    ) as tr:
        model(full_ids)

    layer_in = tr.input
    layer_in = layer_in[0] if isinstance(layer_in, tuple) else layer_in
    positions = [i for i in range(1, len(target_ids)) if q_len + i - 1 < layer_in.shape[1]]
    if not positions:
        return []
    K = layer_in[0, [q_len + i - 1 for i in positions]].float().T  # (d, n-1)
    leverage = (K * torch.linalg.lu_solve(*_lu_factor(cov, eps), K)).sum(0).tolist()
    return [
        (i, lev, tok.decode([target_ids[i].item()]).strip()) for i, lev in zip(positions, leverage)
    ]


_LU_CACHE = {}


def _lu_factor(cov: torch.Tensor, eps: float):
    """LU factorization of Sigma + eps I, computed once per Sigma."""
    key = (cov.data_ptr(), cov.shape[0], eps)
    if key not in _LU_CACHE:
        A = cov.float() + eps * torch.eye(cov.shape[0], device=cov.device, dtype=torch.float32)
        _LU_CACHE[key] = torch.linalg.lu_factor(A)
    return _LU_CACHE[key]


def _select_chunk_position(positions_sorted_by_leverage: List[Tuple[int, float, str]], n_tokens: int) -> int:
    """
    The first of the top-5 positions with 10 < pos < n_tokens - 9 (so that both chunks keep
    about ten tokens); if none qualifies, the top-1 position.
    """
    if not positions_sorted_by_leverage:
        return n_tokens // 2
    low, high = 10, n_tokens - 9
    for pos, _, _ in positions_sorted_by_leverage[:5]:
        if low < pos < high and 0 < pos < n_tokens:
            return pos
    return positions_sorted_by_leverage[0][0]


def get_chunk_positions_from_leverage(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    answer: str,
    n_tokens: int,
    hparams: ManiEditHyperParams,
) -> List[int]:
    """Split position(s) of the answer: [pos] -> chunks [0, pos) and [pos, n_tokens)."""
    rewrite_layer = hparams.rewrite_module_tmp.format(hparams.layers[-1])
    cov = get_cov(model, tok, rewrite_layer, hparams)
    lev_list = compute_leverage_per_position(
        model, tok, question, answer, cov, rewrite_layer, f"cuda:{hparams.device}"
    )
    content_lev = [(p, l, t) for p, l, t in lev_list if is_content_word(t)]
    content_lev.sort(key=lambda x: -x[1])

    if content_lev:
        selected_pos = _select_chunk_position(content_lev, n_tokens)
    elif lev_list:
        # No content word: fall back to the first five answer positions.
        selected_pos = _select_chunk_position(lev_list[:5], n_tokens)
    else:
        selected_pos = n_tokens // 2

    return [selected_pos] if 0 < selected_pos < n_tokens else [n_tokens // 2]
