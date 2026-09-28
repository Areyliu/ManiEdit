from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

from .compute_leverage import get_chunk_positions_from_leverage
from .compute_z import compute_z
from .ManiEdit_hparams import ManiEditHyperParams
from .stats import get_cov


def apply_ManiEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ManiEditHyperParams,
    on_after_each_edit: Optional[Callable[[int, AutoModelForCausalLM], None]] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Sequentially edit ``requests`` ({"question": str, "answer": str}) into ``model`` in place.

    Each answer is split into chunks (Pivot Localization); every chunk is written into the
    down-projection of layer ``hparams.layers[-1]`` by the closed-form update

        Delta = R k^T P (P k k^T P + lambda Sigma)^{-1},

    after which the projector P is deflated along k (recursive null-space alignment).
    P starts from the identity at the beginning of the call and is carried across all
    requests, so the whole edit sequence must be passed in a single call.

    ``on_after_each_edit(num_edits_done, model)`` is called after every request.
    Returns (model, weights_copy) where weights_copy holds the original edited weight.
    """
    requests = deepcopy(requests)
    for r in requests:
        if hparams.answer_prepend_space_if_missing and r["answer"] and not r["answer"].startswith(" "):
            r["answer"] = " " + r["answer"]
    # for r in requests[:5]:
    #     print(f"ManiEdit request sample: [{r['question'][:80]!r}] -> [{r['answer'][:50]!r}]")

    device = f"cuda:{hparams.device}"
    layer = hparams.layers[-1]
    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    weight = nethook.get_parameter(model, weight_name)
    weights_copy = {weight_name: weight.detach().clone()}

    cov = get_cov(model, tok, hparams.rewrite_module_tmp.format(layer), hparams)
    in_dim = cov.shape[0]
    P = torch.eye(in_dim, device=device, dtype=torch.float32)
    n_keys = 0

    for edit_idx, request in enumerate(requests):
        question_ids = tok([request["question"]], return_tensors="pt", padding=True).to(device)["input_ids"]
        question_len = question_ids.size(1)
        target_ids = tok(request["answer"], return_tensors="pt").to(device)["input_ids"][0]
        if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
            target_ids = target_ids[1:]

        chunk_ranges = get_chunk_ranges(model, tok, request, len(target_ids), hparams)
        # print(f"[edit {edit_idx + 1}/{len(requests)}] answer_len={len(target_ids)} chunks={chunk_ranges}")

        full_ids = torch.cat([question_ids, target_ids.unsqueeze(0)], dim=1)
        full_tok = {"input_ids": full_ids, "attention_mask": torch.ones_like(full_ids)}

        for chunk_start, chunk_end in chunk_ranges:
            context_ids = torch.cat([question_ids, target_ids[:chunk_start].unsqueeze(0)], dim=1)
            idx, z_star = compute_z(
                model, tok, context_ids, target_ids[chunk_start:chunk_end], layer, hparams
            )
            if idx >= full_ids.size(1):
                print(f"WARN: key position {idx} outside the sequence, skipping chunk")
                continue

            # Key k and residual R = z* - z at the pivot position
            with torch.no_grad():
                with nethook.Trace(
                    module=model,
                    layer=hparams.rewrite_module_tmp.format(layer),
                    retain_input=True,
                    retain_output=False,
                    detach=True,
                    clone=True,
                ) as tr:
                    model(**full_tok)
            layer_in = tr.input[0] if isinstance(tr.input, tuple) else tr.input
            k = layer_in[0, idx].unsqueeze(1)  # (in_dim, 1)
            resid = (z_star - get_layer_output(model, full_tok, idx, hparams, layer)).unsqueeze(0)  # (1, hidden)

            # Closed form: solve (P k k^T P + lambda Sigma) X = P k R, Delta = X^T
            A = P @ (k @ k.T) @ P + hparams.mom2_update_weight * cov
            B = P @ k @ resid
            upd_matrix = upd_matrix_match_shape(torch.linalg.solve(A, B), weight.shape)
            with torch.no_grad():
                weight[...] += upd_matrix.float()

            # Recursive null-space alignment: deflate P along k
            with torch.no_grad():
                Pk = P @ k
                ktPk = (k.T @ Pk).squeeze().item()
                P_update = (Pk @ Pk.T) / (ktPk + hparams.proj_epsilon)
                if hparams.leaky_eta > 0:
                    P = (1.0 - hparams.leaky_eta) * (P - P_update) + hparams.leaky_eta * torch.eye(
                        in_dim, device=device, dtype=torch.float32
                    )
                else:
                    P = P - P_update
            n_keys += 1
            # print(f"    keys={n_keys} free_dim={P.trace().item():.0f} ktPk={ktPk:.4f}")
            del A, B, upd_matrix, Pk, P_update, layer_in, tr

        if on_after_each_edit is not None:
            on_after_each_edit(edit_idx + 1, model)

    return model, weights_copy


def get_chunk_ranges(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    n_tokens: int,
    hparams: ManiEditHyperParams,
) -> List[Tuple[int, int]]:
    """Half-open answer-token ranges [start, end) of the chunks."""
    if hparams.chunking == "leverage":
        positions = get_chunk_positions_from_leverage(
            model, tok, request["question"], request["answer"], n_tokens, hparams
        )
        boundaries = [0] + [p for p in positions if 0 < p < n_tokens] + [n_tokens]
        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    if hparams.chunking == "uniform":
        chunk_ranges, s = [], 0
        while s < n_tokens:
            e = min(s + hparams.window_size, n_tokens)
            chunk_ranges.append((s, e))
            s += hparams.window_size - hparams.overlap
        return chunk_ranges
    raise ValueError(f"Unknown chunking: {hparams.chunking}")


def get_layer_output(model, contexts_tok: Dict, idx: int, hparams: ManiEditHyperParams, layer: int) -> torch.Tensor:
    """Output of transformer block ``layer`` at position ``idx``."""
    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=False,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            model(**contexts_tok)
    out = tr.output[0] if isinstance(tr.output, tuple) else tr.output
    return out[0, idx]


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError("Update matrix computed by ManiEdit does not match original weight shape.")
