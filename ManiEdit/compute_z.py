from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

from .ManiEdit_hparams import ManiEditHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_ids: torch.Tensor,
    chunk_target_ids: torch.Tensor,
    z_layer: int,
    hparams: ManiEditHyperParams,
) -> Tuple[int, torch.Tensor]:
    """
    Optimize the target representation z* for one chunk.

    ``context_ids`` holds the question and all answer tokens before the chunk, and
    ``chunk_target_ids`` the chunk itself. A vector delta is added to the output of layer
    ``z_layer`` at the position that predicts the first token of the chunk (the last
    context token) and optimized so that the model generates the chunk.

    Returns (lookup_idx, z*), where lookup_idx is that position in the full sequence.
    """
    device = f"cuda:{hparams.device}"
    lm_w = nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    input_ids = torch.cat([context_ids, chunk_target_ids[:-1].unsqueeze(0)], dim=1)
    rewriting_targets = torch.tensor(-100, device=device).repeat(1, input_ids.size(1))
    ex_len = input_ids.size(1)
    rewriting_targets[0, ex_len - len(chunk_target_ids) : ex_len] = chunk_target_ids
    lookup_idx = ex_len - len(chunk_target_ids)
    loss_layer = max(hparams.v_loss_layer, z_layer)

    hidden_size = getattr(model.config, "n_embd", None) or getattr(model.config, "hidden_size")
    delta = torch.zeros((hidden_size,), requires_grad=True, device=device)
    target_init = None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.layer_module_tmp.format(z_layer):
            out = cur_out[0] if isinstance(cur_out, (tuple, list)) else cur_out
            if out.dim() == 2:
                if target_init is None:
                    target_init = out[lookup_idx].detach().clone()
                out[lookup_idx] += delta
            else:
                if target_init is None:
                    target_init = out[0, lookup_idx].detach().clone()
                out[0, lookup_idx] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(z_layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            model(input_ids)

        output = tr[hparams.layer_module_tmp.format(loss_layer)].output
        output = output[0] if isinstance(output, tuple) else output
        if output.shape[1] != rewriting_targets.shape[1]:
            output = output.transpose(0, 1)

        log_probs = torch.log_softmax(
            ln_f(output) @ lm_w.to(output.device) + lm_b.to(output.device), dim=2
        )
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll_loss = (-(loss * mask).sum(1) / chunk_target_ids.size(0)).mean()
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        loss = nll_loss + weight_decay
        if loss < hparams.v_early_stop_loss or it == hparams.v_num_grad_steps - 1:
            break
        loss.backward()
        opt.step()

        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    # print(f"    z*: {it + 1} steps, loss {loss.item():.4f} (nll {nll_loss.item():.4f})")
    return lookup_idx, target_init + delta
