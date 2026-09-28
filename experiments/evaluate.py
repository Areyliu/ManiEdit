"""
Sequential unstructured knowledge editing on UnKE, AKEW (CounterFact / MQuAKE) and
EditEverything, followed by generation and metric computation.

    python -m experiments.evaluate --alg_name=ManiEdit --model_name=Qwen/Qwen2.5-7B-Instruct \
        --hparams_fname=Qwen2.5-7B-Instruct.json --ds_name=unke --dataset_size_limit=500
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from dsets import DS_DICT
from dsets.chat import strip_end_of_turn
from ManiEdit import ManiEditHyperParams, apply_ManiEdit_to_model
from util import nethook
from util.globals import *
from util.model import load_generation_tok, load_model_and_tok, seed_everything

from .summarize import DEFAULT_BERT_MODEL, compute_metrics

ALG_DICT = {
    "ManiEdit": (ManiEditHyperParams, apply_ManiEdit_to_model),
}


def get_run_dir(dir_name: str) -> Path:
    alg_dir = RESULTS_DIR / dir_name
    if alg_dir.exists():
        id_list = [int(str(x).split("_")[-1]) for x in alg_dir.iterdir() if str(x).split("_")[-1].isnumeric()]
        run_id = 0 if not id_list else max(id_list) + 1
    else:
        run_id = 0
    run_dir = alg_dir / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    return run_dir


def load_hparams(alg_name: str, hparams_fname: str, model_name: str, device: int):
    params_class, apply_algo = ALG_DICT[alg_name]
    hparams = params_class.from_json(HPARAMS_DIR / alg_name / hparams_fname)
    hparams.model_name = model_name
    if device is not None:
        hparams.device = device
    return hparams, apply_algo


@torch.no_grad()
def generate(model, gen_tok, prompts: List[str], device: str) -> List[str]:
    inputs = gen_tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        temperature=0.001,
        max_new_tokens=512,
    )
    out = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    return gen_tok.batch_decode(out, skip_special_tokens=True)


def generate_predictions(model, gen_tok, record: Dict, ds_name: str, device: str) -> Dict:
    if ds_name == "editevery":
        record["original_prediction"] = generate(model, gen_tok, [record["question"]], device)[0]
    else:
        record["original_prediction"], record["para_prediction"] = generate(
            model, gen_tok, [record["question"], record["para_question"]], device
        )
        record["sub_pred"] = generate(model, gen_tok, record["sub_question"], device)
    record["answer"] = strip_end_of_turn(record["answer"])
    return record


def main(args):
    seed_everything(42)
    run_dir = get_run_dir(args.alg_name)
    hparams, apply_algo = load_hparams(args.alg_name, args.hparams_fname, args.model_name, args.device)
    if args.chunking is not None:
        hparams.chunking = args.chunking
    if args.v_early_stop_loss is not None:
        hparams.v_early_stop_loss = args.v_early_stop_loss
    with open(run_dir / "params.json", "w") as f:
        json.dump({"args": vars(args), "ds_name": args.ds_name, "hparams": hparams.to_dict()}, f, indent=2)

    model, tok = load_model_and_tok(hparams.model_name, hparams.device)
    gen_tok = load_generation_tok(hparams.model_name)
    device = f"cuda:{hparams.device}"

    ds = DS_DICT[args.ds_name](args.data_dir, hparams.model_name, size=args.dataset_size_limit)
    records = [dict(ds[i]) for i in range(len(ds))]
    print(f"Loaded {len(records)} samples of {args.ds_name}")

    if args.ds_name == "editevery" and args.per_category:
        # Each category is edited sequentially from the unedited model.
        groups: Dict[str, List[Dict]] = {}
        for r in records:
            groups.setdefault(r["category"], []).append(r)
        edit_groups = list(groups.values())
    else:
        edit_groups = [records]

    results = []
    for group in edit_groups:
        _, weights_copy = apply_algo(model, tok, group, hparams)
        for r in tqdm(group, desc="Generate"):
            results.append(generate_predictions(model, gen_tok, r, args.ds_name, device))
        if len(edit_groups) > 1:
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to(device)

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if not args.skip_summarize:
        del model
        torch.cuda.empty_cache()
        metrics = compute_metrics(results, args.ds_name, args.bert_model, hparams.device)
        with open(run_dir / "summary.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_name", choices=list(ALG_DICT), default="ManiEdit")
    parser.add_argument("--model_name", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--hparams_fname", required=True, help="File under hparams/<alg_name>/.")
    parser.add_argument("--ds_name", choices=list(DS_DICT), default="unke")
    parser.add_argument("--dataset_size_limit", type=int, default=None, help="Number of edits (first N samples).")
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--device", type=int, default=None, help="CUDA device index (overrides hparams).")
    parser.add_argument(
        "--chunking",
        choices=["leverage", "uniform"],
        default=None,
        help="Override hparams.chunking; 'uniform' gives the w/o Pivot ablation.",
    )
    parser.add_argument(
        "--v_early_stop_loss",
        type=float,
        default=None,
        help="Override hparams.v_early_stop_loss (z* optimization stops once the loss falls below it).",
    )
    parser.add_argument(
        "--per_category",
        action="store_true",
        help="EditEverything only: edit each category as a separate sequence starting from the unedited model.",
    )
    parser.add_argument("--bert_model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--skip_summarize", action="store_true")
    main(parser.parse_args())
