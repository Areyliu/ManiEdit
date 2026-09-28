"""
General capability during sequential editing: edit AKEW CounterFact samples one after
another and evaluate zero-shot GLUE / MMLU accuracy every ``--eval_every`` edits.

    python -m experiments.evaluate_general --alg_name=ManiEdit --model_name=Qwen/Qwen2.5-7B-Instruct \
        --hparams_fname=Qwen2.5-7B-Instruct.json --dataset_size_limit=500 --eval_every=100 \
        --tasks sst2 cola mrpc
"""
import argparse
import json
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer

from dsets import DS_DICT
from util.globals import *
from util.model import load_model_and_tok, seed_everything

from .evaluate import get_run_dir, load_hparams
from .py.eval_utils_glue import GLUE_TASKS, evaluate_glue_task, load_glue_split
from .py.eval_utils_mmlu import MMLU_SUBJECTS, evaluate_mmlu, load_mmlu_subject


def main(args):
    seed_everything(42)
    run_dir = get_run_dir(args.alg_name + "_general")
    hparams, apply_algo = load_hparams(args.alg_name, args.hparams_fname, args.model_name, args.device)
    if args.chunking is not None:
        hparams.chunking = args.chunking
    with open(run_dir / "params.json", "w") as f:
        json.dump({"args": vars(args), "hparams": hparams.to_dict()}, f, indent=2)

    model, tok = load_model_and_tok(hparams.model_name, hparams.device)
    model.eval()
    device = torch.device(f"cuda:{hparams.device}")

    # Evaluation tokenizer: the model's own chat template, no padding needed (batch size 1)
    eval_tok = AutoTokenizer.from_pretrained(hparams.model_name, trust_remote_code=True)
    if eval_tok.pad_token_id is None:
        eval_tok.pad_token_id = eval_tok.eos_token_id

    glue_data = {t: load_glue_split(t, args.glue_dir) for t in args.tasks if t in GLUE_TASKS}
    mmlu_data = (
        {s: load_mmlu_subject(s, args.mmlu_dir) for s in MMLU_SUBJECTS} if "mmlu" in args.tasks else None
    )

    ds = DS_DICT[args.ds_name](args.data_dir, hparams.model_name, size=args.dataset_size_limit)
    requests = [dict(ds[i]) for i in range(len(ds))]
    log_path = run_dir / "general_eval.jsonl"

    def evaluate_and_log(num_edits: int):
        record = {"cumulative_edits": num_edits, "time": datetime.now(timezone.utc).isoformat()}
        for task, data in glue_data.items():
            record[task] = evaluate_glue_task(
                model, eval_tok, task, data, device, args.gen_max_new_tokens, args.max_samples
            )
        if mmlu_data is not None:
            record["mmlu"] = evaluate_mmlu(
                model, eval_tok, mmlu_data, device, args.gen_max_new_tokens, args.max_samples
            )
        accs = {k: round(v["accuracy"], 4) for k, v in record.items() if isinstance(v, dict)}
        print(f"[edits={num_edits}] accuracy: {accs}", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def on_after_each_edit(num_edits: int, _model):
        if num_edits % args.eval_every == 0 or num_edits == len(requests):
            evaluate_and_log(num_edits)

    if not args.skip_baseline:
        evaluate_and_log(0)
    apply_algo(model, tok, requests, hparams, on_after_each_edit=on_after_each_edit)
    print(f"Results written to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_name", default="ManiEdit")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--hparams_fname", required=True)
    parser.add_argument("--ds_name", choices=list(DS_DICT), default="akew_counterfact",
                        help="Edits applied between evaluations.")
    parser.add_argument("--dataset_size_limit", type=int, default=500)
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--chunking", choices=["leverage", "uniform"], default=None)
    parser.add_argument("--tasks", nargs="+", default=["sst2", "cola", "mrpc"],
                        choices=list(GLUE_TASKS) + ["mmlu"])
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--skip_baseline", action="store_true", help="Skip the evaluation before any edit.")
    parser.add_argument("--gen_max_new_tokens", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None, help="Evaluate only the first N samples per task/subject.")
    parser.add_argument("--glue_dir", default=str(DATA_DIR / "GLUE"),
                        help="Local GLUE parquet files; falls back to nyu-mll/glue on the Hub.")
    parser.add_argument("--mmlu_dir", default=str(DATA_DIR / "MMLU"),
                        help="Local MMLU parquet files; falls back to cais/mmlu on the Hub.")
    main(parser.parse_args())
