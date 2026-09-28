#!/usr/bin/env bash
# General-capability results reported for Qwen2.5-7B-Instruct:
# 500 sequential AKEW CounterFact edits, evaluated every 100 edits on six tasks.
#
# Usage (from the repository root):
#   bash scripts/run_general_capability.sh <gpu>
set -euo pipefail

GPU=${1:-0}
QWEN=${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}

python -m experiments.evaluate_general \
    --alg_name=ManiEdit \
    --model_name="$QWEN" \
    --hparams_fname=Qwen2.5-7B-Instruct.json \
    --ds_name=akew_counterfact \
    --dataset_size_limit=500 \
    --eval_every=100 \
    --tasks sst2 cola mrpc rte mnli_matched mmlu \
    --device="$GPU"
