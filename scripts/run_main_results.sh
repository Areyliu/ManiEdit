#!/usr/bin/env bash
# Main results of ManiEdit (Table: UnKE / AKEW CounterFact / AKEW MQuAKE / EditEverything).
#
# Usage (from the repository root):
#   bash scripts/run_main_results.sh <gpu> [qwen|llama|all]
set -euo pipefail

GPU=${1:-0}
WHICH=${2:-all}
QWEN=${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}
LLAMA=${LLAMA_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}

run() {  # model hparams ds_name n early_stop [extra args]
    local model=$1 hp=$2 ds=$3 n=$4 es=$5
    shift 5
    python -m experiments.evaluate --alg_name=ManiEdit --model_name="$model" --hparams_fname="$hp" \
        --ds_name="$ds" --dataset_size_limit="$n" --v_early_stop_loss="$es" --device="$GPU" "$@"
}

if [[ $WHICH == qwen || $WHICH == all ]]; then
    run "$QWEN" Qwen2.5-7B-Instruct.json unke             500 0.05
    run "$QWEN" Qwen2.5-7B-Instruct.json akew_counterfact 500 0.01
    run "$QWEN" Qwen2.5-7B-Instruct.json akew_mquake      354 0.05
    run "$QWEN" Qwen2.5-7B-Instruct.json editevery        552 0.01 --per_category
fi

if [[ $WHICH == llama || $WHICH == all ]]; then
    run "$LLAMA" Llama3-8B-Instruct.json unke             500 0.01
    run "$LLAMA" Llama3-8B-Instruct.json akew_counterfact 500 0.05
    run "$LLAMA" Llama3-8B-Instruct.json akew_mquake      354 0.05
    run "$LLAMA" Llama3-8B-Instruct.json editevery        552 0.01 --per_category
fi
