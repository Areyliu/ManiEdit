# ManiEdit

- Code for `ManiEdit: Sequential Unstructured Knowledge Editing for Language Models from a Manifold Perspective` .

## Requirements

**At least one 48G GPU** .

```bash
pip install -r requirements.txt
```

## Benchmark datasets

### Editing benchmarks

Place the official JSON files at the paths below:


| Dataset          | Official source                                                                                                                                           | Expected path                        |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| UnKEBench        | [TrustedLLM/UnKE](https://github.com/TrustedLLM/UnKE); [dataset file](https://github.com/TrustedLLM/UnKE/blob/main/data/final_data_v3.json)               | `data/UnKE/final_data_v3.json`       |
| AKEW CounterFact | [bobxwu/AKEW](https://github.com/bobxwu/AKEW); [dataset file](https://github.com/bobxwu/AKEW/blob/master/datasets/CounterFact.json)                       | `data/AKEW/CounterFact.json`         |
| AKEW MQuAKE-CF   | [bobxwu/AKEW](https://github.com/bobxwu/AKEW); [dataset file](https://github.com/bobxwu/AKEW/blob/master/datasets/MQuAKE-CF.json)                         | `data/AKEW/MQuAKE-CF.json`           |
| EditEverything   | [jianghoucheng/AnyEdit](https://github.com/jianghoucheng/AnyEdit); [dataset file](https://github.com/jianghoucheng/AnyEdit/blob/main/data/editevery.json) | `data/EditEverything/editevery.json` |


### General-capability benchmarks

Download GLUE from [GLUE on Hugging Face](https://huggingface.co/datasets/nyu-mll/glue) and MMLU from [MMLU on Hugging Face](https://huggingface.co/datasets/cais/mmlu).


| Dataset | Tasks / split                                              | Expected path                                        |
| ------- | ---------------------------------------------------------- | ---------------------------------------------------- |
| GLUE    | `cola`, `sst2`, `mrpc`, `rte`, `mnli_matched` (validation) | `data/GLUE/<task>/validation-00000-of-00001.parquet` |
| MMLU    | All 57 subjects (test)                                     | `data/MMLU/<subject>/test-00000-of-00001.parquet`    |


## Model weights

The evaluation scripts pass `--model_name` to Transformers `from_pretrained`. A Hugging Face model ID is downloaded automatically and cached by default in `~/.cache/huggingface/hub` on Linux, or `C:\Users\<username>\.cache\huggingface\hub` on Windows. Set `HF_HOME` or `HF_HUB_CACHE` before running the script to use a different disk location; for example:

```bash
export HF_HOME=/path/to/huggingface-cache
```

You can also pass a local model directory as `--model_name`.

`meta-llama/Meta-Llama-3-8B-Instruct` is gated. Request access on the [model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), then log in before the first download.

## Precomputed second-moment statistics

Precomputed second-moment matrices for the two supported models are hosted in the [ManiEdit-stats dataset](https://huggingface.co/datasets/Areyliu/ManiEdit-stats):

- [Qwen2.5-7B-Instruct statistics](https://huggingface.co/datasets/Areyliu/ManiEdit-stats/blob/main/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.8.mlp.down_proj_float32_mom2_100000.npz)
- [Meta-Llama-3-8B-Instruct statistics](https://huggingface.co/datasets/Areyliu/ManiEdit-stats/blob/main/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.8.mlp.down_proj_float32_mom2_100000.npz)

Download the cached statistics into the default `data/stats/` directory before running the experiments:

```bash
hf download Areyliu/ManiEdit-stats --repo-type dataset --local-dir data/stats
```

The command creates the paths expected by the code:

```
data/stats/
|__ Qwen2.5-7B-Instruct/
|   |__ wikipedia_stats/
|       |__ model.layers.8.mlp.down_proj_float32_mom2_100000.npz
|__ Meta-Llama-3-8B-Instruct/
    |__ wikipedia_stats/
        |__ model.layers.8.mlp.down_proj_float32_mom2_100000.npz
```

If a second-moment matrix is missing, the evaluation script computes it automatically at runtime and saves it to the expected statistics path.

## Quick Start

### 1. Sequential unstructured editing on UnKE / AKEW / EditEverything

```
python -m experiments.evaluate --alg_name=ManiEdit --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=Qwen2.5-7B-Instruct.json --ds_name=unke --dataset_size_limit=500
```

- `--model_name`: Hugging Face model id or local path (`Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Meta-Llama-3-8B-Instruct`).
- `--hparams_fname`: hyperparameter file under `hparams/ManiEdit/`.
- `--ds_name`: `unke`, `akew_counterfact`, `akew_mquake` or `editevery`.
- `--dataset_size_limit`: number of samples edited sequentially.
- `--chunking=uniform`: fixed windows of `window_size` answer tokens instead of Pivot Localization (the *w/o Pivot* ablation).
- `--per_category` (EditEverything): edit every category as its own sequence, starting from the unedited model.

To reproduce the main results (both models, all four benchmarks):

```
bash scripts/run_main_results.sh <gpu> [qwen|llama|all]
```

Results are stored at `results/ManiEdit/run_<run_id>`:

```bash
results/
|__ ManiEdit/
    |__ run_<run_id>/
        |__ params.json
        |__ results.json    # targets and predictions (original / paraphrase / sub-questions)
        |__ summary.json    # BLEU, ROUGE-1/2/L and BERT Score
```

### 2. General capability during editing

```
python -m experiments.evaluate_general --alg_name=ManiEdit --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=Qwen2.5-7B-Instruct.json --dataset_size_limit=500 --eval_every=100 \
    --tasks sst2 cola mrpc rte mnli_matched mmlu
```

To reproduce the general-capability results:

```
bash scripts/run_general_capability.sh <gpu>
```

The results are written to `results/ManiEdit_general/run_<run_id>/general_eval.jsonl`.

### 3. Summarize the results

```
python -m experiments.summarize --dir_name=ManiEdit --runs=run_<run1>,run_<run2>
```

## License

This repository is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).


