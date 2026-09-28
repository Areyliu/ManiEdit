from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats, e.g.

        python -m rome.layer_stats --model_name Qwen/Qwen2.5-7B-Instruct \
            --layers 8 --dataset_path data/mom2_datasets/wikipedia
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", required=True)
    aa("--stats_model_name", default=None,
       help="Folder name under stats_dir (default: last component of --model_name).")
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--dataset_path", default=None, help="Local copy of the dataset saved with `save_to_disk`.")
    aa("--layers", default=[8], type=lambda x: list(map(int, x.split(","))))
    aa("--layer_template", default="model.layers.{}.mlp.down_proj")
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=str(STATS_DIR))
    aa("--device", default=0, type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model = model.eval().to(f"cuda:{args.device}")
    set_requires_grad(False, model)

    for layer_num in args.layers:
        layer_name = args.layer_template.format(layer_num)
        print(
            f"Computing stats for {layer_name} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}.'
        )
        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            model_name=args.stats_model_name or args.model_name.rstrip("/").split("/")[-1],
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            device=f"cuda:{args.device}",
            dataset_path=args.dataset_path,
        )


def _max_positions(model) -> int:
    cfg = model.config
    if hasattr(cfg, "n_positions"):
        npos = cfg.n_positions
    elif hasattr(cfg, "max_sequence_length"):
        npos = cfg.max_sequence_length
    elif hasattr(cfg, "max_position_embeddings"):
        npos = cfg.max_position_embeddings
    elif hasattr(cfg, "seq_length"):
        npos = cfg.seq_length
    else:
        raise NotImplementedError
    model_type = getattr(cfg, "model_type", "")
    if "mistral" in model_type:
        npos = getattr(cfg, "sliding_window", None) or 4096
    if "qwen2" in model_type:
        npos = 4096
    return npos


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    progress=tqdm,
    force_recompute=False,
    device="cuda:0",
    dataset_path=None,
):
    """
    Load the cached statistics of the inputs to ``layer_name`` (the keys of the edited
    projection), or compute them over ``sample_size`` texts of ``ds_name`` and cache them to

        {stats_dir}/{model_name}/{ds_name}_stats/{layer_name}_{precision}_{stats}{suffix}.npz
    """

    def get_ds():
        if dataset_path is not None and Path(dataset_path).exists():
            raw_ds = load_from_disk(str(dataset_path))
            if not isinstance(raw_ds, dict):
                raw_ds = {"train": raw_ds}
        else:
            raw_ds = load_dataset(
                ds_name,
                dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
            )
        maxlen = _max_positions(model)
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    npos = _max_positions(model)
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    # ``batch_tokens`` controls tokenization and batching, but is intentionally omitted
    # from the public cache filename so downloaded statistics use one canonical name.
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if model_name is None:
        model_name = model.config._name_or_path.rstrip("/").split("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    if filename.exists() and not force_recompute:
        print(f"Loading cached statistics from {filename}")
    else:
        print(f"Computing statistics, will be cached at {filename}")
    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                with Trace(model, layer_name, retain_input=True, retain_output=False, stop=True) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


if __name__ == "__main__":
    main()
