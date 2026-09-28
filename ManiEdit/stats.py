import torch

from rome.layer_stats import layer_stats
from util.globals import *

from .ManiEdit_hparams import ManiEditHyperParams

COV_CACHE = {}


def get_cov(model, tok, layer_name: str, hparams: ManiEditHyperParams) -> torch.Tensor:
    """
    Second-moment matrix Sigma = E[k k^T] of the keys entering ``layer_name``,
    loaded from (or computed into) the stats cache. Kept on the editing device.
    """
    key = (hparams.model_name, layer_name)
    if key not in COV_CACHE:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir or STATS_DIR,
            hparams.mom2_dataset,
            to_collect=["mom2"],
            model_name=hparams.stats_model_name or hparams.model_name.rstrip("/").split("/")[-1],
            sample_size=hparams.mom2_n_samples,
            precision=hparams.mom2_dtype,
            batch_tokens=hparams.mom2_batch_tokens,
            device=f"cuda:{hparams.device}",
            dataset_path=hparams.mom2_dataset_path,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to(f"cuda:{hparams.device}")
    return COV_CACHE[key]
