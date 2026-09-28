from dataclasses import dataclass
from typing import List, Literal, Optional

from util.hparams import HyperParams


@dataclass
class ManiEditHyperParams(HyperParams):
    # Model
    model_name: str
    layers: List[int]  
    device: int

    # Target representation z* (per chunk)
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float

    # Closed-form update: lambda in ||Delta Sigma^{1/2}||^2
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics (Sigma)
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # Chunking: "leverage" (Pivot Localization) or "uniform" (fixed windows, w/o Pivot ablation)
    chunking: Literal["leverage", "uniform"] = "leverage"
    window_size: int = 50
    overlap: int = 0

    # Recursive null-space alignment:
    # P <- (1 - leaky_eta) * (P - P k k^T P / (k^T P k + proj_epsilon)) + leaky_eta * I
    proj_epsilon: float = 1e-6
    leaky_eta: float = 0.0

    v_early_stop_loss: float = 1e-2
    mom2_dataset_path: Optional[str] = None
    mom2_batch_tokens: Optional[int] = None
    stats_dir: Optional[str] = None  # default: STATS_DIR in globals.yml
    stats_model_name: Optional[str] = None  # default: last component of model_name
    answer_prepend_space_if_missing: bool = True
