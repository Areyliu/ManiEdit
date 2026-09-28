from .akew import AKEWDataset
from .editevery import EditEveryDataset
from .unke import UnKEDataset

DS_DICT = {
    "unke": UnKEDataset,
    "akew_counterfact": lambda data_dir, model_name, size=None: AKEWDataset(
        data_dir, "counterfact", model_name, size
    ),
    "akew_mquake": lambda data_dir, model_name, size=None: AKEWDataset(
        data_dir, "mquake", model_name, size
    ),
    "editevery": EditEveryDataset,
}
