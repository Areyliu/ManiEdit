import json
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from .chat import end_of_turn, model_family, wrap_question


class UnKEDataset(Dataset):
    """
    UnKEBench (https://github.com/TrustedLLM/UnKE). Expects {data_dir}/UnKE/final_data_v3.json.
    Each sample: question, para_question, answer, sub_question, sub_answer.
    """

    def __init__(self, data_dir: str, model_name: str, size: Optional[int] = None):
        path = Path(data_dir) / "UnKE" / "final_data_v3.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        family = model_family(model_name)
        for i in raw:
            i["question"] = wrap_question(i["question"], family)
            i["para_question"] = wrap_question(i["para_question"], family)
            i["answer"] = i["answer"] + end_of_turn(family)
            i["sub_question"] = [wrap_question(q, family) for q in i["sub_question"]]

        self._data = raw[:size] if size is not None else raw

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
