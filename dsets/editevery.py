import json
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from .chat import end_of_turn, model_family, wrap_question


class EditEveryDataset(Dataset):
    """
    EditEverything (https://github.com/jianghoucheng/AnyEdit). Expects
    {data_dir}/EditEverything/editevery.json. Each sample: question, answer, category.
    """

    def __init__(self, data_dir: str, model_name: str, size: Optional[int] = None):
        with open(Path(data_dir) / "EditEverything" / "editevery.json", "r", encoding="utf-8") as f:
            raw = json.load(f)

        family = model_family(model_name)
        for i in raw:
            i["question"] = wrap_question(i["question"], family)
            i["answer"] = i["answer"] + end_of_turn(family)

        self._data = raw[:size] if size is not None else raw

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
