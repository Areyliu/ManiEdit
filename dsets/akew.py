import json
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from .chat import end_of_turn, model_family, wrap_question

AKEW_FILES = {
    "counterfact": Path("AKEW") / "CounterFact.json",
    "mquake": Path("AKEW") / "MQuAKE-CF.json",
}


class AKEWDataset(Dataset):
    """
    Unstructured AKEW (https://github.com/bobxwu/AKEW) CounterFact / MQuAKE-CF.
    Expects {data_dir}/AKEW/CounterFact.json and {data_dir}/AKEW/MQuAKE-CF.json.
    Each sample: question, para_question, answer (fact_new_uns), sub_question, sub_answer
    (the first five unsfact_triplets_GPT).
    """

    def __init__(self, data_dir: str, dataset_type: str, model_name: str, size: Optional[int] = None):
        with open(Path(data_dir) / AKEW_FILES[dataset_type], "r", encoding="utf-8") as f:
            raw = json.load(f)

        family = model_family(model_name)
        data = []
        for case_id, record in enumerate(raw):
            if dataset_type == "counterfact":
                rewrite = record["requested_rewrite"]
                question = rewrite["prompt_full"]
                para_question = record["paraphrase_prompts"][0]
            else:
                rewrite = record["requested_rewrite"][0]
                question = rewrite["prompt"].format(rewrite["subject"]) + "?"
                para_question = rewrite["question"]
            triplets = rewrite["unsfact_triplets_GPT"][:5]
            data.append(
                {
                    "id": case_id,
                    "question": wrap_question(question, family),
                    "para_question": wrap_question(para_question, family),
                    "answer": rewrite["fact_new_uns"] + end_of_turn(family),
                    "sub_question": [wrap_question(q["prompt"].format(q["subject"]), family) for q in triplets],
                    "sub_answer": [q["target"] for q in triplets],
                }
            )
        self._data = data[:size] if size is not None else data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
