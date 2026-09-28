"""
Zero-shot GLUE evaluation (SST-2, CoLA, MRPC, RTE, MNLI) by greedy generation and parsing,
following the AlphaEdit ``glue_eval`` prompts with an explicit output-format instruction.
A prediction that cannot be parsed counts as wrong.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import tqdm

GLUE_TASKS = ("sst2", "cola", "mrpc", "rte", "mnli_matched", "mnli_mismatched")


def build_prompt(task: str, row: Dict) -> str:
    if task == "sst2":
        return (
            f"Review : {row['sentence']}\n"
            "Classify the sentiment of the review. Output exactly one word: positive or negative.\n"
            "Sentiment :"
        )
    if task == "cola":
        return (
            "Is this sentence linguistically acceptable?\n"
            f"Sentence: {row['sentence']}\n"
            "Output exactly one word: yes or no.\n"
            "Answer: "
        )
    if task == "mrpc":
        return (
            "Are the sentences paraphrases of each other.\n"
            f"Sentence 1: {row['sentence1']}\n"
            f"Sentence 2: {row['sentence2']}\n"
            "Output exactly one word: yes or no.\n"
            "Answer:"
        )
    if task == "rte":
        return (
            f"{row['sentence1']}\n"
            f"question: {row['sentence2']} True or False?\n"
            "Answer using exactly one word: True or False.\n"
            "answer:"
        )
    return (
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\n"
        "Classify the relationship as one of: entailment, neutral, contradiction.\n"
        "Output exactly one word: entailment or neutral or contradiction.\n"
        "Answer:"
    )


def parse_prediction(task: str, generated_text: str) -> int:
    """GLUE label parsed from the decoded prompt + continuation, or -1 if unparseable."""
    if task == "sst2":
        low = generated_text.split("Sentiment :")[-1].strip().lower()
        if "positive" in low:
            return 1
        if "negative" in low:
            return 0
        return -1
    low = re.split(r"(?i)answer:", generated_text)[-1].strip().lower()
    if task in ("cola", "mrpc"):
        if "yes" in low:
            return 1
        if "no" in low:
            return 0
        return -1
    if task == "rte":
        # GLUE RTE labels: 0 = entailment ("True"), 1 = not_entailment ("False")
        if "false" in low:
            return 1
        if "true" in low:
            return 0
        return -1
    best_pos, best_lab = None, -1
    for word, lab in (("entailment", 0), ("neutral", 1), ("contradiction", 2)):
        pos = low.find(word)
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best_pos, best_lab = pos, lab
    return best_lab


def load_glue_split(task: str, glue_dir: Optional[str]):
    """
    Validation split of a GLUE task. Uses local parquet
    ``{glue_dir}/{task}/validation-00000-of-00001.parquet`` when present
    (official Hub folder names: ``sst2``, ``cola``, ``mrpc``, ``rte``,
    ``mnli_matched``, …), and otherwise downloads ``nyu-mll/glue``.
    MNLI on the Hub uses the ``validation_matched`` split.
    """
    config, hub_split = task, "validation"
    if task.startswith("mnli"):
        config, hub_split = "mnli", "validation_" + task.split("_")[1]
    if glue_dir is not None:
        p = Path(glue_dir) / task / "validation-00000-of-00001.parquet"
        if p.is_file():
            return pq.read_table(p).to_pylist()
    return load_dataset("nyu-mll/glue", config, split=hub_split)


def format_with_chat_template(tokenizer, user_content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt: str, device, max_new_tokens: int, continuation_only: bool) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kw = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": pad_id}
    if tokenizer.eos_token_id is not None:
        gen_kw["eos_token_id"] = tokenizer.eos_token_id
    out = model.generate(input_ids, **gen_kw)[0]
    if continuation_only:
        out = out[input_ids.shape[1]:]
    return tokenizer.decode(out, skip_special_tokens=True)


def classification_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    """Accuracy over all samples (unparseable = wrong); F1 / MCC over the parseable ones."""
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if p >= 0 and p == t)
    ok = [i for i in range(n) if y_pred[i] >= 0]
    out = {
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "total": n,
        "invalid": n - len(ok),
        "invalid_rate": (n - len(ok)) / max(n, 1),
    }
    if not ok:
        out.update({"f1_macro": 0.0, "mcc": 0.0})
        return out
    yt = np.asarray([y_true[i] for i in ok])
    yp = np.asarray([y_pred[i] for i in ok])
    out["f1_macro"] = float(f1_score(yt, yp, average="macro", zero_division=0))
    if num_classes == 2:
        out["f1_binary"] = float(f1_score(yt, yp, average="binary", zero_division=0))
    out["mcc"] = float(matthews_corrcoef(yt, yp))
    return out


def evaluate_glue_task(
    model, tokenizer, task: str, ds, device, max_new_tokens: int = 5, max_samples: Optional[int] = None
) -> Dict[str, float]:
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    y_true, y_pred = [], []
    for i in tqdm(range(n), desc=task, leave=False):
        row = ds[i]
        label = int(row["label"])
        if label < 0:
            continue
        prompt = format_with_chat_template(tokenizer, build_prompt(task, row))
        text = generate_greedy(model, tokenizer, prompt, device, max_new_tokens, continuation_only=False)
        y_true.append(label)
        y_pred.append(parse_prediction(task, text))
    return classification_metrics(y_true, y_pred, num_classes=3 if task.startswith("mnli") else 2)
