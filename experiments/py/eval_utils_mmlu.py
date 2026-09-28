"""
Zero-shot MMLU (57 subjects, test split) by greedy generation and parsing of the option letter.
Accuracy is computed over all questions; an unparseable answer counts as wrong.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

from .eval_utils_glue import classification_metrics, format_with_chat_template, generate_greedy

MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts",
    "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions",
)


def build_prompt(question: str, choices: List[str]) -> str:
    options = "".join(f"({letter}) {choice}\n" for letter, choice in zip("ABCD", choices))
    return f"Question: {question}\n{options}Output exactly one letter: A, B, C, or D.\nAnswer:"


def parse_prediction(continuation: str) -> int:
    """Earliest ``(X)`` or standalone letter X in A-D after the last ``Answer:``, or -1."""
    tail = re.split(r"(?i)answer:", continuation)[-1].upper()
    best_pos, best_i = None, -1
    for i, letter in enumerate("ABCD"):
        for pat in (rf"\({letter}\)", rf"\b{letter}\b"):
            m = re.search(pat, tail)
            if m and (best_pos is None or m.start() < best_pos):
                best_pos, best_i = m.start(), i
    return best_i


def load_mmlu_subject(subject: str, mmlu_dir: Optional[str]):
    """Test split of one subject from ``{mmlu_dir}/{subject}/test-00000-of-00001.parquet`` or ``cais/mmlu``."""
    if mmlu_dir is not None:
        p = Path(mmlu_dir) / subject / "test-00000-of-00001.parquet"
        if p.is_file():
            return pq.read_table(p).to_pylist()
    return load_dataset("cais/mmlu", subject, split="test")


def evaluate_mmlu(
    model,
    tokenizer,
    subject_data: Dict,
    device,
    max_new_tokens: int = 5,
    max_samples: Optional[int] = None,
) -> Dict:
    y_true, y_pred, per_subject = [], [], {}
    for subject, ds in tqdm(subject_data.items(), desc="MMLU", leave=False):
        n = len(ds) if max_samples is None else min(len(ds), max_samples)
        yt, yp = [], []
        for i in range(n):
            row = ds[i]
            prompt = format_with_chat_template(tokenizer, build_prompt(row["question"], list(row["choices"])))
            text = generate_greedy(model, tokenizer, prompt, device, max_new_tokens, continuation_only=True)
            yt.append(int(row["answer"]))
            yp.append(parse_prediction(text))
        per_subject[subject] = classification_metrics(yt, yp, num_classes=4)
        y_true.extend(yt)
        y_pred.extend(yp)
    out = classification_metrics(y_true, y_pred, num_classes=4)
    out["per_subject"] = per_subject
    return out
