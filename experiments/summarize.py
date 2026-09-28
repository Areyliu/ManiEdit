"""
Metrics for unstructured editing results: BLEU, ROUGE-1/2/L (recall) and BERT Score
(cosine similarity of Sentence-BERT embeddings) between predictions and edit targets.

    python -m experiments.summarize --dir_name ManiEdit --runs run_000,run_001
"""
import argparse
import json
from collections import defaultdict
from typing import Dict, List

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

from util.globals import *

DEFAULT_BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _nonempty(text: str) -> str:
    # ROUGE fails on predictions without any whitespace
    return text if " " in text else text + " "


def _bleu_rouge(preds: List[str], refs: List[str], rouge: Rouge) -> Dict[str, float]:
    bleu, r1, r2, rl = [], [], [], []
    for pred, ref in zip(preds, refs):
        bleu.append(sentence_bleu([ref], pred))
        scores = rouge.get_scores(pred, ref)[0]
        r1.append(scores["rouge-1"]["r"])
        r2.append(scores["rouge-2"]["r"])
        rl.append(scores["rouge-l"]["r"])
    mean = lambda x: sum(x) / len(x) if x else 0
    return {"BLEU SCORE": mean(bleu), "ROUGE-1": mean(r1), "ROUGE-2": mean(r2), "ROUGE-L": mean(rl)}


def _sub_rouge(data: List[Dict], rouge: Rouge) -> Dict[str, float]:
    r1, r2, rl = [], [], []
    for d in data:
        n = len(d["sub_pred"])
        s1 = s2 = sl = 0
        for pred, ref in zip(d["sub_pred"], d["sub_answer"]):
            scores = rouge.get_scores(_nonempty(pred), ref)[0]
            s1 += scores["rouge-1"]["r"]
            s2 += scores["rouge-2"]["r"]
            sl += scores["rouge-l"]["r"]
        r1.append(s1 / n if n else 0)
        r2.append(s2 / n if n else 0)
        rl.append(sl / n if n else 0)
    mean = lambda x: sum(x) / len(x) if x else 0
    return {"ROUGE-1": mean(r1), "ROUGE-2": mean(r2), "ROUGE-L": mean(rl)}


def _bert_score(st_model, preds: List[str], refs: List[str]) -> float:
    emb_ref = st_model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
    emb_pred = st_model.encode(preds, convert_to_tensor=True, show_progress_bar=False)
    return util.cos_sim(emb_ref, emb_pred).diagonal().mean().item()


def compute_metrics(data: List[Dict], ds_name: str, bert_model: str = DEFAULT_BERT_MODEL, device: int = 0) -> Dict:
    """
    UnKE / AKEW: Original, Para and Sub (sub-questions) scores.
    EditEverything: Original scores per category.
    """
    rouge = Rouge()
    st_model = SentenceTransformer(bert_model, device=f"cuda:{device}")

    if ds_name == "editevery":
        by_cat = defaultdict(list)
        for d in data:
            by_cat[d.get("category", "default")].append(d)
        metrics = {}
        for cat, items in by_cat.items():
            preds = [_nonempty(d["original_prediction"]) for d in items]
            refs = [d["answer"] for d in items]
            metrics[cat] = _bleu_rouge(preds, refs, rouge)
            metrics[cat]["Bert Score"] = _bert_score(st_model, preds, refs)
        return metrics

    refs = [d["answer"] for d in data]
    ori = [_nonempty(d["original_prediction"]) for d in data]
    para = [_nonempty(d["para_prediction"]) for d in data]
    metrics = {
        "Original": _bleu_rouge(ori, refs, rouge),
        "Para": _bleu_rouge(para, refs, rouge),
        "Sub": _sub_rouge(data, rouge),
    }
    metrics["Original"]["Bert Score"] = _bert_score(st_model, ori, refs)
    metrics["Para"]["Bert Score"] = _bert_score(st_model, para, refs)
    return metrics


def main(dir_name: str, runs: List[str], bert_model: str, device: int):
    for run in runs:
        run_dir = RESULTS_DIR / dir_name / run
        with open(run_dir / "params.json", "r") as f:
            params = json.load(f)
        with open(run_dir / "results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = compute_metrics(data, params["ds_name"], bert_model, device)
        with open(run_dir / "summary.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"{run_dir}:\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", default="ManiEdit")
    parser.add_argument("--runs", required=True, help="Comma-separated run directories, e.g. run_000,run_001")
    parser.add_argument("--bert_model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    main(args.dir_name, args.runs.split(","), args.bert_model, args.device)
