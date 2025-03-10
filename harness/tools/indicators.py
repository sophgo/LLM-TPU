import os
import nltk

from abc import ABC
from typing import (Any, Dict, List, Union, Tuple)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers.data.metrics.squad_metrics import compute_f1

class Indicators(ABC):
    def __init__(self):
        super().__init__()

    # f1
    def calc_f1(self, answer : List[str], ref : Union[str, List[str]]) -> Dict[str, float]:
        f1 = 0
        if isinstance(ref, list):
            ref_num = len(ref)
            for single_ref in ref:
                f1 = max(f1, compute_f1(single_ref, answer))
        else:
            f1 = compute_f1(ref, answer)

        return {"f1" : f1}

    # bleu
    def calc_bleu(self, answer : List[str], refs : List[List[str]]) -> Dict[str, float]:
        smooth_fn = SmoothingFunction().method1
        bleu = sentence_bleu(refs, answer, smoothing_function=smooth_fn)
        return {"bleu" : bleu}

    # rouge
    def calc_rouge(self, answer : str, ref : str) -> Dict[str, float]:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(answer, ref)
        return scores


    