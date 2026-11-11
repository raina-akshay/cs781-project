"""The published document-mean metrics, with pinned evaluation-only weights."""
import importlib.util

import numpy as np
from bert_score import BERTScorer
from rouge_score import rouge_scorer, tokenizers

from .common import PHASE, verify_metric_assets

OFFICIAL = PHASE / "evaluation/evaluate.py"


class SummaryScorer:
    def __init__(self, snapshot, device):
        verify_metric_assets(snapshot)
        spec = importlib.util.spec_from_file_location("_cs781_official_phase6", OFFICIAL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.lcs = module.lcs
        self.rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        self.tokenizer = tokenizers.DefaultTokenizer(use_stemmer=True)
        self.bert = BERTScorer(
            model_type=str(snapshot),
            num_layers=17,
            batch_size=8,
            device=str(device),
            idf=False,
            rescale_with_baseline=False,
            use_fast_tokenizer=False,
        )
        self.bert._model.requires_grad_(False)

    def score(self, predictions, references):
        if not predictions or len(predictions) != len(references):
            raise ValueError("Metric population differs")
        preds = [p.strip() for p in predictions]
        refs = [r.strip() for r in references]
        if any(not x for x in preds + refs):
            raise ValueError("Empty prediction/reference cannot be silently scored")
        _, _, f = self.bert.score(preds, refs, batch_size=8, verbose=False)
        b = f.detach().float().cpu().numpy().astype(np.float64)
        result = []
        for i, (p, r) in enumerate(zip(preds, refs)):
            pt = self.tokenizer.tokenize(p)
            rt = self.tokenizer.tokenize(r)
            if not pt and not rt:
                rl = 1.0
            elif not pt or not rt:
                rl = 0.0
            else:
                rl = 2.0 * self.lcs(pt, rt) / (len(pt) + len(rt))
            result.append(
                {
                    "bertscore_f1": float(b[i]),
                    "rouge1_f1": float(self.rouge.score(r, p)["rouge1"].fmeasure),
                    "rougel_f1": float(rl),
                }
            )
        if not all(np.isfinite(list(x.values())).all() for x in result):
            raise FloatingPointError("Nonfinite metric")
        return result


def mean_metrics(rows):
    return {
        "documents": len(rows),
        **{
            key: float(np.mean([r[key] for r in rows], dtype=np.float64))
            for key in ("bertscore_f1", "rouge1_f1", "rougel_f1")
        },
    }
