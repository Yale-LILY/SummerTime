from evaluation.summeval_metric import SummEvalMetric
from typing import List, Dict

import nltk


class RougeWe(SummEvalMetric):
    metric_name = "rougeWE"
    range = (0, 1)
    higher_is_better = True
    requires_heavy_compute = True

    def __init__(self):
        from summ_eval.rouge_we_metric import RougeWeMetric

        nltk.download("stopwords")
        se_metric = RougeWeMetric()
        super(RougeWe, self).__init__(se_metric)

    def evaluate(
        self, inputs: List[str], targets: List[str], keys: List[str] = ["rouge_we_3_f"]
    ) -> Dict[str, float]:
        # TODO zhangir: update when dataset api is merged.
        return super(RougeWe, self).evaluate(inputs, targets, keys)
