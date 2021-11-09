from summ_eval.bleu_metric import BleuMetric
from summertime.evaluation.summeval_metric import SummEvalMetric
from typing import List, Dict


class Bleu(SummEvalMetric):
    metric_name = "bleu"
    range = (0, 1)
    higher_is_better = True
    requires_heavy_compute = False

    def __init__(self):
        se_metric = BleuMetric()
        super(Bleu, self).__init__(se_metric)

    def evaluate(
        self, inputs: List[str], targets: List[str], keys: List[str] = ["bleu"]
    ) -> Dict[str, float]:
        # TODO zhangir: potentially update when dataset api is merged.
        result_dict = super(Bleu, self).evaluate(inputs, targets, keys)
        # making this normalization to make sure everything metric is 0-1
        result_dict["bleu"] = result_dict["bleu"] / 100

        return result_dict
