from summ_eval.bleu_metric import BleuMetric
from SummerTime.evaluation.summeval_metric import SummEvalMetric
from typing import List, Dict

class Bleu(SummEvalMetric):
    metric_name = 'bleu'
    range = (0, 10)
    higher_is_better = True
    low_resource = True

    def __init__(self):
        se_metric = BleuMetric()
        super(Bleu, self).__init__(se_metric)

    def evaluate(self,
                 inputs: List[str],
                 targets: List[str],
                 keys: List[str] = ['bleu']) -> Dict[str, float]:
        # TODO zhangir: potentially update when dataset api is merged.
        return super(Bleu, self).evaluate(inputs, targets, keys)
