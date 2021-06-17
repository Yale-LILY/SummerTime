from summ_eval.rouge_metric import RougeMetric
from evaluation.summeval_metric import SummEvalMetric
from typing import List, Dict

class Rouge(SummEvalMetric):
    metric_name = 'rouge'
    range = (0, 1)
    higher_is_better = True
    requires_heavy_compute = False

    def __init__(self):
        se_metric = RougeMetric()
        super(Rouge, self).__init__(se_metric)

    def evaluate(self,
                 inputs: List[str],
                 targets: List[str],
                 keys: List[str] = ['rouge_3_f_score']) -> Dict[str, float]:
        return super(Rouge, self).evaluate(inputs, targets, keys)
