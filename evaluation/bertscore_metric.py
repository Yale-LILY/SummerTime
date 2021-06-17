from summ_eval.bert_score_metric import BertScoreMetric
from SummerTime.evaluation.summeval_metric import SummEvalMetric
from typing import List, Dict

class BertScore(SummEvalMetric):
    metric_name = 'bert score'
    range = (0, 1)
    higher_is_better = True
    requires_heavy_compute = True 

    def __init__(self):
        se_metric = BertScoreMetric()
        super(BertScore, self).__init__(se_metric)

    def evaluate(self,
                 inputs: List[str],
                 targets: List[str],
                 keys: List[str] = ['bert_score_f1']) -> Dict[str, float]:
        #TODO zhangir: update when datasets api is merged
        return super(BertScore, self).evaluate(inputs, targets, keys)
