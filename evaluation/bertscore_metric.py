from summ_eval.bert_score_metric import BertScoreMetric
from .summeval_metric import SummEvalMetric

class BertScore(SummEvalMetric):
    metric_name = 'bert score'
    range = (0, 1)
    higher_is_better = True
    low_resource = False

    def __init__(self):
        se_metric = BertScoreMetric()
        super(BertScore, self).__init__(se_metric)

    def evaluate(self,
                 inputs,
                 targets,
                 keys = ['bert_score_f1']):
        #TODO zhangir: update when datasets api is merged
        return super(BertScore, self).evaluate(inputs, targets, keys)
