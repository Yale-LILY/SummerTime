from summ_eval.rouge_metric import RougeMetric
from .summeval_metric import SummEvalMetric

class Rouge(SummEvalMetric):
    metric_name = 'rouge'
    range = (0, 1)
    higher_is_better = True
    low_resource = True

    def __init__(self):
        se_metric = RougeMetric()
        super(Rouge, self).__init__(se_metric)

    def evaluate(self,
                 inputs,
                 targets,
                 keys = ['rouge_3_f_score']):
        return super(Rouge, self).evaluate(inputs, targets, keys)
