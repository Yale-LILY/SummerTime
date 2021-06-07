from summ_eval.bleu_metric import BleuMetric
from SummerTime.evaluation.summeval_metric import SummEvalMetric

class Bleu(SummEvalMetric):
    metric_name = 'bleu'
    range = (0, 1)
    higher_is_better = True
    low_resource = True

    def __init__(self):
        se_metric = BleuMetric()
        super(Bleu, self).__init__(se_metric)

    def evaluate(self,
                 inputs,
                 targets,
                 keys = ['bleu']):
        # TODO zhangir: potentially update when dataset api is merged.
        return super(Bleu, self).evaluate(inputs, targets, keys)
