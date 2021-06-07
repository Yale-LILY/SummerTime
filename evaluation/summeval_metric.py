from .base_metric import SummMetric
from summ_eval.metric import Metric as SEMetric

class SummEvalMetric(SummMetric):
    """
    Generic class for a summarization metric
    whose backend is SummEval.
    """

    def __init__(self,
                 se_metric: SEMetric):
        self.se_metric = se_metric

    def evaluate(self,
                 inputs,
                 targets,
                 keys):
        score_dict = self.se_metric.evaluate_batch(
            inputs, targets)
        return {key: score_dict[key] for key in keys}
