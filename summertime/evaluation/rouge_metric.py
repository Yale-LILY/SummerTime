from summ_eval.rouge_metric import RougeMetric
from summertime.evaluation.summeval_metric import SummEvalMetric
from typing import List, Dict


class Rouge(SummEvalMetric):
    metric_name = "rouge"
    range = (0, 1)
    higher_is_better = True
    requires_heavy_compute = False

    def __init__(self):
        se_metric = RougeMetric()
        super(Rouge, self).__init__(se_metric)

    def evaluate(
        self,
        inputs: List[str],
        targets: List[str],
        keys: List[str] = ["rouge_1_f_score", "rouge_2_f_score", "rouge_l_f_score"],
    ) -> Dict[str, float]:
        score_dict = self.se_metric.evaluate_batch(inputs, targets)
        return {key: score_dict["rouge"][key] for key in keys}
