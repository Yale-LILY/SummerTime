from typing import List, Tuple

class SummMetric():
    metric_name: str = None
    range: Tuple[float, float] = None
    higher_is_better: bool = None
    low_resource: bool = None

    def evaluate(self,
                 ## TODO zhangir: integrate with dataset api
                 inputs: List[str],
                 targets: List[str],
                 keys: List[str]):
        """
        All metrics should have this function.
        :input: A list of summaries.
        :target: A list of target summaries corresponding to each entry of input.
        :keys: Which metrics to return,
        e.g, ['rouge_1_f_score', 'rouge_2_f_score']
        :return: A dictionary with keys metrics and values scores.
        """
        raise NotImplementedError("the base class for metrics shouldn't be instantiated!")
