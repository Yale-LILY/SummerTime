from .base_metric import SummMetric
from typing import List, Dict
from nltk.translate import meteor_score as nltk_meteor
import nltk
import statistics


class Meteor(SummMetric):
    metric_name = "meteor"
    range = (0, 1)
    higher_is_better = True
    requires_heavy_compute = False

    def __init__(self):
        nltk.download("wordnet")

    def evaluate(
        self, inputs: List[str], targets: List[str], keys=["meteor"]
    ) -> Dict[str, float]:

        for key in keys:
            if key != "meteor":
                raise KeyError(key, "is not a valid key")

        meteor_scores = [
            nltk_meteor.meteor_score([input], target)
            for input, target in zip(inputs, targets)
        ]
        meteor_score = statistics.mean(meteor_scores)

        return {key: meteor_score for key in keys}
