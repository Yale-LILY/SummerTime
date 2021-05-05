from summ_eval.bleu_metric import BleuMetric
from .Metric import Metric

class bleu(Metric):
    def __init__(self):
        super().__init__('bleu')
        self.se_bleu = BleuMetric()

    def evaluate(self, model, data):
        predictions = model.summarize(data['article'])
        self.score_dict = self.se_bleu.evaluate_batch(
            predictions, data['highlights'])
