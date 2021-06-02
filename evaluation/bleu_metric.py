from summ_eval.bleu_metric import BleuMetric
from .base_metric import SummMetric

class Bleu(SummMetric):
    def __init__(self):
        super(Bleu, self).__init__()
        self.se_bleu = BleuMetric()

    def evaluate(self, model, data):
        # TODO zhangir: update when dataset api is merged.
        predictions = model.summarize(data['article'])
        self.score_dict = self.se_bleu.evaluate_batch(
            predictions, data['highlights'])

    def get_dict(self, keys=['bleu']):
        return super(Bleu, self).get_dict(keys)
