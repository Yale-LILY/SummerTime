from summ_eval.bert_score_metric import BertScoreMetric
from .base_metric import SummMetric

class BertScore(SummMetric):
    def __init__(self):
        super(BertScore, self).__init__()
        self.se_bert_score = BertScoreMetric()

    def evaluate(self, model, data):
        #TODO zhangir: update when datasets api is merged
        predictions = model.summarize(data['article'])
        self.score_dict = self.se_bert_score.evaluate_batch(
            predictions, data['highlights'])

    def get_dict(self, keys=['bert_score_f1']):
        return super(BertScore, self).get_dict(keys)
