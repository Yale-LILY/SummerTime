from summ_eval.bert_score_metric import BertScoreMetric
from .Metric import Metric

class bertscore(Metric):
    def __init__(self):
        super().__init__('bert_score')
        self.se_bert_score = BertScoreMetric()

    def evaluate(self, model, data):
        predictions = model.summarize(data['article'])
        self.score_dict = self.se_bert_score.evaluate_batch(
            predictions, data['highlights'])
