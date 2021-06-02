from summ_eval.rouge_we_metric import RougeWeMetric
from .base_metric import SummMetric
import nltk

class RougeWE(SummMetric):
    def __init__(self):
        nltk.download('stopwords')
        super(RougeWE, self).__init__()
        self.se_rouge_we = RougeWeMetric()


    def evaluate(self, model, data):
        #TODO zhangir: update when dataset api is merged.
        predictions = model.summarize(data['article'])
        self.score_dict = self.se_rouge_we.evaluate_batch(
            predictions, data['highlights'])

    def get_dict(self, keys=['rouge_we_3_f']):
        return super(RougeWE, self).get_dict(keys)
