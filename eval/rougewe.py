from summ_eval.rouge_we_metric import RougeWeMetric
from .Metric import Metric
import nltk

class rougewe(Metric):
    def __init__(self):
        nltk.download('stopwords')
        super().__init__('rouge_we')
        self.se_rouge_we = RougeWeMetric()


    def evaluate(self, model, data):
        predictions = model.summarize(data['article'])
        self.score_dict = self.se_rouge_we.evaluate_batch(
            predictions, data['highlights'])
