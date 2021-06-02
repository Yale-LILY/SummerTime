from summ_eval.rouge_metric import RougeMetric
from .base_metric import SummMetric

class Rouge(SummMetric):
    def __init__(self):
        super(Rouge, self).__init__()
        self.se_rouge = RougeMetric()

    def evaluate(self, model, data):
        # TODO zhangir: fix when dataset api is  merged.
        predictions = model.summarize(data['article'])
        self.score_dict = self.se_rouge.evaluate_batch(predictions,
            data['highlights'])

    def get_dict(self,
                 keys = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score']
                 ):
        return {key: self.score_dict['rouge'][key]
            for key in keys}
