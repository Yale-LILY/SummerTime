
from summ_eval.rouge_metric import RougeMetric
"""rouge = RougeMetric()

summaries = ["This is one summary", "This is another summary"]
references = ["This is one reference", "This is another"]

rouge_dict = rouge.evaluate_batch(summaries, references)
print('hello')
print(rouge_dict)
"""

class rouge():
    def __init__(self):
        self.score_dict = {}
        self.se_rouge = RougeMetric()

    def evaluate(self, model, data):
        predictions = model.summarize(data['article'])
        self.score_dict = self.se_rouge.evaluate_batch(predictions,
            data['highlights'])

    def get(self, keys):
        return {key: self.score_dict['rouge'][key] for key in keys}
