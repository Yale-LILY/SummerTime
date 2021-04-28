class Metric():
    def __init__(self, metric_name):
        self.score_dict = {}
        self.metric_name =  metric_name

    def evaluate(self, model, data):
        pass

    def get_dict(self, keys):
        return {key: self.score_dict[key]
            for key in keys}
