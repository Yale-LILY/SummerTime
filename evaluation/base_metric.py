from typing import List

class SummMetric():
    def __init__(self):
        self.score_dict = {}

    def evaluate(self,
                ## TODO zhangir: figure out how to import SummModel 
                 model,
                 ## TODO zhangir: integrate with dataset api
                 data):
        """
        All metrics should have this function
        """
        raise NotImplementedError("the base class for metrics shouldn't be instantiated!")

    def get_dict(self, keys: List[str]):
        return {key: self.score_dict[key]
            for key in keys}
