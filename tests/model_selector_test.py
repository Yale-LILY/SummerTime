import unittest
from evaluation import SUPPORTED_EVALUATION_METRICS
from evaluation.model_selector import ModelSelector
from model.base_model import SummModel
from dataset.st_dataset import SummInstance


class ToyModel(SummModel):
    def __init__(self, num: int):
        super().__init__(ToyModel, self)
        self.model_name = str(num)

    def summarize(self, corpus):
        return (
            [
                """
        Glowing letters that had been hanging above
        the Yankee stadium from 1976 to 2008 were placed for auction at
        Sotheby’s on Wednesday, but were not sold, The current owner
        of the sign is Reggie Jackson, a Yankee hall-of-famer."""
            ]
            * len(corpus)
        )


class TestModelSelector(unittest.TestCase):
    def model_selector_evaluate(self):
        print(f"{'#'*10} model_selector_evaluate STARTS {'#'*10}")

        model_1 = ToyModel(1)
        model_2 = ToyModel(2)
        models = [model_1, model_2]
        generator1 = iter([SummInstance("A context.", "A summary.")] * 10)
        generator2 = iter([SummInstance("A context.", "A summary.")] * 10)
        metrics = [metric() for metric in SUPPORTED_EVALUATION_METRICS]

        selector = ModelSelector(models, generator1, metrics)
        table = selector.run()
        print(table)

        new_selector = ModelSelector(models, generator2, metrics)
        smart_table = new_selector.run_halving(min_instances=2, factor=2)
        print(smart_table)

        print(f"{'#'*10} model_selector_evaluate ENDS {'#'*10}")