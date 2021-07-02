import unittest

from model import SUPPORTED_SUMM_MODELS
from model.base_model import SummModel
from evaluation import SUPPORTED_EVALUATION_METRICS
from evaluation.base_metric import SummMetric
from dataset import SUPPORTED_SUMM_DATASETS
from dataset.st_dataset import SummInstance, SummDataset

import random
import time
from typing import Dict, List, Union

from evaluation.rouge_metric import Rouge
from evaluation.rougewe_metric import RougeWe
from model.single_doc import LexRankModel
from dataset.non_huggingface_datasets import ScisummnetDataset
from dataset.huggingface_datasets import CnndmDataset

class IntegrationTests(unittest.TestCase):
    @staticmethod
    def print_with_color(s: str, color: str):
        print(f"\033[{color}m{s}\033[0m")

    @staticmethod
    def flatten_list_to_str(doc: Union[str, List[str]]) -> str:
        return " ".join(doc) if isinstance(doc, list) else doc

    def retrieve_test_instances(self, dataset_instances: List[SummInstance], num_instances = 5) -> List[SummInstance]:
        test_instances = []
        for i in range(num_instances):
            test_instances.append(dataset_instances[random.randint(0, len(dataset_instances) - 1)])
        return test_instances

    def _test_single_integration(self, dataset: SummDataset, test_instances: List[SummInstance], model: SummModel, metric: SummMetric):
        IntegrationTests.print_with_color(f"{'#' * 20} Testing: {dataset.dataset_name} dataset, {model.model_name} model, {metric.metric_name} evaluation metric {'#' * 20}", "35")
        # if dataset.is_multi_document != model.is_multi_document or dataset.is_dialogue_based != model.is_dialogue_based or dataset.is_query_based != dataset.is_query_based:
        #     print("Skipping because summarization tasks of dataset and model don't match\n")
        #     return
        src = [ins.source[0] for ins in test_instances] if isinstance(dataset, ScisummnetDataset) else [IntegrationTests.flatten_list_to_str(ins.source) for ins in test_instances]
        tgt = [ins.summary for ins in test_instances]
        prediction = model.summarize([src] if model.is_multi_document else src)
        print(prediction)
        print(tgt)
        score_dict = metric.evaluate(prediction, [" ".join(tgt)] if model.is_multi_document else tgt)
        print(score_dict)

    def test_all(self):
        IntegrationTests.print_with_color("Initializing all datasets...", "35")
        # datasets = []
        # for dataset_cls in SUPPORTED_SUMM_DATASETS:
        #     print(dataset_cls)
        #     ds = dataset_cls()
        #     datasets.append(ds)
        lxr_dataset = CnndmDataset()
        IntegrationTests.print_with_color("Initializing all models...", "35")
        models = []
        for model_cls in SUPPORTED_SUMM_MODELS:
            print(model_cls)
            models.append(model_cls([x.source for x in list(lxr_dataset.train_set)[0:100]]) if model_cls == LexRankModel else model_cls())
        IntegrationTests.print_with_color("Initializing all evaluation metrics...", "35")
        evaluation_metrics = []
        for eval_cls in SUPPORTED_EVALUATION_METRICS:
            print(eval_cls)
            evaluation_metrics.append(eval_cls())

        print('\n\n')
        IntegrationTests.print_with_color("Beginning integration tests...", "32")
        for dataset_cls in SUPPORTED_SUMM_DATASETS:
            dataset = dataset_cls()
            if dataset.train_set is not None:
                dataset_instances = list(dataset.train_set)
                print(f"{dataset.dataset_name} has a training set of {len(dataset_instances)} examples")
                test_instances = self.retrieve_test_instances(dataset_instances)
                for model in models:
                    for metric in evaluation_metrics:
                        print('\n')
                        self._test_single_integration(dataset = dataset, test_instances = test_instances, model = model, metric = metric)

if __name__ == '__main__':
    random.seed(time.time())
    unittest.main()
