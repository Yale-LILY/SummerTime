import unittest

from model import SUPPORTED_SUMM_MODELS
from model.base_model import SummModel
from evaluation import SUPPORTED_EVALUATION_METRICS
from evaluation.base_metric import SummMetric
from dataset import SUPPORTED_SUMM_DATASETS
from dataset.st_dataset import SummInstance

import random
import time
from typing import Dict, List, Union

from evaluation.rouge_metric import Rouge
from evaluation.rougewe_metric import RougeWe
from model.single_doc import LexRankModel
from model.multi_doc import MultiDocJointModel
from model.multi_doc import MultiDocSeparateModel
from dataset.non_huggingface_datasets import ScisummnetDataset

class IntegrationTests(unittest.TestCase):
    @staticmethod
    def flatten_list_to_str(doc: Union[str, List[str]]) -> str:
        if isinstance(doc, list):
            return " ".join(doc)
        return doc

    def retrieve_test_instances(self, dataset_instances: List[SummInstance], num_instances = 5) -> List[SummInstance]:
        test_instances = []
        for i in range(num_instances):
            test_instances.append(dataset_instances[random.randint(0, len(dataset_instances) - 1)])
        return test_instances

    def test_single_integration(self, dataset_class, dataset_instances: List[SummInstance], model_class, model_instance: SummModel, evaluation_metric: SummMetric):
        print(f"\n{'#' * 20} TESTING: dataset {dataset_class}, model {model_instance.model_name}, evaluation metric {evaluation_metric.metric_name} {'#' * 20}")
        src = [ins.source[0] for ins in dataset_instances] if dataset_class == ScisummnetDataset else [IntegrationTests.flatten_list_to_str(ins.source) for ins in dataset_instances]
        tgt = [ins.summary for ins in dataset_instances]
        prediction = model_instance.summarize([src] if model_class == MultiDocJointModel or model_class == MultiDocSeparateModel else src)
        print(prediction)
        print(tgt)
        score_dict = evaluation_metric.evaluate(prediction, [" ".join(tgt)] if model_class == MultiDocJointModel or model_class == MultiDocSeparateModel else tgt)
        print(score_dict)
        return

    def test_all(self):
        for dataset_class in SUPPORTED_SUMM_DATASETS:
            dataset = dataset_class()
            if dataset.train_set is not None:
                train_set = list(dataset.train_set)
                print(f"{dataset_class} has a training set of {len(train_set)} examples")
                dataset_instances = self.retrieve_test_instances(train_set)
                for model_class in SUPPORTED_SUMM_MODELS:
                    model = model_class([IntegrationTests.flatten_list_to_str(x.source[0] if dataset_class == ScisummnetDataset else x.source) for x in train_set[0:100]]) if model_class == LexRankModel else model_class()
                    for eval_class in SUPPORTED_EVALUATION_METRICS:
                        if eval_class != Rouge and eval_class != RougeWe:
                            eval_metric = eval_class()
                            self.test_single_integration(dataset_class = dataset_class, dataset_instances = dataset_instances, model_class = model_class, model_instance = model, evaluation_metric = eval_metric)

if __name__ == '__main__':
    random.seed(time.time())
    unittest.main()
