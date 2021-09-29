import unittest

from model.base_model import SummModel
from model import SUPPORTED_SUMM_MODELS

from pipeline import assemble_model_pipeline, assemble_model_pipeline_2

from evaluation.base_metric import SummMetric
from evaluation import SUPPORTED_EVALUATION_METRICS

from dataset.st_dataset import SummInstance, SummDataset
from dataset import SUPPORTED_SUMM_DATASETS
from dataset.dataset_loaders import ScisummnetDataset, ArxivDataset

from helpers import print_with_color, retrieve_random_test_instances

import random
import time
from typing import List, Union, Tuple
import sys
import re


class IntegrationTests(unittest.TestCase):
    def get_prediction(
        self, model: SummModel, dataset: SummDataset, test_instances: List[SummInstance]
    ) -> Tuple[Union[List[str], List[List[str]]], Union[List[str], List[List[str]]]]:
        """
        Get summary prediction given model and dataset instances.

        :param SummModel `model`: Model for summarization task.
        :param SummDataset `dataset`: Dataset for summarization task.
        :param List[SummInstance] `test_instances`: Instances from `dataset` to summarize.
        :returns Tuple containing summary list of summary predictions and targets corresponding to each instance in `test_instances`.
        """

        src = (
            [ins.source[0] for ins in test_instances]
            if isinstance(dataset, ScisummnetDataset)
            else [ins.source for ins in test_instances]
        )
        tgt = [ins.summary for ins in test_instances]
        query = (
            [ins.query for ins in test_instances] if dataset.is_query_based else None
        )
        prediction = model.summarize(src, query)
        return prediction, tgt

    def get_eval_dict(self, metric: SummMetric, prediction: List[str], tgt: List[str]):
        """
        Run evaluation metric on summary prediction.

        :param SummMetric `metric`: Evaluation metric.
        :param List[str] `prediction`: Summary prediction instances.
        :param List[str] `tgt`: Target prediction instances from dataset.
        """
        score_dict = metric.evaluate(prediction, tgt)
        return score_dict

    def test_all(self):
        """
        Runs integration test on all compatible dataset + model + evaluation metric pipelines supported by SummerTime.
        """

        print_with_color("\nInitializing all evaluation metrics...", "35")
        evaluation_metrics = []
        for eval_cls in SUPPORTED_EVALUATION_METRICS:
            # # TODO: Temporarily skipping Rouge/RougeWE metrics to avoid local bug.
            # if eval_cls in [Rouge, RougeWe]:
            #     continue
            print(eval_cls)
            evaluation_metrics.append(eval_cls())

        print_with_color("\n\nBeginning integration tests...", "35")
        for dataset_cls in SUPPORTED_SUMM_DATASETS:
            # TODO: Temporarily skipping Arxiv (size/time)
            if dataset_cls in [ArxivDataset]:
                continue
            dataset = dataset_cls()
            if dataset.train_set is not None:
                dataset_instances = list(dataset.train_set)
                print(
                    f"\n{dataset.dataset_name} has a training set of {len(dataset_instances)} examples"
                )
                print_with_color(
                    f"Initializing all matching model pipelines for {dataset.dataset_name} dataset...",
                    "35",
                )
                # matching_model_instances = assemble_model_pipeline(dataset_cls, list(filter(lambda m: m != PegasusModel, SUPPORTED_SUMM_MODELS)))
                matching_model_instances = assemble_model_pipeline_2(
                    dataset_cls, SUPPORTED_SUMM_MODELS
                )
                print(matching_model_instances)
                for model, model_name in matching_model_instances:
                    test_instances = retrieve_random_test_instances(
                        dataset_instances=dataset_instances, num_instances=1
                    )
                    print_with_color(
                        f"{'#' * 20} Testing: {dataset.dataset_name} dataset, {model_name} model {'#' * 20}",
                        "35",
                    )
                    prediction, tgt = self.get_prediction(
                        model, dataset, test_instances
                    )
                    print(f"Prediction: {prediction}\nTarget: {tgt}\n")
                    for metric in evaluation_metrics:
                        print_with_color(f"{metric.metric_name} metric", "35")
                        score_dict = self.get_eval_dict(metric, prediction, tgt)
                        print(score_dict)

                    print_with_color(
                        f"{'#' * 20} Test for {dataset.dataset_name} dataset, {model_name} model COMPLETE {'#' * 20}\n\n",
                        "32",
                    )


if __name__ == "__main__":
    if len(sys.argv) > 2 or (
        len(sys.argv) == 2 and not re.match("^\\d+$", sys.argv[1])
    ):
        print("Usage: python tests/integration_test.py [seed]", file=sys.stderr)
        sys.exit(1)

    seed = int(time.time()) if len(sys.argv) == 1 else int(sys.argv.pop())
    random.seed(seed)
    print_with_color(f"(to reproduce) random seeded with {seed}\n", "32")
    unittest.main()
