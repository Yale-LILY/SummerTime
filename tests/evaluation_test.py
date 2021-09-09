import unittest
from typing import Tuple, List, Dict

from evaluation import SUPPORTED_EVALUATION_METRICS

from helpers import print_with_color


class TestEvaluationMetrics(unittest.TestCase):
    def get_summary_pairs(self, size: int = 1) -> Tuple[List[str]]:
        test_output = (
            [
                """
        Glowing letters that had been hanging above
        the Yankee stadium from 1976 to 2008 were placed for auction at
        Sotheby’s on Wednesday, but were not sold, The current owner
        of the sign is Reggie Jackson, a Yankee hall-of-famer."""
            ]
            * size
        )
        test_target = (
            [
                """
        An auction for the lights from Yankee Stadium failed to
        produce any bids on Wednesday at Sotheby’s. The lights,
        currently owned by former Yankees player Reggie Jackson,
        lit the stadium from 1976 until 2008."""
            ]
            * size
        )

        return test_output, test_target

    def test_evaluate(self):
        print_with_color(f"{'#'*10} Testing all evaluation metrics... {'#'*10}\n", "35")

        num_eval_metrics = 0

        for metric_class in SUPPORTED_EVALUATION_METRICS:
            # if metric_class in [Rouge, RougeWe]:
            #     # TODO: Temporarily skipping Rouge/RougeWE metrics to avoid local bug.
            #     continue

            print_with_color(f"Testing {metric_class.metric_name}...", "35")

            metric = metric_class()

            test_output, test_target = self.get_summary_pairs()
            score_dict = metric.evaluate(test_output, test_target)
            print(f"{metric_class} output dictionary")
            print(score_dict)
            self.assertTrue(isinstance(score_dict, Dict))
            self.assertNotEqual(score_dict, {})

            for k, v in score_dict.items():
                self.assertTrue(isinstance(k, str) and isinstance(v, float))
                # # TODO: add metric score range assertions
                # self.assertTrue(self.range[0] <= score_dict[k])
                # self.assertTrue(score_dict[k] <= self.range[1])

            print_with_color(f"{metric_class.metric_name} test complete\n", "32")
            num_eval_metrics += 1

        print_with_color(
            f"{'#'*10} Evaluation metrics test complete ({num_eval_metrics} metrics) {'#'*10}",
            "32",
        )


if __name__ == "__main__":
    unittest.main()
