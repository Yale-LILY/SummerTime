import unittest
from typing import Tuple, List, Dict

from evaluation import SUPPORTED_EVALUATION_METRICS

class TestEvaluationMetrics(unittest.TestCase):
    def get_summary_pair(self, size: int=1) -> Tuple[List[str]]:
        test_output = [ """
        Glowing letters that had been hanging above
        the Yankee stadium from 1976 to 2008 were placed for auction at
        Sotheby’s on Wednesday, but were not sold, The current owner
        of the sign is Reggie Jackson, a Yankee hall-of-famer."""]
        test_target = ["""
        An auction for the lights from Yankee Stadium failed to
        produce any bids on Wednesday at Sotheby’s. The lights,
        currently owned by former Yankees player Reggie Jackson,
        lit the stadium from 1976 until 2008."""]

        return test_output, test_target


    def test_evaluate(self):
        print(f"{'#'*10} test_evaluate STARTS {'#'*10}")

        for metric_class in SUPPORTED_EVALUATION_METRICS:
            print(f"Test on {metric_class}")
            metric = metric_class()

            test_output, test_target = self.get_summary_pairs()
            score_dict = metric.evaluate(test_output, test_target)
            print(f"{metric_class} output dictionary")
            print(score_dict)
            self.assertIs(score_dict, Dict[str, float])
            self.assertNotEqual(score_dict, {})
            for key in score_dict:
                self.assertTrue(self.range[0] <= score_dict[key])
                self.assertTrue(score_dict[key] <= self.range[1])

                
if __name__ = '__main__':
    unittest.main()
