import unittest
from typing import Tuple, List

from dataset.huggingface_datasets import CnndmDataset
from model import SUPPORTED_SUMM_MODELS, list_all_models
from model.single_doc import LexRankModel, LongformerModel
from model.multi_doc import MultiDocJointModel, MultiDocSeparateModel


class TestModels(unittest.TestCase):
    dataset = CnndmDataset()

    def get_summarization_set(self, size: int = 1) -> Tuple[List[str], List[str]]:
        """
        return some dummy summarization examples, in the format of (a list of sources, a list of targets)
        """
        subset = []
        for i in range(size):
            subset.append(next(self.dataset.train_set))

        src, tgt = zip(*(list(map(lambda x: (x.source, x.summary), subset))))

        return list(src), list(tgt)

    def test_list_models(self):
        print(f"{'#'*10} test_list_models STARTS {'#'*10}")
        all_models = list_all_models()
        for model_class, model_description in all_models:
            print(f"{model_class} : {model_description}")
            self.assertTrue(True)
        self.assertEqual(len(all_models), len(SUPPORTED_SUMM_MODELS))
        print(f"{'#'*10} test_list_models {__name__} ENDS {'#'*10}\n\n")

    def test_model_summarize(self):
        print(f"{'#'*10} test_model_summarize STARTS {'#'*10}")
        all_models = list_all_models()
        for model_class, _ in all_models:
            print(f"\nTest on {model_class}")

            if model_class == LexRankModel:
                # current LexRankModel requires a training set
                training_src, training_tgt = self.get_summarization_set(100)
                model = model_class(training_src)
            else:
                model = model_class()

            if isinstance(model, (MultiDocJointModel, MultiDocSeparateModel)):
                test_src, test_tgt = self.get_summarization_set(3)
                prediction = model.summarize([test_src])
                print(f"Gold summary: {test_tgt} \nPredicted summary: {prediction}")
            else:
                test_src, test_tgt = self.get_summarization_set(1)
                prediction = model.summarize([test_src[0] * 5] if model_class == LongformerModel else test_src)
                print(f"Gold summary: {test_tgt} \nPredicted summary: {prediction}")

        print(f"{'#'*10} test_model_summarize ENDS {'#'*10}\n\n")


if __name__ == '__main__':
    unittest.main()
