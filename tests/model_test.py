import unittest
from typing import Tuple, List

from dataset.huggingface_datasets import CnndmDataset, MultinewsDataset
from dataset.non_huggingface_datasets import QMsumDataset
from model import SUPPORTED_SUMM_MODELS, list_all_models
from model.single_doc import LexRankModel, LongformerModel
from model.multi_doc import MultiDocJointModel, MultiDocSeparateModel
from model.dialogue import HMNetModel

from helpers import print_with_color


class TestModels(unittest.TestCase):
    single_doc_dataset = CnndmDataset()
    multi_doc_dataset = MultinewsDataset()
    dialogue_based_dataset = QMsumDataset()

    def get_summarization_set(self, size: int = 1) -> Tuple[List[str], List[str]]:
        """
        return some dummy summarization examples, in the format of (a list of sources, a list of targets)
        """
        subset = []
        for i in range(size):
            subset.append(next(self.single_doc_dataset.train_set))

        src, tgt = zip(*(list(map(lambda x: (x.source, x.summary), subset))))

        return list(src), list(tgt)
    
    def get_multi_doc_summarization_set(self, size: int = 1) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Same as `get_summarization_set`, but for multi-document models.
        """
        subset = []
        for i in range(size):
            subset.append(next(self.multi_doc_dataset.train_set))

        src, tgt = zip(*(list(map(lambda x: (x.source, x.summary), subset))))

        return list(src), list(tgt)
    
    def get_dialogue_summarization_set(self, size: int = 1) -> Tuple[List[str], List[str], List[str]]:
        """
        Same as `get_summarization_set`, but for dialogue-based models.
        """
        subset = []
        for i in range(size):
            subset.append(self.dialogue_based_dataset.train_set[i])
        
        src, tgt, queries = zip(*(list(map(lambda x: (x.source, x.summary, x.query), subset))))

        return list(src), list(tgt), list(queries)

    def test_list_models(self):
        print_with_color(f"{'#'*10} test_list_models starts {'#'*10}", "35")
        all_models = list_all_models()
        for model_class, model_description in all_models:
            print(f"{model_class} : {model_description}")
            self.assertTrue(True)
        self.assertEqual(len(all_models), len(SUPPORTED_SUMM_MODELS))
        print_with_color(f"{'#'*10} test_list_models {__name__} ends {'#'*10}\n\n", "32")
    
    def validate_prediction(self, prediction: List[str], src: List):
        """
        Verify that prediction instances match source instances.
        """
        self.assertTrue(isinstance(prediction, list))
        self.assertTrue(all([isinstance(ins, str) for ins in prediction]))
        self.assertTrue(len(prediction) == len(src))
        print("Prediction typing and length matches source instances!")

    def test_model_summarize(self):
        """
        Test all supported models on instances from CNNDM or QMSumm datasets.
        """
        print_with_color(f"{'#'*10} Testing all models... {'#'*10}\n", "35")
        all_models = list_all_models()
        for model_class, _ in all_models:
            print_with_color(f"Testing {model_class.model_name} model...", "35")

            if model_class == LexRankModel:
                # current LexRankModel requires a training set
                training_src, training_tgt = self.get_summarization_set(100)
                model = model_class(training_src)
            elif model_class.is_query_based:
                # Ensure backend model matches for QMSumm
                model = model_class(model_backend=HMNetModel)
            else:
                model = model_class()

            if model.is_multi_document:
                test_src, test_tgt = self.get_multi_doc_summarization_set(1)
                prediction = model.summarize(test_src)
                print(f"Gold summary: {test_tgt} \nPredicted summary: {prediction}")
                self.validate_prediction(prediction, test_src)
            elif model.is_dialogue_based:
                test_src, test_tgt, test_query = self.get_dialogue_summarization_set(1)
                prediction = model.summarize(test_src, test_query)
                print(f"Query: {test_query}\nGold summary: {test_tgt}\nPredicted summary: {prediction}")
                self.validate_prediction(prediction, test_src)
            elif model.is_query_based:
                test_src, test_tgt, test_query = self.get_dialogue_summarization_set(1)
                prediction = model.summarize(test_src, test_query)
                print(f"Query: {test_query}\nGold summary: {test_tgt}\nPredicted summary: {prediction}")
            else:
                test_src, test_tgt = self.get_summarization_set(1)
                prediction = model.summarize([test_src[0] * 5] if model_class == LongformerModel else test_src)
                print(f"Gold summary: {test_tgt} \nPredicted summary: {prediction}")
                self.validate_prediction(prediction, [test_src[0] * 5] if model_class == LongformerModel else test_src)
            
            print_with_color(f"{model_class.model_name} model test complete\n", "32")

        print_with_color(f"{'#'*10} test_model_summarize complete {'#'*10}\n", "32")


if __name__ == '__main__':
    unittest.main()
