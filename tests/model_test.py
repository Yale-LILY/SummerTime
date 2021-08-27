import unittest
from typing import Tuple, List

from dataset.huggingface_datasets import CnndmDataset, MultinewsDataset, PubmedqaDataset, SamsumDataset
from dataset.non_huggingface_datasets import QMsumDataset
from model import SUPPORTED_SUMM_MODELS, list_all_models
from model.single_doc import LexRankModel, LongformerModel, PegasusModel
from model.multi_doc import MultiDocJointModel, MultiDocSeparateModel
from model.dialogue import HMNetModel

from helpers import print_with_color, get_summarization_set, get_query_based_summarization_set


class TestModels(unittest.TestCase):

    single_doc_dataset = CnndmDataset()
    multi_doc_dataset = MultinewsDataset()
    query_based_dataset = PubmedqaDataset()
    # # TODO: temporarily skipping HMNet, no dialogue-based dataset needed
    # dialogue_based_dataset = SamsumDataset()


    def test_list_models(self):
        print_with_color(f"{'#'*10} Testing test_list_models... {'#'*10}\n", "35")
        all_models = list_all_models()
        for model_class, model_description in all_models:
            print(f"{model_class} : {model_description}")
            self.assertTrue(True)
        self.assertEqual(len(all_models), len(SUPPORTED_SUMM_MODELS))
        print_with_color(f"{'#'*10} test_list_models {__name__} test complete {'#'*10}\n\n", "32")


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
        Test all supported models on instances from datasets.
        """

        print_with_color(f"{'#'*10} Testing all models... {'#'*10}\n", "35")

        num_models = 0
        all_models = list_all_models()

        for model_class, _ in all_models:
            if model_class in [PegasusModel, HMNetModel]:
                # TODO: Temporarily skip Pegasus (times out on Travis) and HMNet (requires large pre-trained model download + GPU)
                continue

            print_with_color(f"Testing {model_class.model_name} model...", "35")

            if model_class == LexRankModel:
                # current LexRankModel requires a training set
                training_src, training_tgt = get_summarization_set(self.single_doc_dataset, 100)
                model = model_class(training_src)
            else:
                model = model_class()

            if model.is_query_based:
                test_src, test_tgt, test_query = get_query_based_summarization_set(self.query_based_dataset, 1)
                prediction = model.summarize(test_src, test_query)
                print(f"Query: {test_query}\nGold summary: {test_tgt}\nPredicted summary: {prediction}")
            elif model.is_multi_document:
                test_src, test_tgt = get_summarization_set(self.multi_doc_dataset, 1)
                prediction = model.summarize(test_src)
                print(f"Gold summary: {test_tgt} \nPredicted summary: {prediction}")
                self.validate_prediction(prediction, test_src)
            elif model.is_dialogue_based:
                test_src, test_tgt = get_summarization_set(self.dialogue_based_dataset, 1)
                prediction = model.summarize(test_src)
                print(f"Gold summary: {test_tgt}\nPredicted summary: {prediction}")
                self.validate_prediction(prediction, test_src)
            else:
                test_src, test_tgt = get_summarization_set(self.single_doc_dataset, 1)
                prediction = model.summarize([test_src[0] * 5] if model_class == LongformerModel else test_src)
                print(f"Gold summary: {test_tgt} \nPredicted summary: {prediction}")
                self.validate_prediction(prediction, [test_src[0] * 5] if model_class == LongformerModel else test_src)
            
            print_with_color(f"{model_class.model_name} model test complete\n", "32")
            num_models += 1

        print_with_color(f"{'#'*10} test_model_summarize complete ({num_models} models) {'#'*10}\n", "32")


if __name__ == '__main__':
    unittest.main()
