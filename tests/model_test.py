import unittest
from typing import List

from summertime.dataset.dataset_loaders import (
    CnndmDataset,
    MultinewsDataset,
    PubmedqaDataset,
    SamsumDataset,
)
from summertime.model import SUPPORTED_SUMM_MODELS, list_all_models
from summertime.model.single_doc import LexRankModel, LongformerModel
from summertime.model.dialogue import HMNetModel

from helpers import (
    print_with_color,
    get_summarization_set,
    get_query_based_summarization_set,
)

DUMMY_DOC_INPUT = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions."
    " The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected"
    " by the shutoffs which were expected to last through at least midday tomorrow."
)

DUMMY_DOC_OUTPUT = "California's largest electricity provider has turned off power to hundreds of thousands of customers."

DUMMY_DIALOGUE_INPUT = [
    "Alice : I am a girl.",
    "Bob : I am a boy.",
]

DUMMY_QUERY_INPUT = "What is the main topic of this?"


def get_dummy_single_doc_instances(n: int):
    return [DUMMY_DOC_INPUT] * n


def get_dummy_multi_doc_instances(n: int, m: int = 5):
    return [[DUMMY_DOC_INPUT] * m for _ in range(n)]


def get_dummy_query_based_instances(n: int):
    return [DUMMY_DOC_INPUT] * n, [DUMMY_QUERY_INPUT] * n


def get_dummy_dialogue_instances(n: int):
    return [DUMMY_DIALOGUE_INPUT for _ in range(n)]


class TestIndividualModels(unittest.TestCase):
    """more tests for different aspects of the models"""

    def test_hmnet_model(self):
        from summertime.model.dialogue.hmnet_model import HMNetModel

        dummy_corpus = get_dummy_dialogue_instances(2)
        model = HMNetModel(min_gen_length=10, max_gen_length=30, beam_width=2)
        result = model.summarize(dummy_corpus)

        assert all([isinstance(r, str) for r in result])


class TestModels(unittest.TestCase):
    def test_list_models(self):
        print_with_color(f"{'#'*10} Testing test_list_models... {'#'*10}\n", "35")
        all_models = list_all_models()
        for model_class, model_description in all_models:
            print(f"{model_class} : {model_description}")
            self.assertTrue(True)
        self.assertEqual(len(all_models), len(SUPPORTED_SUMM_MODELS))
        print_with_color(
            f"{'#'*10} test_list_models {__name__} test complete {'#'*10}\n\n", "32"
        )

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

        single_doc_dataset = CnndmDataset()
        multi_doc_dataset = MultinewsDataset()
        query_based_dataset = PubmedqaDataset()
        dialogue_based_dataset = SamsumDataset()

        print_with_color(f"{'#'*10} Testing all models... {'#'*10}\n", "35")

        num_models = 0
        all_models = list_all_models()

        for model_class, _ in all_models:
            print_with_color(f"Testing {model_class.model_name} model...", "35")

            if model_class in [HMNetModel]:
                model = HMNetModel(min_gen_length=10, max_gen_length=100, beam_width=2)

            if model_class == LexRankModel:
                # current LexRankModel requires a training set
                training_src, training_tgt = get_summarization_set(
                    single_doc_dataset, 100
                )
                model = model_class(training_src)
            else:
                model = model_class()

            if model.is_query_based:
                test_src, test_tgt, test_query = get_query_based_summarization_set(
                    query_based_dataset, 1
                )
                prediction = model.summarize(test_src, test_query)
                print(
                    f"Query: {test_query}\nGold summary: {test_tgt}\nPredicted summary: {prediction}"
                )
            elif model.is_multi_document:
                test_src, test_tgt = get_summarization_set(multi_doc_dataset, 1)
                prediction = model.summarize(test_src)
                print(f"Gold summary: {test_tgt} \nPredicted summary: {prediction}")
                self.validate_prediction(prediction, test_src)
            elif model.is_dialogue_based:
                test_src, test_tgt = get_summarization_set(dialogue_based_dataset, 1)
                prediction = model.summarize(test_src)
                print(f"Gold summary: {test_tgt}\nPredicted summary: {prediction}")
                self.validate_prediction(prediction, test_src)
            else:
                test_src, test_tgt = get_summarization_set(single_doc_dataset, 1)
                prediction = model.summarize(
                    [test_src[0] * 5] if model_class == LongformerModel else test_src
                )
                print(f"Gold summary: {test_tgt} \nPredicted summary: {prediction}")
                self.validate_prediction(
                    prediction,
                    [test_src[0] * 5] if model_class == LongformerModel else test_src,
                )

            print_with_color(f"{model_class.model_name} model test complete\n", "32")
            num_models += 1

        print_with_color(
            f"{'#'*10} test_model_summarize complete ({num_models} models) {'#'*10}\n",
            "32",
        )


if __name__ == "__main__":
    unittest.main()
