from typing import List, Union

from model.dialogue.base_dialogue_model import DialogueSummModel
from model.base_model import SummModel
from model.single_doc.bart_model import BartModel


class FlattenDialogueModel(DialogueSummModel):

    model_name = "FlattenDialogueModel"

    def __init__(self, model_backend: SummModel = BartModel, **kwargs):
        self.model: SummModel = model_backend(**kwargs)

        super(DialogueSummModel, self).__init__(
            trained_domain=self.model.trained_domain,
            max_input_length=self.model.max_input_length,
            max_output_length=self.model.max_output_length,
        )

    def summarize(
        self,
        corpus: Union[List[str], List[List[str]]],
        query: Union[List[str], List[List[str]]] = None,
    ) -> List[str]:
        self.assert_summ_input_type(corpus, None)
        joint_corpus = []
        for instance in corpus:
            joint_corpus.append(" ".join(instance))

        summaries = self.model.summarize(joint_corpus)

        return summaries

    @classmethod
    def generate_basic_description(cls) -> str:
        basic_description = (
            "FlattenDialogueModel performs multi-document summarization by"
            " first concatenating all dialogue utterances,"
            " and then treat the concatenated text as a single document and use"
            " single document models to solve it."
        )
        return basic_description

    @classmethod
    def show_capability(cls):
        basic_description = cls.generate_basic_description()
        more_details = (
            "A dialogue summarization model."
            " Allows for custom model backend selection at initialization."
            " Concatenates the utterances in the dialogue and returns single-document summarization of joint corpus.\n"
            "Strengths: \n - Allows for control of backend model.\n"
            "Weaknesses: \n - Disregards the dialogue structure.\n"
        )
        print(f"{basic_description}\n{'#' * 20}\n{more_details}")
