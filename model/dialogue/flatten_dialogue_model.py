import re

from itertools import chain
from typing import List, Union

from model.base_model import SummModel
from model.single_doc.textrank_model import TextRankModel


class FlattenDialogueModel(SummModel):

    model_name = "FlattenDialogueModel"
    is_dialogue_based = True

    def __init__(
        self, model_backend: SummModel = TextRankModel, **kwargs
    ):
        super(SummModel, self).__init__()
        self.model = model_backend(**kwargs)

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
    def assert_summ_input_type(cls, corpus: Union[List[str], List[List[str]]], queries: Union[List[str], None]):
        """each instance must be a list of \"speaker : utterance\""""
        assert all([isinstance(instance, list) for instance in corpus])

        pattern = re.compile(r"\w+\s:\s\w+")
        assert all([pattern.match(instance) for instance in chain.from_iterable(corpus)]), \
            "each instance must be a list of \"[speaker] : [utterance]\", the \":\" is essential"

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
