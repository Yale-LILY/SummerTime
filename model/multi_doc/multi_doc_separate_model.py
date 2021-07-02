from .base_multi_doc_model import MultiDocSummModel
from model.base_model import SummModel
from model.single_doc import TextRankModel
from typing import Union, List


class MultiDocSeparateModel(MultiDocSummModel):

    model_name = "Multi-document separate"
    is_multi_document = True

    def __init__(self, model_backend: SummModel = TextRankModel, **kwargs):
        super(MultiDocSeparateModel, self).__init__()
        model = model_backend(**kwargs)
        self.model = model

    def summarize(self, corpus: Union[List[str], List[List[str]]]) -> List[str]:
        self.assert_summ_input_type(corpus, None)
        summaries = []
        for instance in corpus:
            instance_summaries = self.model.summarize(instance)
            summaries.append(" ".join(instance_summaries))

        return summaries

    @classmethod
    def generate_basic_description(cls) -> str:
        basic_description = ("MultiDocSeparateModel performs multi-document summarization by"
                             " first performing single-document summarization on each document,"
                             " and then concatenating the results.")
        return basic_description

    @classmethod
    def show_capability(cls):
        basic_description = cls.generate_basic_description()
        more_details = ("A multi-document summarization model."
                        " Allows for custom model backend selection at initialization."
                        " Performs single-document summarization on each document in corpus and returns concatenated result.\n"
                        "Strengths: \n - Allows for control of backend model.\n"
                        "Weaknesses: \n - Assumes all documents are equally weighted.\n - May produce redundant information for similar documents.\n")
        print(f"{basic_description}\n{'#' * 20}\n{more_details}")
