from model.base_model import SummModel
from model.textrank_model import TextRankModel
from typing import Union, List


class MultiDocJointModel(SummModel):

    def __init__(self, model_backend: SummModel = TextRankModel, **kwargs):
        super(MultiDocJointModel, self).__init__()
        model = model_backend(**kwargs)
        self.model = model

    def summarize(self, corpus: Union[List[str], List[List[str]]]) -> str:
        flattened_corpus = ["\n".join(x) if type(x[0]) is list else x for x in corpus]
        joint_corpus = [" ".join(flattened_corpus)]

        summary = self.model.summarize(joint_corpus)
        return summary[0]

    @classmethod
    def show_capability(cls):
        more_details = ("A multi-document summarization model."
                        " Allows for custom model backend selection at initialization."
                        " Concatenates each document in corpus and returns single-document summarization of joint corpus.\n"
                        "Strengths: \n - Allows for control of backend model.\n"
                        "Weaknesses: \n - Assumes all documents are equally weighted.\n"
                        " - May fail to extract information from certain documents.\n")
        print(more_details)
