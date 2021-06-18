from model.base_model import SummModel
from model.textrank_model import TextRankModel
from typing import Union, List


class MultiDocSeparateModel(SummModel):

    def __init__(self, model_backend: SummModel = TextRankModel, **kwargs):
        super(MultiDocSeparateModel, self).__init__()
        model = model_backend(**kwargs)
        self.model = model
    
    def summarize(self, corpus: Union[List[str], List[List[str]]]) -> List[str]:
        summaries = []
        for instance in corpus:
            print(type(instance))
            if not isinstance(instance, list):
                raise TypeError("Multi-document summarization models summarize instances of multiple documents (`List[List[str]]`).")

            instance_summaries = self.model.summarize(instance)
            summaries.append(" ".join(instance_summaries))

        return summaries
    
    @classmethod
    def show_capability(cls):
        more_details = ("A multi-document summarization model."
                        " Allows for custom model backend selection at initialization."
                        " Performs single-document summarization on each document in corpus and returns concatenated result.\n"
                        "Strengths: \n - Allows for control of backend model.\n"
                        "Weaknesses: \n - Assumes all documents are equally weighted.\n - May produce redundant information for similar documents.\n")
        print(more_details)
