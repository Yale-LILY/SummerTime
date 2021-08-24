from model.base_model import SummModel
from typing import List, Union


class MultiDocSummModel(SummModel):

    is_multi_document = True

    def __init__(self,
                 trained_domain: str = None,
                 max_input_length: int = None,
                 max_output_length: int = None,
                 ):
        super(MultiDocSummModel, self).__init__(trained_domain = trained_domain, max_input_length = max_input_length, max_output_length = max_output_length)

    @classmethod
    def assert_summ_input_type(cls, corpus, query):
        if not all([isinstance(ins, list) and all([isinstance(doc, str) for doc in ins]) for ins in corpus]):
            raise TypeError("Multi-document summarization models summarize instances of multiple documents (`List[List[str]]`).")

        if query is not None:
            if not isinstance(query, list):
                raise TypeError("Query-based single-document summarization requires query of `List[str]`.")
            if not all([isinstance(q, str) for q in query]):
                raise TypeError("Query-based single-document summarization requires query of `List[str]`.")
