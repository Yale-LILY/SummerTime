from model.base_model import SummModel
from typing import List, Union


class SingleDocSummModel(SummModel):
    def __init__(self,
                 trained_domain: str = None,
                 max_input_length: int = None,
                 max_output_length: int = None,
                 ):
        super(SingleDocSummModel, self).__init__(trained_domain = trained_domain, max_input_length = max_input_length, max_output_length = max_output_length)

    def summarize(self,
                  corpus: Union[List[str], List[List[str]]],
                  queries: List[str]=None) -> List[str]:
        """
        :param corpus: each string in the list is a source document to be summarized
        :param queries: a list of queries if this is a query-based model
        :return: a list of generated summaries
        """
        raise NotImplementedError("The base class for single-document models shouldn't be instantiated!")

    @classmethod
    def assert_summ_input_type(cls, corpus, query):
        if not isinstance(corpus, list):
            raise TypeError("Single-document summarization requires corpus of `List[str]`.")
        for instance in corpus:
            if not type(instance) == str:
                raise TypeError("Single-document summarization requires corpus of `List[str]`.")

        if query is not None:
            if not isinstance(query, list):
                raise TypeError("Query-based single-document summarization requires query of `List[str]`.")
            for q in query:
                if not type(q) == str:
                    raise TypeError("Query-based single-document summarization requires query of `List[str]`.")

    @classmethod
    def show_capability(cls) -> None:
        """
        Use concise language to show the strength and weakness for each model. Try not to use NLP terminologies
        """
        raise NotImplementedError("The base class for models shouldn't be instantiated!")


    @classmethod
    def generate_basic_description(cls) -> str:
        """
        Automatically generate the basic description string based on the attributes
        """
        extractive_abstractive = "extractive" if cls.is_extractive else "abstractive"
        neural = "neural" if cls.is_neural else "non-neural"
                
        basic_description = f"{cls.model_name} is a" \
                            f"{'query-based' if cls.is_query_based else ''} " \
                            f"{extractive_abstractive}, {neural} model for summarization."
        if cls.is_multi_document or cls.is_dialogue_based:
            basic_description += f"It can handle {'multi-document' if cls.is_multi_document else ''} " \
                                f"{'dialogue' if cls.is_dialogue_based else ''} textual data."
                    
        return basic_description
