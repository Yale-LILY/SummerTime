from .base_query_based_model import QueryBasedSummModel
from model.base_model import SummModel
from typing import Union, List

from gensim.summarization.bm25 import BM25
from nltk import sent_tokenize, word_tokenize


class BM25SummModel(QueryBasedSummModel):

    # static variables
    model_name = "BM25"
    is_extractive = True  # only represents the retrieval part
    is_neural = False  # only represents the retrieval part
    is_query_based = True

    def __init__(self,
                 trained_domain: str = None,
                 max_input_length: int = None,
                 max_output_length: int = None,
                 model_backend: SummModel = None,
                 retrieval_ratio: float = 0.5,
                 preprocess: bool = True,
                 **kwargs
                 ):
        super(BM25SummModel, self).__init__(trained_domain=trained_domain,
                                            max_input_length=max_input_length,
                                            max_output_length=max_output_length,
                                            model_backend=model_backend,
                                            retrieval_ratio=retrieval_ratio,
                                            preprocess=preprocess,
                                            **kwargs)

    def _retrieve(self, instance: List[str], query: List[str], n_best):
        bm25 = BM25(word_tokenize(s) for s in instance)
        scores = bm25.get_scores(query)
        best_sent_ind = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_best]
        top_n_sent = [instance[ind] for ind in sorted(best_sent_ind)]
        return top_n_sent

