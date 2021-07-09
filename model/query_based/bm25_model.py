from .base_query_based_model import QueryBasedSummModel
from typing import Union, List

from gensim.summarization.bm25 import BM25
from nltk import sent_tokenize, word_tokenize


class BM25Model(QueryBasedSummModel):

    # static variables
    model_name = "BM25"
    is_extractive = True
    is_neural = True  # TODO not sure about this value

    def __init__(self):
        super(BM25Model, self).__init__()

    def _retrieve(self, instance: List[str], query: List[str], n_best):
        bm25 = BM25(word_tokenize(s) for s in instance)
        scores = bm25.get_scores(query)
        best_sent_ind = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_best]
        top_n_sent = [instance[ind] for ind in sorted(best_sent_ind)]
        return top_n_sent
