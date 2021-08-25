from .base_query_based_model import QueryBasedSummModel
from model.base_model import SummModel
from model.single_doc import TextRankModel
from typing import Union, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSummModel(QueryBasedSummModel):

    # static variables
    model_name = "TF-IDF"
    is_extractive = True
    is_neural = False
    is_query_based = True

    def __init__(self,
                 trained_domain: str = None,
                 max_input_length: int = None,
                 max_output_length: int = None,
                 model_backend: SummModel = TextRankModel,
                 retrieval_ratio: float = 0.5,
                 preprocess: bool = True,
                 **kwargs
                 ):
        super(TFIDFSummModel, self).__init__(trained_domain=trained_domain,
                                             max_input_length=max_input_length,
                                             max_output_length=max_output_length,
                                             model_backend=model_backend,
                                             retrieval_ratio=retrieval_ratio,
                                             preprocess=preprocess,
                                             **kwargs)
        self.vectorizer = TfidfVectorizer()

    def _retrieve(self, instance: List[str], query: List[str], n_best):
        instance_vectors = self.vectorizer.fit_transform(instance)
        query_vector = self.vectorizer.transform(query)

        similarities = cosine_similarity(query_vector, instance_vectors).squeeze()
        top_n_index = similarities.argsort()[::-1][0:n_best]
        top_n_sent = [instance[ind] for ind in top_n_index]  # List[str]
        return top_n_sent


