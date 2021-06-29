from .base_query_based_model import QueryBasedSummModel
from typing import Union, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFModel(QueryBasedSummModel):

    # static variables
    model_name = "TF-IDF"
    is_extractive = True
    is_neural = True  # TODO not sure about this value

    def __init__(self):
        super(TFIDFModel, self).__init__()
        self.vectorizer = TfidfVectorizer()

    def _retrieve(self, instance: List[str], query: List[str], n_best):
        instance_vectors = self.vectorizer.fit_transform(instance)
        query_vector = self.vectorizer.transform(query)

        similarities = cosine_similarity(query_vector, instance_vectors).squeeze()
        top_n_index = similarities.argsort()[::-1][0:n_best]
        top_n_sent = [instance[ind] for ind in top_n_index]  # List[str]
        return top_n_sent


