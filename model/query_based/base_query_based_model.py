from model.base_model import SummModel
from model.single_doc import TextRankModel
from typing import List, Union

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *


class QueryBasedSummModel(SummModel):
    def __init__(self,
                 trained_domain: str = None,
                 max_input_length: int = None,
                 max_output_length: int = None,
                 model_backend: SummModel = TextRankModel, **kwargs
                 ):
        super(QueryBasedSummModel, self).__init__(trained_domain = trained_domain, max_input_length = max_input_length, max_output_length = max_output_length)
        self.model = model_backend(**kwargs)

    def _retrieve(self, instance: List[str], query: List[str], n_best) -> List[str]:
        raise NotImplementedError()

    def summarize(self, corpus: Union[List[str], List[List[str]]], queries: List[str] = None,
                  retrieval_ratio: float = 0.5, preprocess=True) -> List[str]:
        self.assert_summ_input_type(corpus, queries)

        retrieval_output = [] # List[str]
        for instance, query in zip(corpus, queries):
            if isinstance(instance, str):
                is_dialogue = False
                instance = sent_tokenize(instance)
            else:
                is_dialogue = True
            query = [query]

            # instance & query now are List[str] for sure
            if preprocess:
                preprocessor = Preprocessor()
                instance = preprocessor.preprocess(instance)
                query = preprocessor.preprocess(query)

            n_best = max(int(len(instance) * retrieval_ratio), 1)
            top_n_sent = self._retrieve(instance, query, n_best)

            if not is_dialogue:
                top_n_sent = ' '.join(top_n_sent)  # str
            retrieval_output.append(top_n_sent)

        summaries = self.model.summarize(retrieval_output) # List[str] or List[List[str]]
        return summaries


    @classmethod
    def assert_summ_input_type(cls, corpus, query):
        if query is None:
            raise TypeError("Query-based summarization models summarize instances of query-text pairs， however, query is missing.")

        if not isinstance(query, list):
            raise TypeError("Query-based single-document summarization requires query of `List[str]`.")
        if not all([isinstance(q, str) for q in query]):
            raise TypeError("Query-based single-document summarization requires query of `List[str]`.")

    @classmethod
    def generate_basic_description(cls) -> str:
        basic_description = ("QueryBasedSummModel performs query-based summarization. Given a query-text pair,"
                             "the model will first extract the most relevant sentences in articles or turns in "
                             "dialogues, then use the single document summarization model to generate the summary")
        return basic_description

    @classmethod
    def show_capability(cls):
        basic_description = cls.generate_basic_description()
        more_details = ("A query-based summarization model."
                        " Allows for custom model backend selection at initialization."
                        " Retrieve relevant turns and then summarize the retrieved turns\n"
                        "Strengths: \n - Allows for control of backend model.\n"
                        "Weaknesses: \n - Heavily depends on the performance of both retriever and summarizer.\n")
        print(f"{basic_description}\n{'#' * 20}\n{more_details}")


class Preprocessor:
    def __init__(self):
        self.sw = stopwords.words('english')
        self.stemmer = PorterStemmer()

    def preprocess(self, corpus: List[str], remove_stopwords=True, lower_case=True, stem=False) -> List[str]:
        if lower_case:
            corpus = [sent.lower() for sent in corpus]
        tokenized_corpus = [word_tokenize(sent) for sent in corpus]
        if remove_stopwords:
            tokenized_corpus = [[word for word in sent if word not in self.sw] for sent in tokenized_corpus]
        if stem:
            tokenized_corpus = [[self.stemmer.stem(word) for word in sent] for sent in tokenized_corpus]
        return [' '.join(sent) for sent in tokenized_corpus]

