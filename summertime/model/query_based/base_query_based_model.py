from summertime.model.base_model import SummModel
from summertime.model.single_doc import TextRankModel
from typing import List, Union

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class QueryBasedSummModel(SummModel):

    is_query_based = True

    def __init__(
        self,
        trained_domain: str = None,
        max_input_length: int = None,
        max_output_length: int = None,
        model_backend: SummModel = TextRankModel,
        retrieval_ratio: float = 0.5,
        preprocess: bool = True,
        **kwargs,
    ):
        super(QueryBasedSummModel, self).__init__(
            trained_domain=trained_domain,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )
        self.model = model_backend(**kwargs)
        self.retrieval_ratio = retrieval_ratio
        self.preprocess = preprocess

    def _retrieve(self, instance: List[str], query: List[str], n_best) -> List[str]:
        raise NotImplementedError()

    def summarize(
        self,
        corpus: Union[List[str], List[List[str]]],
        queries: List[str] = None,
    ) -> List[str]:
        self.assert_summ_input_type(corpus, queries)

        retrieval_output = []  # List[str]
        for instance, query in zip(corpus, queries):
            if isinstance(instance, str):
                is_dialogue = False
                instance = sent_tokenize(instance)
            else:
                is_dialogue = True
            query = [query]

            # instance & query now are List[str] for sure
            if self.preprocess:
                preprocessor = Preprocessor()
                instance = preprocessor.preprocess(instance)
                query = preprocessor.preprocess(query)

            n_best = max(int(len(instance) * self.retrieval_ratio), 1)
            top_n_sent = self._retrieve(instance, query, n_best)

            if not is_dialogue:
                top_n_sent = " ".join(top_n_sent)  # str
            retrieval_output.append(top_n_sent)

        summaries = self.model.summarize(
            retrieval_output
        )  # List[str] or List[List[str]]
        return summaries

    def generate_specific_description(self):
        is_neural = self.model.is_neural & self.is_neural
        is_extractive = self.model.is_extractive | self.is_extractive
        model_name = "Pipeline with retriever: {}, summarizer: {}".format(
            self.model_name, self.model.model_name
        )

        extractive_abstractive = "extractive" if is_extractive else "abstractive"
        neural = "neural" if is_neural else "non-neural"

        basic_description = (
            f"{model_name} is a "
            f"{'query-based' if self.is_query_based else ''} "
            f"{extractive_abstractive}, {neural} model for summarization."
        )

        return basic_description

    @classmethod
    def assert_summ_input_type(cls, corpus, query):
        if query is None:
            raise TypeError(
                "Query-based summarization models summarize instances of query-text pairs， however, query is missing."
            )

        if not isinstance(query, list):
            raise TypeError(
                "Query-based single-document summarization requires query of `List[str]`."
            )
        if not all([isinstance(q, str) for q in query]):
            raise TypeError(
                "Query-based single-document summarization requires query of `List[str]`."
            )

    @classmethod
    def generate_basic_description(cls) -> str:
        basic_description = (
            "QueryBasedSummModel performs query-based summarization. Given a query-text pair,"
            "the model will first extract the most relevant sentences in articles or turns in "
            "dialogues, then use the single document summarization model to generate the summary"
        )
        return basic_description

    @classmethod
    def show_capability(cls):
        basic_description = cls.generate_basic_description()
        more_details = (
            "A query-based summarization model."
            " Allows for custom model backend selection at initialization."
            " Retrieve relevant turns and then summarize the retrieved turns\n"
            "Strengths: \n - Allows for control of backend model.\n"
            "Weaknesses: \n - Heavily depends on the performance of both retriever and summarizer.\n"
        )
        print(f"{basic_description}\n{'#' * 20}\n{more_details}")


class Preprocessor:
    def __init__(self, remove_stopwords=True, lower_case=True, stem=False):
        self.sw = stopwords.words("english")
        self.stemmer = PorterStemmer()
        self.remove_stopwords = remove_stopwords
        self.lower_case = lower_case
        self.stem = stem

    def preprocess(self, corpus: List[str]) -> List[str]:
        if self.lower_case:
            corpus = [sent.lower() for sent in corpus]
        tokenized_corpus = [word_tokenize(sent) for sent in corpus]
        if self.remove_stopwords:
            tokenized_corpus = [
                [word for word in sent if word not in self.sw]
                for sent in tokenized_corpus
            ]
        if self.stem:
            tokenized_corpus = [
                [self.stemmer.stem(word) for word in sent] for sent in tokenized_corpus
            ]
        return [" ".join(sent) for sent in tokenized_corpus]
