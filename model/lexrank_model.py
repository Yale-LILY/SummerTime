from lexrank import STOPWORDS
from lexrank import LexRank as LR
import nltk

from .Model import Model

class lex_rank(Model):
    def __init__(self, data, summary_length=2, threshold=.1):
        nltk.download('punkt', quiet=True)
        corpus = [nltk.sent_tokenize(example) for example in data]
        self.lxr = LR(corpus, stopwords=STOPWORDS['en'])
        self.summary_length = summary_length
        self.threshold = threshold

    def summarize(self, documents):
        documents = [nltk.sent_tokenize(document) for document in documents]
        summaries = [self.lxr.get_summary(document, summary_size=self.summary_length, threshold=self.threshold) for document in documents]

        return summaries


    def show_capability(self):
        print("A non-neural model for extractive summarization. \n Works by using a graph-based method to identify the most salient sentences in the document. \n Strengths: \n - Fast with low memory usage \n - Allows for control of summary length \n Weaknesses: \n - Not as accurate as neural methods. \n Initialization arguments: \n - `corpus`: Unlabelled corpus of documents. ` \n - `summary_length`: sentence length of summaries \n - `threshold`: Level of salience required for sentence to be included in summary.")
