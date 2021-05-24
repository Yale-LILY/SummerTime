from .Model import Model
from summa.summarizer import summarize

class text_rank(Model):
    def __init__(self, num_sentences=1):
        self.num_sentences = num_sentences

    def summarize(self, corpus):
        return [summarize(corpus)]

    def show_capability(self):
        print("A graphbased ranking model for text processing. Extractive sentence summarization. \n Strengths: \n - Fast with low memory usage \n - Allows for control of summary length \n Weaknesses: \n - Not as accurate as neural methods.")
