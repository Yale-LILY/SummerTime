from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from .base_model import SummModel


class PegasusModel(SummModel):
    # static variables
    model_name = "Pegasus"
    is_extractive = False
    is_neural = True

    def __init__(self, device='cpu'):
        super(PegasusModel, self).__init__()

        self.device = device
        model_name = 'google/pegasus-xsum'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    def summarize(self, corpus, queries=None):
        if not isinstance(corpus, list):
            raise TypeError("Pegasus single-document summarization requires corpus of `List[str]`.")
        for instance in corpus:
            if not type(instance) == str:
                raise TypeError("Pegasus single-document summarization requires corpus of `List[str]`.")

        batch = self.tokenizer(corpus, truncation=True, padding='longest', return_tensors="pt").to(self.device)
        encoded_summaries = self.model.generate(**batch)
        summaries = self.tokenizer.batch_decode(encoded_summaries, skip_special_tokens=True)

        return summaries

    @classmethod
    def show_capability(cls):
        basic_description = cls.generate_basic_description()
        more_details = ("Introduced in 2019, a large neural abstractive summarization model trained on web crawl and "
                        "news data.\n "
                        "Strengths: \n - High accuracy \n - Performs well on almost all kinds of non-literary written "
                        "text \n "
                        "Weaknesses: \n - High memory usage \n "
                        "Initialization arguments: \n "
                        "- `device = 'cpu'` specifies the device the model is stored on and uses for computation. "
                        "Use `device='gpu'` to run on an Nvidia GPU.")
        print(f"{basic_description} \n {'#'*20} \n {more_details}")
