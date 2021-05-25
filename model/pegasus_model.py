from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from .model import SummModel


class PegasusModel(SummModel):
    def __init__(self, device='cpu'):
        super(PegasusModel, self).__init__("Pegasus", is_extractive=False, is_neural=True)
        
        self.device=device
        model_name = 'google/pegasus-xsum'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    def summarize(self, corpus):
        batch = self.tokenizer(corpus, truncation=True, padding='longest', return_tensors="pt").to(self.device)
        encoded_summaries = self.model.generate(**batch)
        summaries = self.tokenizer.batch_decode(encoded_summaries, skip_special_tokens=True)

        return summaries

    def show_capability(self):
        print("Introduced in 2019, a large neural abstractive summarization model trained on web crawl and news data.\n"
              "Strengths: \n - High accuracy \n - Performs well on almost all kinds of non-literary written text \n "
              "Weaknesses: \n - High memory usage \n "
              "Initialization arguments: \n "
              "- `device = 'cpu'` specifies the device the model is stored on and uses for computation. "
              "Use `device='gpu'` to run on an Nvidia GPU.")
