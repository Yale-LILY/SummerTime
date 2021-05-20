from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from .Model import Model

class bart(Model):
    def __init__(self, device='cpu'):
        self.device = device
        model_name = 'facebook/bart-large-cnn'
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, corpus):
        batch = self.tokenizer(corpus, truncation=True,
            padding = 'longest', return_tensors="pt").to(self.device)
        encoded_summaries = self.model.generate(**batch)
        summaries = self.tokenizer.batch_decode(encoded_summaries, skip_special_tokens=True)

        return summaries 
