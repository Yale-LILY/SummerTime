from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch 
from Model import Model

class pegasus(Model): 
    def __init__(self, device='cpu'):
        self.device=device
        model_name = 'google/pegasus-xsum'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    def summarize(self, corpus): 
        batch = self.tokenizer(corpus, truncation=True, padding='longest', return_tensors="pt").to(self.device)
        encoded_summaries = self.model.generate(**batch)
        summaries = self.tokenizer.batch_decode(encoded_summaries, skip_special_tokens=True)

        return summaries 



