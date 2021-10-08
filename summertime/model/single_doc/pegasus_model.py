from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from .base_single_doc_model import SingleDocSummModel


class PegasusModel(SingleDocSummModel):
    # static variables
    model_name = "Pegasus"
    is_extractive = False
    is_neural = True

    def __init__(self, device="cpu"):
        super(PegasusModel, self).__init__(
            trained_domain="News", max_input_length=1024, max_output_length=None
        )

        self.device = device
        model_name = "google/pegasus-xsum"
        print("init load pretrained tokenizer")
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        print("init load pretrained model with tokenizer on " + device)
        # self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    def summarize(self, corpus, queries=None):
        self.assert_summ_input_type(corpus, queries)

        print("batching")
        # batch = self.tokenizer(corpus, truncation=True, padding='longest', return_tensors="pt").to(self.device)
        batch = self.tokenizer(corpus, truncation=True, return_tensors="pt").to(device)
        print("encoding batches")
        # encoded_summaries = self.model.generate(**batch, max_length=40, max_time=120)
        encoded_summaries = self.model.generate(batch["input_ids"], max_time=1024)
        print("decoding batches")
        # summaries = self.tokenizer.batch_decode(encoded_summaries, skip_special_tokens=True)
        summaries = [self.tokenizer.decode(encoded_summaries[0])]

        return summaries

    @classmethod
    def show_capability(cls):
        basic_description = cls.generate_basic_description()
        more_details = (
            "Introduced in 2019, a large neural abstractive summarization model trained on web crawl and "
            "news data.\n "
            "Strengths: \n - High accuracy \n - Performs well on almost all kinds of non-literary written "
            "text \n "
            "Weaknesses: \n - High memory usage \n "
            "Initialization arguments: \n "
            "- `device = 'cpu'` specifies the device the model is stored on and uses for computation. "
            "Use `device='gpu'` to run on an Nvidia GPU."
        )
        print(f"{basic_description} \n {'#'*20} \n {more_details}")
