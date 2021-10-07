from transformers import BartForConditionalGeneration, BartTokenizer
from .base_single_doc_model import SingleDocSummModel


class BartModel(SingleDocSummModel):

    # static variables
    model_name = "BART"
    is_extractive = False
    is_neural = False

    def __init__(self, device="cpu"):
        super(BartModel, self).__init__(
            trained_domain="News", max_input_length=1024, max_output_length=None
        )

        self.device = device
        model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, corpus, queries=None):
        self.assert_summ_input_type(corpus, queries)

        batch = self.tokenizer(
            corpus, truncation=True, padding="longest", return_tensors="pt"
        ).to(self.device)
        encoded_summaries = self.model.generate(**batch)
        summaries = self.tokenizer.batch_decode(
            encoded_summaries, skip_special_tokens=True
        )

        return summaries

    @classmethod
    def show_capability(cls) -> None:
        # TODO zhangir: add the show capability function for BART
        print(cls.generate_basic_description())
