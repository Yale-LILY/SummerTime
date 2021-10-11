from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from .base_single_doc_model import SingleDocSummModel


class MBartModel(SingleDocSummModel):
    # static variables
    model_name = "mBART"
    is_extractive = False
    is_neural = True
    is_multilingual = True

    def __init__(self, device="cpu"):
        super(MBartModel, self).__init__(
            # TODO: trained domain not news (at least not exclusively)
            trained_domain="News",
            max_input_length=1024,
            max_output_length=None,
        )

        self.device = device

        model_name = "facebook/mbart-large-50"
        self.tokenizer = MBart50Tokenizer.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(
            device
        )

    def summarize(self, corpus, queries=None):
        self.assert_summ_input_type(corpus, queries)

        batch = self.tokenizer(
            corpus, truncation=True, padding="longest", return_tensors="pt"
        ).to(self.device)
        encoded_summaries = self.model.generate(**batch)
        # num_beams=4, max_length=5, early_stopping=True
        # add hyperparameters to .generate? above are what's used in the huggingface docs
        summaries = self.tokenizer.batch_decode(
            encoded_summaries, skip_special_tokens=True
        )

        return summaries

    @classmethod
    def show_capability(cls) -> None:
        basic_description = cls.generate_basic_description()
        more_details = (
            "Introduced in 2020, a multilingual variant of BART (a large neural model) "
            "trained on web crawl data.\n"
            "Strengths: \n - Multilinguality: supports 50 different languages\n"
            "Weaknesses: \n - High memory usage"
            "Initialization arguments: \n "
            "- `device = 'cpu'` specifies the device the model is stored on and uses for computation. "
            "Use `device='gpu'` to run on an Nvidia GPU."
        )
        print(f"{basic_description} \n {'#'*20} \n {more_details}")
