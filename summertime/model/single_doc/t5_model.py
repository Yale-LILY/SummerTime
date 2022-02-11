from transformers import T5Tokenizer, T5ForConditionalGeneration
from .base_single_doc_model import SingleDocSummModel


class T5Model(SingleDocSummModel):

    # static variables
    model_name = "T5"
    is_extractive = False
    is_neural = True

    def __init__(self, device="cpu"):
        super(T5Model, self).__init__(
            trained_domain="Web Crawl", max_input_length=1024, max_output_length=None
        )

        self.device = device
        model_name = "t5-large"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

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
        basic_description = cls.generate_basic_description()
        more_details = (
            "Introduced in 2020, T5 is a large pretrained language model trained on web crawl using "
            "transfer learning approaches and teacher forcing.\n "
            "Strengths: \n - High accuracy \n "
            "Weaknesses: \n - High memory usage \n "
            "Initialization arguments: \n "
            "- `device = 'cpu'` specifies the device the model is stored on and uses for computation. "
            "Use `device='gpu'` to run on an Nvidia GPU."
        )
        print(f"{basic_description} \n {'#'*20} \n {more_details}")
