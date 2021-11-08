from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from .base_multilingual_model import MultilingualSummModel


class MT5Model(MultilingualSummModel):
    """
    MT5 Model for Multilingual Summarization
    """

    # static variables
    model_name = "mT5"
    is_extractive = False
    is_neural = True
    is_multilingual = True

    lang_tag_dict = {
        "am": "am",
        "ar": "ar",
        "az": "az",
        "bn": "bn",
        "my": "my",
        "zh-CN": "zh-CN",
        "zh-TW": "zh-TW",
        "en": "en",
        "fr": "fr",
        "gu": "gu",
        "ha": "ha",
        "hi": "hi",
        "ig": "ig",
        "id": "id",
        "ja": "ja",
        "rn": "rn",
        "ko": "ko",
        "ky": "ky",
        "mr": "mr",
        "np": "np",
        "om": "om",
        "ps": "ps",
        "fa": "fa",
        "pt": "pt",  # missing pidgin from XLSum--does not have ISO 639-1 code
        "pa": "pa",
        "ru": "ru",
        "gd": "gd",
        "sr": "sr",
        "si": "si",
        "so": "so",
        "es": "es",
        "sw": "sw",
        "ta": "ta",
        "te": "te",
        "th": "th",
        "ti": "ti",
        "tr": "tr",
        "uk": "uk",
        "ur": "ur",
        "uz": "uz",
        "vi": "vi",
        "cy": "cy",
        "yo": "yo",
    }  # TODO: add supported langs from mT5 that are not in the XLSum dataset (not finetuned on)

    def __init__(self, device="cpu"):

        super(MT5Model, self).__init__(
            trained_domain="News",
            max_input_length=512,
            max_output_length=None,
        )

        self.device = device

        model_name = "csebuetnlp/mT5_multilingual_XLSum"
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

    def summarize(self, corpus, queries=None):
        self.assert_summ_input_type(corpus, queries)

        self.assert_summ_input_language(corpus, queries)

        with self.tokenizer.as_target_tokenizer():
            batch = self.tokenizer(
                corpus,
                truncation=True,
                padding="longest",
                max_length=self.max_input_length,
                return_tensors="pt",
            ).to(self.device)

        encoded_summaries = self.model.generate(
            **batch, num_beams=4, length_penalty=1.0, early_stopping=True
        )

        summaries = self.tokenizer.batch_decode(
            encoded_summaries, skip_special_tokens=True
        )

        return summaries

    @classmethod
    def show_capability(cls) -> None:
        basic_description = cls.generate_basic_description()
        more_details = (
            "Introduced in ____, a massively multilingual variant of Google's T5, a large neural model. "
            "Trained on web crawled data and fine-tuned on XLSum, a 45-language multilingual news dataset.\n"
            "Strengths: \n - Massively multilingual: supports 101 different languages\n"
            "Weaknesses: \n - High memory usage\n - Lower max input length (512)"
            "Initialization arguments: \n "
            "- `device = 'cpu'` specifies the device the model is stored on and uses for computation. "
            "Use `device='cuda'` to run on an Nvidia GPU."
        )
        print(f"{basic_description} \n {'#'*20} \n {more_details}")