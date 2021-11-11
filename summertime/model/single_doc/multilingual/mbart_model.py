from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from .base_multilingual_model import MultilingualSummModel


class MBartModel(MultilingualSummModel):
    # static variables
    model_name = "mBART"
    is_extractive = False
    is_neural = True
    is_multilingual = True

    lang_tag_dict = {
        "ar": "ar_AR",
        "cs": "cs_CZ",
        "de": "de_DE",
        "en": "en_XX",
        "es": "es_XX",
        "et": "et_EE",
        "fi": "fi_FI",
        "fr": "fr_XX",
        "gu": "gu_IN",
        "hi": "hi_IN",
        "it": "it_IT",
        "ja": "ja_XX",
        "kk": "kk_KZ",
        "ko": "ko_KR",
        "lt": "lt_LT",
        "lv": "lv_LV",
        "my": "my_MM",
        "ne": "ne_NP",
        "nl": "nl_XX",
        "ro": "ro_RO",
        "ru": "ru_RU",
        "si": "si_LK",
        "tr": "tr_TR",
        "vi": "vi_VN",
        "zh": "zh_CN",
        "af": "af_ZA",
        "az": "az_AZ",
        "bn": "bn_IN",
        "fa": "fa_IR",
        "he": "he_IL",
        "hr": "hr_HR",
        "id": "id_ID",
        "ka": "ka_GE",
        "km": "km_KH",
        "mk": "mk_MK",
        "ml": "ml_IN",
        "mn": "mn_MN",
        "mr": "mr_IN",
        "pl": "pl_PL",
        "ps": "ps_AF",
        "pt": "pt_XX",
        "sv": "sv_SE",
        "ta": "ta_IN",
        "te": "te_IN",
        "th": "th_TH",
        "tl": "tl_XX",
        "uk": "uk_UA",
        "ur": "ur_PK",
        "xh": "xh_ZA",
        "sl": "sl_SI",
    }

    def __init__(self, device="cpu"):
        super(MBartModel, self).__init__(
            # TODO: trained domain not news (at least not exclusively)
            trained_domain="News",
            max_input_length=1024,
            max_output_length=None,
        )

        self.device = device

        model_name = "facebook/mbart-large-50"
        self.tokenizer = MBart50Tokenizer.from_pretrained(
            model_name, src_lang="en_XX", tgt_lang="en_XX"
        )
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(
            device
        )

    def summarize(self, corpus, queries=None):
        lang_code = self.assert_summ_input_type(corpus, queries)

        self.tokenizer.src_lang = lang_code
        self.tokenizer.tgt_lang = lang_code

        with self.tokenizer.as_target_tokenizer():
            batch = self.tokenizer(
                corpus, truncation=True, padding="longest", return_tensors="pt"
            ).to(self.device)

        encoded_summaries = self.model.generate(
            **batch,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[lang_code],
            length_penalty=1.0,
            num_beams=4,
            early_stopping=True,
        )

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
            " - Higher max input length than mT5 (1024)"
            "Weaknesses: \n - High memory usage"
            "Initialization arguments: \n "
            "- `device = 'cpu'` specifies the device the model is stored on and uses for computation. "
            "Use `device='cuda'` to run on an Nvidia GPU."
        )
        print(f"{basic_description} \n {'#'*20} \n {more_details}")
