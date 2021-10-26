from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from .base_multilingual_model import MultilingualSummModel


class MBartModel(MultilingualSummModel):
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
        self.tokenizer = MBart50Tokenizer.from_pretrained(
            model_name, src_lang="en_XX", tgt_lang="en_XX"
        )
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(
            device
        )

        lang_tag_dict = {
            "Arabic": "ar_AR",
            "Czech": "cs_CZ",
            "German": "de_DE",
            "English": "en_XX",
            "Spanish": "es_XX",
            "Estonian": "et_EE",
            "Finnish": "fi_FI",
            "French": "fr_XX",
            "Gujarati": "gu_IN",
            "Hindi": "hi_IN",
            "Italian": "it_IT",
            "Japanese": "ja_XX",
            "Kazakh": "kk_KZ",
            "Korean": "ko_KR",
            "Lithuanian": "lt_LT",
            "Latvian": "lv_LV",
            "Burmese": "my_MM",
            "Nepali": "ne_NP",
            "Dutch": "nl_XX",
            "Romanian": "ro_RO",
            "Russian": "ru_RU",
            "Sinhala": "si_LK",
            "Turkish": "tr_TR",
            "Vietnamese": "vi_VN",
            "Chinese": "zh_CN",
            "Afrikaans": "af_ZA",
            "Azerbaijani": "az_AZ",
            "Bengali": "bn_IN",
            "Persian": "fa_IR",
            "Hebrew": "he_IL",
            "Croatian": "hr_HR",
            "Indonesian": "id_ID",
            "Georgian": "ka_GE",
            "Khmer": "km_KH",
            "Macedonian": "mk_MK",
            "Malayalam": "ml_IN",
            "Mongolian": "mn_MN",
            "Marathi": "mr_IN",
            "Polish": "pl_PL",
            "Pashto": "ps_AF",
            "Portuguese": "pt_XX",
            "Swedish": "sv_SE",
            "Tamil": "ta_IN",
            "Telugu": "te_IN",
            "Thai": "th_TH",
            "Tagalog": "tl_XX",
            "Ukrainian": "uk_UA",
            "Urdu": "ur_PK",
            "Xhosa": "xh_ZA",
            "Slovenian": "sl_SI",
        }

    def summarize(self, corpus, queries=None):
        self.assert_summ_input_type(corpus, queries)

        self.assert_summ_input_language(corpus, queries)

        # self.tokenizer.src_lang = language token
        # self.tokenizer.tgt_lang = language token

        with self.tokenizer.as_target_tokenizer:
            batch = self.tokenizer(
                corpus, truncation=True, padding="longest", return_tensors="pt"
            ).to(self.device)
        encoded_summaries = self.model.generate(
            **batch
        )  # ,decoder_start_token_id=self.tokenizer.lang_code_to_id[""] )
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
