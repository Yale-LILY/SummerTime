from .base_multilingual_model import MultilingualSummModel, fasttext_predict
from summertime.model.base_model import SummModel
from summertime.model.single_doc import BartModel

from easynmt import EasyNMT


class TranslationPipelineModel(MultilingualSummModel):
    """
    A class for multilingual summarization performed by first
    translating into English then performing summarization in English.
    """

    model_name = "Translation Pipeline"
    is_multilingual = True
    # TODO: change to Pegasus as default?
    # language codes from https://github.com/UKPLab/EasyNMT#Opus-MT documentation
    # language codes not supported by https://fasttext.cc/docs/en/language-identification.html
    # are commented out.
    supported_langs = [
        # "aav",
        "aed",
        "af",
        # "alv",
        "am",
        "ar",
        # "art",
        # "ase",
        "az",
        "bat",
        "bcl",
        "be",
        # "bem",
        # "ber",
        "bg",
        # "bi",
        "bn",
        # "bnt",
        # "bzs",
        "ca",
        # "cau",
        # "ccs",
        "ceb",
        # "cel",
        # "chk",
        # "cpf",
        # "crs",
        "cs",
        # "csg",
        # "csn",
        # "cus",
        "cy",
        "da",
        "de",
        # "dra",
        # "ee",
        # "efi",
        "el",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        # "euq",
        "fi",
        # "fj",
        "fr",
        # "fse",
        "ga",
        # "gaa",
        # "gil",
        "gl",
        # "grk",
        # "guw",
        "gv",
        # "ha",
        "he",
        "hi",
        # "hil",
        # "ho",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        # "ig",
        "ilo",
        "is",
        # "iso",
        "it",
        "ja",
        # "jap",
        "ka",
        # "kab",
        # "kg",
        # "kj",
        # "kl",
        "ko",
        # "kqn",
        # "kwn",
        # "kwy",
        # "lg",
        # "ln",
        # "loz",
        "lt",
        # "lu",
        # "lua",
        # "lue",
        # "lun",
        # "luo",
        # "lus",
        "lv",
        # "map",
        # "mfe",
        # "mfs",
        "mg",
        # "mh",
        "mk",
        # "mkh",
        "ml",
        # "mos",
        "mr",
        "ms",
        "mt",
        # "mul",
        # "ng",
        # "nic",
        # "niu",
        "nl",
        "no",
        # "nso",
        # "ny",
        # "nyk",
        # "om",
        "pa",
        # "pag",
        # "pap",
        # "phi",
        # "pis",
        "pl",
        # "pon",
        # "poz",
        # "pqe",
        # "pqw",
        # "prl",
        "pt",
        # "rn",
        # "rnd",
        "ro",
        # "roa",
        "ru",
        "run",
        # "rw",
        # "sal",
        # "sg",
        "sh",
        # "sit",
        "sk",
        "sl",
        # "sm",
        # "sn",
        "sq",
        # "srn",
        # "ss",
        # "ssp",
        # "st",
        "sv",
        "sw",
        # "swc",
        # "taw",
        # "tdt",
        "th",
        # "ti",
        # "tiv",
        "tl",
        # "tll",
        # "tn",
        # "to",
        # "toi",
        # "tpi",
        "tr",
        # "trk",
        # "ts",
        # "tum",
        # "tut",
        # "tvl",
        # "tw",
        # "ty",
        # "tzo",
        "uk",
        # "umb",
        "ur",
        # "ve",
        "vi",
        # "vsl",
        "wa",
        # "wal",
        "war",
        # "wls",
        # "xh",
        # "yap",
        "yo",
        # "yua",
        # "zai",
        "zh",
        # "zne",
    ]

    lang_tag_dict = {lang: lang for lang in supported_langs}

    def __init__(self, model_backend: SummModel = BartModel, **kwargs):

        model: SummModel = model_backend(**kwargs)
        self.model = model

        super(TranslationPipelineModel, self).__init__(
            trained_domain=self.model.trained_domain,
            max_input_length=self.model.max_input_length,
            max_output_length=self.model.max_output_length,
        )

        # translation module
        self.translator = EasyNMT("opus-mt")

    def summarize(self, corpus, queries=None):
        self.assert_summ_input_type(corpus, queries)

        src_lang = fasttext_predict(corpus)
        # translate to English
        corpus = self.translator.translate(
            corpus, source_lang=src_lang, target_lang="en", beam_size=4
        )
        # TODO: translate each doc separately if provided multiple docs in corpus?
        if queries:
            queries = self.translator.translate(queries, target_lang="en", beam_size=4)

        # summarize in English
        english_summaries = self.model.summarize(corpus, queries)

        summaries = self.translator.translate(
            english_summaries, source_lang="en", target_lang=src_lang, beam_size=4
        )

        return summaries

    @classmethod
    def show_capability(cls) -> None:
        basic_description = cls.generate_basic_description()
        more_details = (
            "A simple pipeline model for multilingual translation. "
            "Uses machine translation to translate input into English, "
            "then performs summarization in English before translating results "
            "back to the original language.\n"
            "Strengths: \n - Massively multilingual: supports ~150 languages\n"
            "Weaknesses: \n - Information loss from translation to and from English"
            "Initialization arguments: \n "
            " - model_backend: the monolingual model to use for summarization. Defaults to BART"
            # TODO: if change to Pegasus, change this to reflect that!!
            "- `device = 'cpu'` specifies the device the model is stored on and uses for computation. "
            "Use `device='cuda'` to run on an Nvidia GPU."
        )
        print(f"{basic_description} \n {'#'*20} \n {more_details}")
