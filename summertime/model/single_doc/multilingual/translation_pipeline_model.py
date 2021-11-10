from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from .base_multilingual_model import MultilingualSummModel, fasttext_predict
from summertime.model.base_model import SummModel
from summertime.model.single_doc import BartModel

from easynmt import EasyNMT

class TranslationPipelineModel(MultilingualSummModel):
    """
    A class for multilingual summarization performed by first 
    translating into English then performing summarization in English.
    """

    model_name = 'Translation Pipeline'
    is_multilingual = True
    # TODO: change to Pegasus as default?
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
        corpus = self.translator.translate(corpus, source_lang=src_lang, target_lang="en", beam_size=4)
        # TODO: translate each doc separately if provided multiple docs in corpus?
        if queries:
            queries = self.translator.translate(queries, target_lang="en", beam_size=4)

        # summarize in English
        english_summaries = self.model.summarize(corpus, queries)

        summaries = self.translator.translate(english_summaries, source_lang="en", target_lang=src_lang, beam_size=4)

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


