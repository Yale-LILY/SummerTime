from summertime.model.single_doc.base_single_doc_model import SingleDocSummModel
from summertime.util.download_utils import (
    get_cached_file_path,
)
import fasttext



class MultilingualSummModel(SingleDocSummModel):

    lang_tag_dict = None

    def __init__(
        self,
        trained_domain: str = None,
        max_input_length: int = None,
        max_output_length: int = None,
    ):
        super(MultilingualSummModel, self).__init__(
            trained_domain=trained_domain,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )

    @classmethod
    def assert_summ_input_language(cls, corpus, query):

        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

        filepath = get_cached_file_path("fasttext", "lid.176.ftz", url)

        classifier = fasttext.load_model(
            str(filepath)
        )  # TODO: change download location,
        # do not redownload every time if not necessary

        if all([isinstance(ins, list) for ins in corpus]):
            prediction = classifier.predict(corpus[0])

        elif isinstance(corpus, list):
            prediction = classifier.predict(corpus)

        label = prediction[0][0][0]

        label = label.replace("__label__", "")

        if label in cls.lang_tag_dict:
            print(f"Supported language '{label}' detected.")
            return cls.lang_tag_dict[label]
        else:
            raise ValueError(
                f"Unsupported language '{label}' detected! \
Try checking if another of our multilingual models \
supports this language."
            )

    # @classmethod
    # def show_supported_languages(
    #     cls,
    # ):
    #     langs = [iso639.to_name(lang) for lang in cls.lang_tag_dict.keys()]
    #     return " ".join(langs)
