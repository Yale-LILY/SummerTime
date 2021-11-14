from summertime.model.single_doc.base_single_doc_model import SingleDocSummModel
from summertime.util.download_utils import (
    get_cached_file_path,
)
import fasttext
from typing import List, Union, Dict, List, Tuple


def fasttext_predict(corpus: Union[List[str], List[List[str]]]):
    """
    Utility function to predict the language of input text
    using fasttext classifier.
    """
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

    filepath = get_cached_file_path("fasttext", "lid.176.ftz", url)

    fasttext.FastText.eprint = lambda x: None
    classifier = fasttext.load_model(str(filepath))

    # fasttext returns a tuple of 2 lists:
    # the first list contains a list of predicted language labels
    # of the form {__label__<lang_code>}
    # and the second list contains the corresponding probabilities
    prediction: Tuple[List[List[str]], List] = None
    if all([isinstance(ins, list) for ins in corpus]):
        prediction = classifier.predict(corpus[0])

    elif isinstance(corpus, list):
        prediction = classifier.predict(corpus)

    # access the first (most likely) predicted language label
    label = prediction[0][0][0]

    # remove prefix from label string to get language code
    label = label.replace("__label__", "")

    return label

class MultilingualSummModel(SingleDocSummModel):

    # a dictionary of languages supported by the model.
    # The key is the language code (ISO-639-1 format currently),
    # and the value is the language code/token used by the model.
    lang_tag_dict: Dict[str, str] = None

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
    def assert_summ_input_type(cls, corpus, query):

        super().assert_summ_input_type(corpus, query)

        label = fasttext_predict(corpus)

        # check if language code is in the supported language dictionary
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
