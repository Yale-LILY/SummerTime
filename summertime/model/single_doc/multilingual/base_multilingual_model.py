from summertime.model.single_doc.base_single_doc_model import SingleDocSummModel

import urllib.request
import fasttext
from typing import Dict, List, Tuple


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

        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        # currently using compressed fasttext model from FB
        urllib.request.urlretrieve(url, "lid.176.ftz")

        classifier = fasttext.load_model("./lid.176.ftz")

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

        # check if language code is in the supported language dictionary
        if label in cls.lang_tag_dict:
            print(f"Language '{label}' detected.")
            return cls.lang_tag_dict[label]
        else:
            raise ValueError(
                f"Unsupported language '{label}'' detected!\n\
                    Try checking if another of our multilingual models \
                    supports this language."
            )
