from summertime.model.single_doc.base_single_doc_model import SingleDocSummModel

import urllib.request
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
        # TODO: add fasttext language detection here

        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        # currently using compressed fasttext model
        urllib.request.urlretrieve(url, "lid.176.ftz")
        # tqdm(
        #     urllib.request.urlretrieve(url, "lid.176.ftz"),
        #     desc="Downloading language detector",
        # )

        classifier = fasttext.load_model(
            "./lid.176.ftz"
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

    @classmethod
    def get_supported_languages(cls): #TODO: implement a display of supported languages for all models?
        return cls.lang_tag_dict.keys()