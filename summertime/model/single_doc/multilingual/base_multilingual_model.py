from summertime.model.single_doc.base_single_doc_model import SingleDocSummModel

from tqdm import tqdm
import urllib.request
import fasttext #fasttext 0.9.2


class MultilingualSummModel(SingleDocSummModel):
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

        lang_tag_dict = None

    @classmethod
    def assert_summ_input_language(cls, corpus, query):
        # TODO: add fasttext language detection here

        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" 
        # currently using compressed fasttext model

        tqdm(urllib.request.urlretrieve(url, "lid.176.ftz"), desc="Language Detection Model Download")

        model = fasttext.load_model("./lid.176.ftz") #TODO: change download location, 
        # do not redownload every time if not necessary

        




