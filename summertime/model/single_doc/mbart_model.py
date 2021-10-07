from transformers import MBartForConditionalGeneration, MBartTokenizer
from .base_single_doc_model import SingleDocSummModel

class MBartModel(SingleDocSummModel):
    # static variables
    model_name = "mBART"
    is_extractive = False
    is_neural = True

    def __init__()