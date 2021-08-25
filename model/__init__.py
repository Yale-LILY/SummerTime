from .single_doc import BartModel, LexRankModel, LongformerModel, PegasusModel, TextRankModel
from .multi_doc import MultiDocJointModel, MultiDocSeparateModel
from .dialogue import HMNetModel
from .query_based import TFIDFSummModel, BM25SummModel
from .defaults import summarizer

SUPPORTED_SUMM_MODELS = [BartModel, LexRankModel, LongformerModel,
                         PegasusModel, TextRankModel,
                         MultiDocJointModel, MultiDocSeparateModel, HMNetModel,
                         TFIDFSummModel, BM25SummModel]


def list_all_models():
    all_model_tuples = []
    for model_class in SUPPORTED_SUMM_MODELS:
        model_description = model_class.generate_basic_description()
        
        all_model_tuples.append((model_class, model_description))
        
    return all_model_tuples
