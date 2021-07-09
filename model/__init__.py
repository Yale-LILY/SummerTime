from .single_doc import BartModel, LexRankModel, LongformerModel, PegasusModel, TextRankModel
from .multi_doc import MultiDocJointModel, MultiDocSeparateModel
from .defaults import summarizer

SUPPORTED_SUMM_MODELS = [BartModel, LexRankModel, LongformerModel, MultiDocJointModel, MultiDocSeparateModel, PegasusModel, TextRankModel]


def list_all_models():
    all_model_tuples = []
    for model_class in SUPPORTED_SUMM_MODELS:
        model_description = model_class.generate_basic_description()
        
        all_model_tuples.append((model_class, model_description))
        
    return all_model_tuples
