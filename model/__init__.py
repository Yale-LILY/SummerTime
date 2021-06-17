from .bart_model import BartModel
from .pegasus_model import PegasusModel
from .lexrank_model import LexRankModel
from .longformer_model import LongformerModel
from .multi_doc_joint_model import MultiDocJointModel
from .multi_doc_separate_model import MultiDocSeparateModel
from .textrank_model import TextRankModel
from .defaults import summarizer

SUPPORTED_SUMM_MODELS = [BartModel, LexRankModel, LongformerModel, MultiDocSeparateModel, PegasusModel, TextRankModel]


def list_all_models():
    all_model_tuples = []
    for model_class in SUPPORTED_SUMM_MODELS:
        model_description = model_class.generate_basic_description()
        
        all_model_tuples.append((model_class, model_description))
        
    return all_model_tuples
