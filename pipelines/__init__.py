from model import SUPPORTED_SUMM_MODELS
from model.base_model import SummModel

from dataset.st_dataset import SummDataset

from typing import List


def assemble_model_pipeline(dataset: SummDataset) -> List[SummModel]:
    """
    Returns filtered subset of `SUPPORTED_SUMM_MODELS` where the summarization
    task matches given `dataset`.
    """
    return list(
        filter(
            lambda model: model.is_dialogue_based == dataset.is_dialogue_based
            and model.is_multi_document == dataset.is_multi_document
            and model.is_query_based == dataset.is_query_based,
            SUPPORTED_SUMM_MODELS,
        )
    )
