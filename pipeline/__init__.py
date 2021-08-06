from model import SUPPORTED_SUMM_MODELS
from model.base_model import SummModel
from model.single_doc import LexRankModel

from dataset.st_dataset import SummDataset

from typing import List


def get_train_set(dataset: SummDataset, size: int = 100) -> List[str]:

    """
    return some dummy summarization examples, in the format of a list of sources
    """
    subset = []
    for i in range(size):
        subset.append(next(dataset.train_set))

    src = list(map(lambda x: x.source, subset))

    return src


def assemble_model_pipeline(dataset: SummDataset, model_list: List[SummModel] = SUPPORTED_SUMM_MODELS) -> List[SummModel]:

    """
    Returns filtered initialized subset of `model_list` where the summarization
    task matches given `dataset`. Initial `model_list` defaults to `SUPPORTED_SUMM_MODELS`.
    """

    dataset = dataset if isinstance(dataset, SummDataset) else dataset()

    single_doc_model_list = list(
        filter(
            lambda model_cls:
                not (model_cls.is_dialogue_based or model_cls.is_query_based or model_cls.is_multi_document),
            model_list))
    single_doc_model_instances = [
        model_cls(get_train_set(dataset)) if model_cls == LexRankModel else model_cls() for model_cls in single_doc_model_list
    ] if not (dataset.is_dialogue_based or dataset.is_query_based or dataset.is_multi_document) else []
        
    multi_doc_model_list = list(filter(lambda model_cls: model_cls.is_multi_document, model_list))
    
    query_based_model_list = list(filter(lambda model_cls: model_cls.is_query_based, model_list))

    dialogue_based_model_list = list(filter(lambda model_cls: model_cls.is_dialogue_based, model_list))
    dialogue_based_model_instances = [model_cls() for model_cls in dialogue_based_model_list] if dataset.is_dialogue_based else []

    matching_models = []
    if dataset.is_query_based:
        if dataset.is_dialogue_based:
            for query_model_cls in query_based_model_list:
                for dialogue_model in dialogue_based_model_instances:
                    full_query_dialogue_model = query_model_cls(model_backend=dialogue_model)
                    matching_models.append(full_query_dialogue_model)
        else:
            for query_model_cls in query_based_model_list:
                for single_doc_model in single_doc_model_instances:
                    full_query_model = query_model_cls(model_backend=single_doc_model)
                    matching_models.append(full_query_dialogue_model)
        return matching_models
    
    if dataset.is_multi_document:
        for multi_doc_model_cls in multi_doc_model_list:
            for single_doc_model in single_doc_model_instances:
                full_multi_doc_model = multi_doc_model_cls(model_backend=single_doc_model)
                matching_models.append(full_multi_doc_model)
        return matching_models
    
    if dataset.is_dialogue_based:
        return dialogue_based_model_instances
    
    return single_doc_model_instances
