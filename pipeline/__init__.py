from model import SUPPORTED_SUMM_MODELS
from model.base_model import SummModel
from model.single_doc import LexRankModel

from dataset.st_dataset import SummDataset
from dataset.non_huggingface_datasets import ScisummnetDataset

from typing import List, Tuple


def get_lxr_train_set(dataset: SummDataset, size: int = 100) -> List[str]:

    """
    return some dummy summarization examples, in the format of a list of sources
    """
    subset = []
    for i in range(size):
        subset.append(next(iter(dataset.train_set)))

    src = list(map(lambda x: " ".join(x.source) if dataset.is_dialogue_based or dataset.is_multi_document else x.source[0] if isinstance(dataset, ScisummnetDataset) else x.source, subset))

    return src


def assemble_model_pipeline(dataset: SummDataset, model_list: List[SummModel] = SUPPORTED_SUMM_MODELS) -> List[Tuple[SummModel, str]]:

    """
    Return initialized list of all model pipelines that match the summarization task of given dataset.

    :param SummDataset `dataset`: Dataset to retrieve model pipelines for.
    :param List[SummModel] `model_list`: List of candidate model classes (uninitialized). Defaults to `model.SUPPORTED_SUMM_MODELS`.
    :returns List of tuples, where each tuple contains an initialized model and the name of that model as `(model, name)`.
    """

    dataset = dataset if isinstance(dataset, SummDataset) else dataset()

    single_doc_model_list = list(
        filter(
            lambda model_cls:
                not (model_cls.is_dialogue_based or model_cls.is_query_based or model_cls.is_multi_document),
            model_list))
    single_doc_model_instances = [
        model_cls(get_lxr_train_set(dataset)) if model_cls == LexRankModel else model_cls() for model_cls in single_doc_model_list
    ]
        
    multi_doc_model_list = list(filter(lambda model_cls: model_cls.is_multi_document, model_list))
    
    query_based_model_list = list(filter(lambda model_cls: model_cls.is_query_based, model_list))

    dialogue_based_model_list = list(filter(lambda model_cls: model_cls.is_dialogue_based, model_list))
    dialogue_based_model_instances = [model_cls() for model_cls in dialogue_based_model_list] if dataset.is_dialogue_based else []

    matching_models = []
    if dataset.is_query_based:
        if dataset.is_dialogue_based:
            for query_model_cls in query_based_model_list:
                for dialogue_model in dialogue_based_model_list:
                    full_query_dialogue_model = query_model_cls(model_backend=dialogue_model)
                    matching_models.append((full_query_dialogue_model, f"{query_model_cls.model_name} ({dialogue_model.model_name})"))
        else:
            for query_model_cls in query_based_model_list:
                for single_doc_model in single_doc_model_list:
                    full_query_model = query_model_cls(model_backend=single_doc_model, data=get_lxr_train_set(dataset)) if single_doc_model == LexRankModel else query_model_cls(model_backend=single_doc_model)
                    matching_models.append((full_query_model, f"{query_model_cls.model_name} ({single_doc_model.model_name})"))
        return matching_models

    if dataset.is_multi_document:
        for multi_doc_model_cls in multi_doc_model_list:
            for single_doc_model in single_doc_model_list:
                full_multi_doc_model = multi_doc_model_cls(model_backend=single_doc_model, data=get_lxr_train_set(dataset)) if single_doc_model == LexRankModel else multi_doc_model_cls(model_backend=single_doc_model)
                matching_models.append((full_multi_doc_model, f"{multi_doc_model_cls.model_name} ({single_doc_model.model_name})"))
        return matching_models
    
    if dataset.is_dialogue_based:
        return list(map(lambda db_model: (db_model, db_model.model_name), dialogue_based_model_instances))
    
    return list(map(lambda s_model: (s_model, s_model.model_name), single_doc_model_instances))
