from model import SUPPORTED_SUMM_MODELS
from model.base_model import SummModel
from model.single_doc import LexRankModel

from dataset.st_dataset import SummDataset
from dataset import ScisummnetDataset

from typing import Dict, List, Tuple, Union, Set


def retrieve_task_nodes(model_or_dataset: Union[SummModel, SummDataset]) -> List[str]:
    """Generates list of summarization task nodes as strings
    given model or dataset.

    Args:
        model_or_dataset (Union[SummModel, SummDataset]): SummerTime model or dataset

    Returns:
        List[str]: Model/dataset task types in string form
    """
    task_nodes = ["is_single_document"]
    if model_or_dataset.is_dialogue_based:
        task_nodes.append("is_dialogue_based")
    if model_or_dataset.is_multi_document:
        task_nodes.append("is_multi_document")
    if model_or_dataset.is_query_based:
        task_nodes.append("is_query_based")
    return task_nodes


def top_sort_dfs(
    list_nodes: List[str],
    graph: Dict[str, List],
    cur_node: str,
    sorted_list: List[str],
    visited: Set,
):
    """DFS helper for topological sort

    Args:
        list_nodes (List[str]): List of nodes to sort
        graph (Dict[List]): Directed graph of nodes
        cur_node (str): Current node in dfs
        sorted_list (List[str]): Sorted list to append new nodes to
        visited (Dict[bool]): Tracks whether nodes have been visited previously
    """
    if cur_node in visited:
        return
    visited.add(cur_node)
    sorted_list.append(cur_node)
    for neighbor in graph[cur_node]:
        if neighbor in list_nodes and not neighbor in visited:
            top_sort_dfs(list_nodes, graph, neighbor, sorted_list, visited)


def top_sort_options(list_nodes: List[str], graph: Dict[str, List]) -> List[str]:
    """Sorts list according to topological order in graph.

    Args:
        list (List[str]): list of nodes to sort
        graph (Dict[List]): graph containing topological order of nodes

    Returns:
        List[str]: topologically sorted list
    """
    in_degrees = {}
    for node in list_nodes:
        in_degrees[node] = 0
    for node in list_nodes:
        for neighbor in graph[node]:
            if neighbor in list_nodes:
                in_degrees[neighbor] += 1

    sorted_list = []
    visited = set()
    for node in list_nodes:
        if in_degrees[node] == 0:
            top_sort_dfs(list_nodes, graph, node, sorted_list, visited)
    if len(sorted_list) == 0:
        print(list_nodes)
        print("Graph is cyclical!!!")
        return []

    print(sorted_list)
    return sorted_list


def create_model_composition_graph() -> Dict[str, List]:
    """Returns directed graph where each node
    is a summarization task and each edge represents
    an appropriate order for models to be applied
    to a multi-layered summarization task.

    Returns:
        Dict[List]: Adjacency list representation of
        graph.
    """
    graph = {}
    graph["is_single_document"] = []
    graph["is_multi_document"] = ["is_single_document"]
    graph["is_dialogue_based"] = ["is_multi_document", "is_single_document"]
    graph["is_query_based"] = ["is_dialogue_based", "is_multi_document"]
    print(graph)
    return graph


def get_lxr_train_set(dataset: SummDataset, size: int = 100) -> List[str]:

    """
    return some dummy summarization examples, in the format of a list of sources
    """
    subset = []
    for i in range(size):
        subset.append(next(iter(dataset.train_set)))

    src = list(
        map(
            lambda x: " ".join(x.source)
            if dataset.is_dialogue_based or dataset.is_multi_document
            else x.source[0]
            if isinstance(dataset, ScisummnetDataset)
            else x.source,
            subset,
        )
    )

    return src


def assemble_model_pipeline_2(
    dataset: SummDataset, model_list: List[SummModel] = SUPPORTED_SUMM_MODELS
) -> List[Tuple[SummModel, str]]:
    """
    Return initialized list of all model pipelines that match the summarization task of given dataset.

    :param SummDataset `dataset`: Dataset to retrieve model pipelines for.
    :param List[SummModel] `model_list`: List of candidate model classes (uninitialized). Defaults to `model.SUPPORTED_SUMM_MODELS`.
    :returns List of tuples, where each tuple contains an initialized model and the name of that model as `(model, name)`.
    """

    dataset = dataset if isinstance(dataset, SummDataset) else dataset()

    single_doc_model_list = list(
        filter(
            lambda model_cls: not (
                model_cls.is_dialogue_based
                or model_cls.is_query_based
                or model_cls.is_multi_document
            ),
            model_list,
        )
    )
    single_doc_model_instances = [
        model_cls(get_lxr_train_set(dataset))
        if model_cls == LexRankModel
        else model_cls()
        for model_cls in single_doc_model_list
    ]

    multi_doc_model_list = list(
        filter(lambda model_cls: model_cls.is_multi_document, model_list)
    )

    query_based_model_list = list(
        filter(lambda model_cls: model_cls.is_query_based, model_list)
    )

    dialogue_based_model_list = list(
        filter(lambda model_cls: model_cls.is_dialogue_based, model_list)
    )
    dialogue_based_model_instances = (
        [model_cls() for model_cls in dialogue_based_model_list]
        if dataset.is_dialogue_based
        else []
    )

    task_node_list = retrieve_task_nodes(dataset)
    graph = create_model_composition_graph()
    sorted_task_node_list = top_sort_options(task_node_list, graph)

    print(sorted_task_node_list)
    if len(sorted_task_node_list) == 0:
        return [(model, model.model_name) for model in single_doc_model_instances]

    task_node_to_model_list = {
        "is_single_document": single_doc_model_list,
        "is_dialogue_based": dialogue_based_model_list,
        "is_multi_document": multi_doc_model_list,
        "is_query_based": query_based_model_list,
    }

    matching_models = []
    sorted_task_node_list.reverse()
    for task_node in sorted_task_node_list:
        if len(matching_models) == 0:
            for model_cls in task_node_to_model_list[task_node]:
                # TODO: How to tell if last task needs model backend?
                matching_models.append((model_cls, model_cls.model_name))
        else:
            new_matching_models = []
            for model_cls in task_node_to_model_list[task_node]:
                for model_backend, model_backend_name in matching_models:
                    new_matching_models.append(
                        (
                            model_cls(
                                model_backend=model_backend,
                                data=get_lxr_train_set(dataset),
                            ),
                            f"{model_cls.model_name} ({model_backend_name})",
                        )
                        if model_backend == LexRankModel
                        else model_cls(model_backend=model_backend)
                    )
            matching_models = new_matching_models
    return matching_models


def assemble_model_pipeline(
    dataset: SummDataset, model_list: List[SummModel] = SUPPORTED_SUMM_MODELS
) -> List[Tuple[SummModel, str]]:

    """
    Return initialized list of all model pipelines that match the summarization task of given dataset.

    :param SummDataset `dataset`: Dataset to retrieve model pipelines for.
    :param List[SummModel] `model_list`: List of candidate model classes (uninitialized). Defaults to `model.SUPPORTED_SUMM_MODELS`.
    :returns List of tuples, where each tuple contains an initialized model and the name of that model as `(model, name)`.
    """

    dataset = dataset if isinstance(dataset, SummDataset) else dataset()

    single_doc_model_list = list(
        filter(
            lambda model_cls: not (
                model_cls.is_dialogue_based
                or model_cls.is_query_based
                or model_cls.is_multi_document
            ),
            model_list,
        )
    )
    single_doc_model_instances = [
        model_cls(get_lxr_train_set(dataset))
        if model_cls == LexRankModel
        else model_cls()
        for model_cls in single_doc_model_list
    ]

    multi_doc_model_list = list(
        filter(lambda model_cls: model_cls.is_multi_document, model_list)
    )

    query_based_model_list = list(
        filter(lambda model_cls: model_cls.is_query_based, model_list)
    )

    dialogue_based_model_list = list(
        filter(lambda model_cls: model_cls.is_dialogue_based, model_list)
    )
    dialogue_based_model_instances = (
        [model_cls() for model_cls in dialogue_based_model_list]
        if dataset.is_dialogue_based
        else []
    )

    matching_models = []
    if dataset.is_query_based:
        if dataset.is_dialogue_based:
            for query_model_cls in query_based_model_list:
                for dialogue_model in dialogue_based_model_list:
                    full_query_dialogue_model = query_model_cls(
                        model_backend=dialogue_model
                    )
                    matching_models.append(
                        (
                            full_query_dialogue_model,
                            f"{query_model_cls.model_name} ({dialogue_model.model_name})",
                        )
                    )
        else:
            for query_model_cls in query_based_model_list:
                for single_doc_model in single_doc_model_list:
                    full_query_model = (
                        query_model_cls(
                            model_backend=single_doc_model,
                            data=get_lxr_train_set(dataset),
                        )
                        if single_doc_model == LexRankModel
                        else query_model_cls(model_backend=single_doc_model)
                    )
                    matching_models.append(
                        (
                            full_query_model,
                            f"{query_model_cls.model_name} ({single_doc_model.model_name})",
                        )
                    )
        return matching_models

    if dataset.is_multi_document:
        for multi_doc_model_cls in multi_doc_model_list:
            for single_doc_model in single_doc_model_list:
                full_multi_doc_model = (
                    multi_doc_model_cls(
                        model_backend=single_doc_model, data=get_lxr_train_set(dataset)
                    )
                    if single_doc_model == LexRankModel
                    else multi_doc_model_cls(model_backend=single_doc_model)
                )
                matching_models.append(
                    (
                        full_multi_doc_model,
                        f"{multi_doc_model_cls.model_name} ({single_doc_model.model_name})",
                    )
                )
        return matching_models

    if dataset.is_dialogue_based:
        return list(
            map(
                lambda db_model: (db_model, db_model.model_name),
                dialogue_based_model_instances,
            )
        )

    return list(
        map(lambda s_model: (s_model, s_model.model_name), single_doc_model_instances)
    )
