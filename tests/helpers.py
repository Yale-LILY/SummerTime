from dataset.st_dataset import SummDataset, SummInstance
from dataset.non_huggingface_datasets import ScisummnetDataset

import random
from typing import List, Tuple


def print_with_color(s: str, color: str):
    """
    Print formatted string.

    :param str `s`: String to print.
    :param str `color`: ANSI color code.

    :see https://gist.github.com/RabaDabaDoba/145049536f815903c79944599c6f952a
    """

    print(f"\033[{color}m{s}\033[0m")


def retrieve_random_test_instances(
    dataset_instances: List[SummInstance], num_instances=3
) -> List[SummInstance]:
    """
    Retrieve random test instances from a dataset training set.

    :param List[SummInstance] `dataset_instances`: Instances from a dataset `train_set` to pull random examples from.
    :param int `num_instances`: Number of random instances to pull. Defaults to `3`.
    :return List of SummInstance to summarize.
    """

    test_instances = []
    for i in range(num_instances):
        test_instances.append(
            dataset_instances[random.randint(0, len(dataset_instances) - 1)]
        )
    return test_instances


def get_summarization_set(dataset: SummDataset, size=1) -> Tuple[List, List]:
    """
    Return instances from given summarization dataset, in the format of (sources, targets).
    """
    subset = []
    for i in range(size):
        subset.append(next(dataset.train_set))

    src, tgt = zip(*(list(map(lambda x: (x.source, x.summary), subset))))

    return list(src), list(tgt)


def get_query_based_summarization_set(
    dataset: SummDataset, size=1
) -> Tuple[List, List, List]:
    """
    Return instances from given query-based summarization dataset, in the format of (sources, targets, queries).
    """
    subset = []
    for i in range(size):
        subset.append(next(dataset.train_set))

    src, tgt, queries = zip(
        *(list(map(lambda x: (x.source, x.summary, x.query), subset)))
    )

    return list(src), list(tgt), list(queries)
