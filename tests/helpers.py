from dataset.st_dataset import SummDataset, SummInstance

import random
from typing import List


def print_with_color(s: str, color: str):
    """
    Print formatted string.

    :param str `s`: String to print.
    :param str `color`: ANSI color code.

    :see https://gist.github.com/RabaDabaDoba/145049536f815903c79944599c6f952a
    """

    print(f"\033[{color}m{s}\033[0m")

def retrieve_random_test_instances(self, dataset: SummDataset, num_instances = 3) -> List[SummInstance]:
    """
    Retrieve random test instances from a dataset training set.

    :param List[SummInstance] `dataset_instances`: Instances from a dataset `train_set` to pull random examples from.
    :param int `num_instances`: Number of random instances to pull. Defaults to `3`.
    :return List of SummInstance to summarize.
    """

    dataset_instances = list(dataset.train_set)
    print(f"\n{dataset.dataset_name} has a training set of {len(dataset_instances)} examples")
    test_instances = []
    for i in range(num_instances):
        test_instances.append(dataset_instances[random.randint(0, len(dataset_instances) - 1)])
    return test_instances