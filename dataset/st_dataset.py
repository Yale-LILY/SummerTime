import sys
import time
from typing import Dict, List, Optional, Union, Generator

from datasets import Dataset, DatasetDict, DatasetInfo, concatenate_datasets, load_dataset

DEFAULT_NUMBER_OF_RETRIES_ALLOWED = 5
DEFAULT_WAIT_SECONDS_BEFORE_RETRY = 5

class SummInstance:
    """
    Basic instance for summarization tasks
    """
    
    def __init__(self, source: Union[List[str], str],
                 summary: str,
                 query: Optional[str]=None):
        """
        Create a summarization instance
        :rtype: object
        :param source: either `List[str]` or `str`, depending on the dataset itself, string joining may needed to fit
            into specific models. For example, for the same document, it could be simply `str` or `List[str]` for
            a list of sentences in the same document
        :param summary: a string summary that serves as ground truth
        :param query: Optional, applies when a string query is present
        """
        self.source = source
        self.summary = summary
        self.query = query

    def __str__(self):
        instance_str = {"source": self.source, "summary":self.summary}
        if self.query:
            instance_str["query"] = self.query

        return str(instance_str)


class SummDataset:
    """
    Dataset class for summarization, which takes into account of the following tasks:
        * Single document summarization
        * Multi-document/Dialogue summarization
        * Query-based summarization
    """
    
    def __init__(self,
                 dataset_name: str,
                 description: str,
                 citation: str = None,
                 homepage: str = None,
                 huggingface_page: str = None,
                 train_set: Optional[Generator[SummInstance, None, None]] = None,
                 dev_set: Optional[Generator[SummInstance, None, None]] = None,
                 test_set: Optional[Generator[SummInstance, None, None]] = None):
        """
        Following huggingface, the dataset contains train, dev and test set.
        :param train_set:
        :param dev_set:
        :param test_set:
        
        The following attributes should have been class attributes, however, python initialize all class variables
            at the time of importation, and we don't want the dataset to be loaded once imported
        :param dataset_name:
        :param description:
        :param is_query_based:
        :param is_dialogue_based:
        :param is_multi_document:
        """
        
        assert train_set or dev_set or test_set, "At least one of train/dev/test needs to be not empty!"
        
        self.dataset_name = dataset_name
        self.description = description
        self.citation = citation
        self.homepage = homepage
        self.huggingface_page = huggingface_page
        
        self._train_set = train_set
        self._dev_set = dev_set
        self._test_set = test_set
        
    @property
    def train_set(self) -> Union[Generator[SummInstance, None, None], List]:
        if self._train_set is not None:
            return self._train_set
        else:
            print(f"{self.dataset_name} does not contain a train set, empty list returned")
            return list()

    @property
    def dev_set(self) -> Union[Generator[SummInstance, None, None], List]:
        if self._dev_set is not None:
            return self._dev_set
        else:
            print(f"{self.dataset_name} does not contain a dev set, empty list returned")
            return list()
    
    @property
    def test_set(self) -> Union[Generator[SummInstance, None, None], List]:
        if self._test_set is not None:
            return self._test_set
        else:
            print(f"{self.dataset_name} does not contain a test set, empty list returned")
            return list()

    
    def load_dataset_safe(self, *args, **kwargs) -> Dataset:
        """
        This method creates a wrapper around the huggingface 'load_dataset()' function for a more robust download function,
            the original 'load_dataset()' function occassionally fails when it cannot reach a server especially after multiple requests.
            This method tackles this problem by attempting the download multiple times with a wait time before each retry
        
        The wrapper method passes all arguments and keyword arguments to the 'load_dataset' function with no alteration.
        """

        tries = DEFAULT_NUMBER_OF_RETRIES_ALLOWED
        wait_time = DEFAULT_WAIT_SECONDS_BEFORE_RETRY

        for i in range(tries):
            try:
                dataset = load_dataset(*args, **kwargs)
            except :
                if i < tries - 1: # i is zero indexed
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError("Wait for a minute and attempt downloading the dataset again. \
                                        The server hosting the dataset occassionally times out.")
            break

        return dataset




def generate_train_dev_test_splits(dataset: Dataset, seed: int) -> DatasetDict:
    """
    Creating the train, dev and test splits from a dataset
        the generated sets are 'train: ~.80', 'dev: ~.10', and 'test: ~10' in size
        the splits are randomized for each object unless a seed is provided for the random generator
    
    :param dataset: Arrow Dataset with containing, usually the train set
    :param seed: seed for the random generator to shuffle the dataset
    :rtype: Arrow DatasetDict containing the three splits
    """
    
    # First split train into: train and test splits
    # Further split the remaining train set into: train and dev sets
    # The dev set split has a higher ratio than the test set (0.11 vs 0.1) due to the smaller train set used
    dataset_traintest_split = dataset.train_test_split(test_size=0.1, seed=seed)
    dataset_traindev_split = dataset_traintest_split['train'].train_test_split(test_size=0.11, seed=seed) 

    temp_dict = {}
    temp_dict['train'] = dataset_traindev_split['train']
    temp_dict['dev'] = dataset_traindev_split['test']
    temp_dict['test'] = dataset_traintest_split['test']

    return DatasetDict(temp_dict)




def concatenate_dataset_dicts(dataset_dicts: List[DatasetDict]) -> DatasetDict:    
    """
    Concatenate two dataset dicts with similar splits and columns tinto one

    :param dataset_dicts: A list of DatasetDicts
    :rtype: DatasetDict containing the combined data
    """

    # Ensure all dataset dicts have the same splits
    setsofsplits = set(tuple(dataset_dict.keys()) for dataset_dict in dataset_dicts)
    if (len(setsofsplits) != 1):
        raise ValueError("Splits must match for all datasets")

    # Concatenate all datasets into one according to the splits
    temp_dict = {}
    for split in setsofsplits.pop():
        split_set = [dataset_dict[split] for dataset_dict in dataset_dicts]
        temp_dict[split] = concatenate_datasets(split_set)
    
    return DatasetDict(temp_dict)



# Get the information set from the dataset
# The information set contains: dataset name, description, version, citation and licence
# Param: DatasetDict
# rtype: DatasetInfo 
def get_dataset_info(data_dict: DatasetDict) -> DatasetInfo:
    return data_dict["train"].info
