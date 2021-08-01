
from typing import Dict, List, Optional, Union, Generator

from datasets import DatasetDict, concatenate_datasets


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
                 is_query_based: bool = False,
                 is_dialogue_based: bool = False,
                 is_multi_document: bool = False,
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
        self.is_query_based = is_query_based
        self.is_dialogue_based = is_dialogue_based
        self.is_multi_document = is_multi_document
        
        self._train_set = train_set
        self._dev_set = dev_set
        self._test_set = test_set
        
    @property
    def train_set(self) -> Generator[SummInstance, None, None]:
        if self._train_set is not None:
            return self._train_set
        else:
            print(f"{self.dataset_name} does not contain a train set, empty list returned")
            return list()

    @property
    def dev_set(self) -> Generator[SummInstance, None, None]:
        if self._dev_set is not None:
            return self._dev_set
        else:
            print(f"{self.dataset_name} does not contain a dev set, empty list returned")
            return list()
    
    @property
    def test_set(self) -> Generator[SummInstance, None, None]:
        if self._test_set is not None:
            return self._test_set
        else:
            print(f"{self.dataset_name} does not contain a test set, empty list returned")
            return list()


# Creating the train, dev and test splits from a dataset
# the generated sets are 'train: ~.80', 'dev: ~.10', and 'test: ~10' in size
# the splits are randomized for each object unless a seed is provided for the random generator
# Param: Arrow Dataset, seed for the random generator to shuffle the dataset
# rtype: Arrow DatasetDict containing the three splits
def generate_train_dev_test_splits(dataset: Dataset, seed: int) -> DatasetDict['train', 'dev', 'test']:
    
    # Creating the train, dev and test splits from a dataset
    # First split train into: train and test splits
    # Further split the remaining train set into: train and dev sets
    dataset_traintest_split = dataset.train_test_split(test_size=0.1, seed=seed)
    dataset_traindev_split = dataset_traintest_split['train'].train_test_split(test_size=0.11, seed=seed) 

    temp_dict = {}
    temp_dict['train'] = dataset_traindev_split['train']
    temp_dict['dev'] = dataset_traindev_split['test']
    temp_dict['test'] = dataset_traintest_split['test']

    return DatasetDict(temp_dict)

    

# Concatenate two dataset dicts with similar splits and columns tinto one
# Param: A list of DatasetDicts
# rtype: DatasetDict containing the combined data
def concatenate_dataset_dicts(dataset_dicts: List[DatasetDict]) -> DatasetDict['train', 'dev', 'test']:
    
    temp_dict = None
    for dataset in dataset_dicts:

        # Create a temporary dict from the first dataset and concatenate the rest of the datasets to it
        if not temp_dict:
            temp_dict = dataset

        else:        
            temp_dict['train'] = concatenate_datasets([temp_dict['train'], dataset['train']])
            temp_dict['validation'] = concatenate_datasets([temp_dict['validation'], dataset['validation']])
            temp_dict['test'] = concatenate_datasets([temp_dict['test'], dataset['test']])

    return temp_dict