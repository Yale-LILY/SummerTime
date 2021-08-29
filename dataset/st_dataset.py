import abc
import sys
import time
import json
from typing import Dict, List, Optional, Union, Generator

from datasets import Dataset, DatasetDict, DatasetInfo, concatenate_datasets, load_dataset


DEFAULT_NUMBER_OF_RETRIES_ALLOWED = 5
DEFAULT_WAIT_SECONDS_BEFORE_RETRY = 5

TEST_OR_VAL_SPLIT_RATIO = .1



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

    def __repr__(self):
        instance_dict = {"source": self.source, "summary":self.summary}
        if self.query:
            instance_dict["query"] = self.query

        return str(instance_dict)

    def __str__(self):
        instance_dict = {"source": self.source, "summary":self.summary}
        if self.query:
            instance_dict["query"] = self.query

        return json.dumps(instance_dict, indent=1)



class SummDataset:
    """
    Dataset class for summarization, which takes into account of the following tasks:
        * Single document summarization
        * Multi-document/Dialogue summarization
        * Query-based summarization
    """

    def __init__(self,
                 dataset_args=None,
                 splitseed=None):
        """ Create dataset information from the huggingface Dataset class """


        # Load dataset from huggingface, use default huggingface arguments
        if self.huggingface_dataset:
            dataset = self._load_dataset_safe(*dataset_args)
        # Load non-huggingface dataset, use custom dataset builder
        else:
            dataset = self._load_dataset_safe(path=self.builder_script_path)

        info_set = self._get_dataset_info(dataset)


        # Ensure any dataset with a val or dev or validation split is standardised to validation split
        if 'val' in dataset:
            dataset['validation'] = dataset['val']
            dataset.remove('val')
        elif 'dev' in dataset:
            dataset['validation'] = dataset['dev']
            dataset.remove('dev')


        # If no splits other other than training, generate them
        assert 'train' in dataset or 'validation' in dataset  or 'test' in dataset, "At least one of train/validation test needs to be not empty!"
        
        if not ('validation' in dataset  or 'test' in dataset):
            dataset = self._generate_missing_val_test_splits(dataset, splitseed)
        

        self.dataset_name = type(self).__name__
        self.description = info_set.description
        self.citation = info_set.citation
        self.homepage = info_set.homepage

        # Extract the dataset entries from folders and load into dataset
        self._train_set = self._process_data(dataset["train"])
        self._validation_set = self._process_data(dataset["validation"])    # Some datasets have a validation split
        self._test_set = self._process_data(dataset["test"])

        
    @property
    def train_set(self) -> Union[Generator[SummInstance, None, None], List]:
        if self._train_set is not None:
            return self._train_set
        else:
            print(f"{self.dataset_name} does not contain a train set, empty list returned")
            return list()

    @property
    def validation_set(self) -> Union[Generator[SummInstance, None, None], List]:
        if self._validation_set is not None:
            return self._validation_set
        else:
            print(f"{self.dataset_name} does not contain a validation set, empty list returned")
            return list()
    
    @property
    def test_set(self) -> Union[Generator[SummInstance, None, None], List]:
        if self._test_set is not None:
            return self._test_set
        else:
            print(f"{self.dataset_name} does not contain a test set, empty list returned")
            return list()



    def _load_dataset_safe(self, *args, **kwargs) -> Dataset:
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


    def _get_dataset_info(self, data_dict: DatasetDict) -> DatasetInfo:
        ''' 
        Get the information set from the dataset
        The information set contains: dataset name, description, version, citation and licence
        :param data_dict: DatasetDict
        :rtype: DatasetInfo 
        '''
        return data_dict["train"].info



    @abc.abstractmethod
    def _process_data(self, dataset: Dataset) -> Generator[SummInstance, None, None]:
        """
        Abstract class method to process the data contained within each dataset.
        Each dataset class processes it's own information differently due to the diversity in domains
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        return



    def _generate_missing_val_test_splits(self, dataset_dict: DatasetDict, seed: int) -> DatasetDict:
        """
        Creating the train, val and test splits from a dataset
            the generated sets are 'train: ~.80', 'validation: ~.10', and 'test: ~10' in size
            the splits are randomized for each object unless a seed is provided for the random generator
        
        :param dataset: Arrow Dataset with containing, usually the train set
        :param seed: seed for the random generator to shuffle the dataset
        :rtype: Arrow DatasetDict containing the three splits
        """

        if 'train' not in dataset_dict:
            if 'validation' not in dataset_dict:
                dataset_dict['validation'] = None
            if 'test' not in dataset_dict:
                dataset_dict['test'] = None

            return dataset_dict
        
        # Create a 'test' split from 'train' if none is available
        if 'test' not in  dataset_dict:
            dataset_traintest_split = dataset_dict['train'].train_test_split(test_size=TEST_OR_VAL_SPLIT_RATIO, seed=seed)
            dataset_dict['train'] = dataset_traintest_split['train']
            dataset_dict['test'] = dataset_traintest_split['test']
        
        
        # Create a 'validation' split from the new 'train' set if no 'validation' split is available
        if 'validation' not in dataset_dict:
            dataset_trainval_split = dataset_dict['train'].train_test_split(test_size=TEST_OR_VAL_SPLIT_RATIO, seed=seed)      
            dataset_dict['train'] = dataset_traintest_split['train']
            dataset_dict['validation'] = dataset_traintest_split['test']

        return dataset_dict



    def _concatenate_dataset_dicts(self, dataset_dicts: List[DatasetDict]) -> DatasetDict:    
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

        
    def show_description(self):
        '''
        Print the description of the dataset.
        '''
        print(self.dataset_name, ":\n", self.description)
