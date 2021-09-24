import re

from abc import abstractmethod
from pprint import pformat
from time import sleep
from typing import List, Tuple, Optional, Union, Generator

from datasets import (
    Dataset,
    DatasetDict,
    DatasetInfo,
    concatenate_datasets,
    load_dataset,
)

# Defualt values for retrying dataset download
DEFAULT_NUMBER_OF_RETRIES_ALLOWED = 5
DEFAULT_WAIT_SECONDS_BEFORE_RETRY = 5

# Default value for creating missing val/test splits
TEST_OR_VAL_SPLIT_RATIO = 0.1


class SummInstance:
    """
    Basic instance for summarization tasks
    """

    def __init__(
        self, source: Union[List[str], str], summary: str, query: Optional[str] = None
    ):
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
        instance_dict = {"source": self.source, "summary": self.summary}
        if self.query:
            instance_dict["query"] = self.query

        return str(instance_dict)

    def __str__(self):
        instance_dict = {"source": self.source, "summary": self.summary}
        if self.query:
            instance_dict["query"] = self.query

        return pformat(instance_dict, indent=1)

    def ensure_dialogue_format(self):
        pattern = re.compile(r"\w+\s:\s\w+")

        assert isinstance(
            self.source, list
        ), "Source should be a list of strings for dialogue"

        for i in range(len(self.source)):
            if not pattern.match(self.source[i]):
                self.source[i] = f"None : {self.source[i]}"


class SummDataset:
    """
    Dataset class for summarization, which takes into account of the following tasks:
        * Single document summarization
        * Multi-document/Dialogue summarization
        * Query-based summarization
    """

    def __init__(
        self,
        dataset_args: Optional[Tuple[str]] = (),
        dataset_kwargs: Optional[Tuple[str]] = {},
        splitseed: Optional[int] = None,
    ):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param dataset_args: a tuple containing arguments to passed on to the 'load_dataset_safe' method.
            The args are used by the huggingface 'load_dataset' method
            The arguments for each dataset are different and comprise of a string or multiple strings
        :param dataset_kwargs: a dictionary containing keyword arguments to be passed on to the 'load_dataset_safe' method
        :param splitseed: a number to instantiate the random generator used to generate val/test splits
            for the datasets without them
        """

        # Load dataset from huggingface, use default huggingface arguments
        dataset = self._load_dataset_safe(dataset_args, dataset_kwargs)

        info_set = self._get_dataset_info(dataset)

        # Ensure any dataset with a val or dev or validation split is standardised to validation split
        if "val" in dataset:
            dataset["validation"] = dataset["val"]
            dataset.remove("val")
        elif "dev" in dataset:
            dataset["validation"] = dataset["dev"]
            dataset.remove("dev")

        # If no splits other other than training, generate them
        assert (
            "train" in dataset or "validation" in dataset or "test" in dataset
        ), "At least one of train/validation test needs to be not empty!"

        if not ("validation" in dataset or "test" in dataset):
            dataset = self._generate_missing_val_test_splits(dataset, splitseed)

        self.description = info_set.description
        self.citation = info_set.citation
        self.homepage = info_set.homepage

        # Extract the dataset entries from folders and load into dataset
        self._train_set = self._process_data(dataset["train"])
        self._validation_set = self._process_data(
            dataset["validation"]
        )  # Some datasets have a validation split
        self._test_set = self._process_data(dataset["test"])

    @property
    def train_set(self) -> Union[Generator[SummInstance, None, None], List]:
        if self._train_set is not None:
            return self._train_set
        else:
            print(
                f"{self.dataset_name} does not contain a train set, empty list returned"
            )
            return list()

    @property
    def validation_set(self) -> Union[Generator[SummInstance, None, None], List]:
        if self._validation_set is not None:
            return self._validation_set
        else:
            print(
                f"{self.dataset_name} does not contain a validation set, empty list returned"
            )
            return list()

    @property
    def test_set(self) -> Union[Generator[SummInstance, None, None], List]:
        if self._test_set is not None:
            return self._test_set
        else:
            print(
                f"{self.dataset_name} does not contain a test set, empty list returned"
            )
            return list()

    def _load_dataset_safe(self, args=(), kwargs={}) -> Dataset:
        """
        This method creates a wrapper around the huggingface 'load_dataset()' function for a more robust download function,
            the original 'load_dataset()' function occassionally fails when it cannot reach a server especially after multiple requests.
            This method tackles this problem by attempting the download multiple times with a wait time before each retry

        The wrapper method passes all arguments and keyword arguments to the 'load_dataset' function with no alteration.
        :rtype: Dataset
        :param args: non-keyword arguments to passed on to the 'load_dataset' function
        :param kwargs: keyword arguments to passed on to the 'load_dataset' function
        """

        tries = DEFAULT_NUMBER_OF_RETRIES_ALLOWED
        wait_time = DEFAULT_WAIT_SECONDS_BEFORE_RETRY

        for i in range(tries):
            try:
                dataset = load_dataset(*args, **kwargs)
            except ConnectionError:
                if i < tries - 1:  # i is zero indexed
                    sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        "Wait for a minute and attempt downloading the dataset again. \
                         The server hosting the dataset occassionally times out."
                    )
            break

        return dataset

    def _get_dataset_info(self, data_dict: DatasetDict) -> DatasetInfo:
        """
        Get the information set from the dataset
        The information set contains: dataset name, description, version, citation and licence
        :param data_dict: DatasetDict
        :rtype: DatasetInfo
        """
        return data_dict["train"].info

    @abstractmethod
    def _process_data(self, dataset: Dataset) -> Generator[SummInstance, None, None]:
        """
        Abstract class method to process the data contained within each dataset.
        Each dataset class processes it's own information differently due to the diversity in domains
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object,
            the SummInstance has the following properties [source, summary, query[optional]]
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        return

    def _generate_missing_val_test_splits(
        self, dataset_dict: DatasetDict, seed: int
    ) -> DatasetDict:
        """
        Creating the train, val and test splits from a dataset
            the generated sets are 'train: ~.80', 'validation: ~.10', and 'test: ~10' in size
            the splits are randomized for each object unless a seed is provided for the random generator

        :param dataset: Arrow Dataset with containing, usually the train set
        :param seed: seed for the random generator to shuffle the dataset
        :rtype: Arrow DatasetDict containing the three splits
        """

        # Return dataset if no train set available for splitting
        if "train" not in dataset_dict:
            if "validation" not in dataset_dict:
                dataset_dict["validation"] = None
            if "test" not in dataset_dict:
                dataset_dict["test"] = None

            return dataset_dict

        # Create a 'test' split from 'train' if no 'test' set is available
        if "test" not in dataset_dict:
            dataset_traintest_split = dataset_dict["train"].train_test_split(
                test_size=TEST_OR_VAL_SPLIT_RATIO, seed=seed
            )
            dataset_dict["train"] = dataset_traintest_split["train"]
            dataset_dict["test"] = dataset_traintest_split["test"]

        # Create a 'validation' split from the remaining 'train' set if no 'validation' set is available
        if "validation" not in dataset_dict:
            dataset_trainval_split = dataset_dict["train"].train_test_split(
                test_size=TEST_OR_VAL_SPLIT_RATIO, seed=seed
            )
            dataset_dict["train"] = dataset_trainval_split["train"]
            dataset_dict["validation"] = dataset_trainval_split["test"]

        return dataset_dict

    def _concatenate_dataset_dicts(
        self, dataset_dicts: List[DatasetDict]
    ) -> DatasetDict:
        """
        Concatenate two dataset dicts with similar splits and columns tinto one
        :param dataset_dicts: A list of DatasetDicts
        :rtype: DatasetDict containing the combined data
        """

        # Ensure all dataset dicts have the same splits
        setsofsplits = set(tuple(dataset_dict.keys()) for dataset_dict in dataset_dicts)
        if len(setsofsplits) > 1:
            raise ValueError("Splits must match for all datasets")

        # Concatenate all datasets into one according to the splits
        temp_dict = {}
        for split in setsofsplits.pop():
            split_set = [dataset_dict[split] for dataset_dict in dataset_dicts]
            temp_dict[split] = concatenate_datasets(split_set)

        return DatasetDict(temp_dict)

    @classmethod
    def generate_basic_description(cls) -> str:
        """
        Automatically generate the basic description string based on the attributes
        :rtype: string containing the description
        :param cls: class object
        """

        basic_description = (
            f": {cls.dataset_name} is a "
            f"{'query-based ' if cls.is_query_based else ''}"
            f"{'dialogue ' if cls.is_dialogue_based else ''}"
            f"{'multi-document' if cls.is_multi_document else 'single-document'} "
            f"summarization dataset."
        )

        return basic_description

    def show_description(self):
        """
        Print the description of the dataset.
        """
        print(self.dataset_name, ":\n", self.description)

class CustomDataset(SummDataset):
    """
    This is a dataset class that acts as an API to load custom user dataset
    The dataset class to contains data that the users source themselves or from a third party
    Once created, it can be used with any of our models and contains most properties and methods
        similar to other SummerTime datasets
    """
    ## TODO: Create a caching mechanism for user created datasets

    def __init__(self,
                train_set=[],
                test_set=[],
                validation_set=[],
                query_based=False, 
                multi_doc=False, 
                dialogue_based=False):
        """Create dataset information from the huggingface Dataset class
        :rtype: dataset object
        :param train_set: List[Dictionary], list of dictionaries that contain a data instance.
            Contains the training examples and is in the form listed below.
                The dictionary is in the form:
                    {"source": "source_data", "summary": "summary_data", "query":"query_data"}
                        * source_data is either of type List[str] or str
                        * summary_data is of type str
                        * query_data is of type str
                The list of dictionaries looks as follows:
                    [dictionary_instance_1, dictionary_instance_2, ...]
        :param validation_set: Optional[List[Dictionary]], similar format to train_set
            Contains the validation examples
        :param test_set: Optional[List[Dictionary]], similar format to train_set
            Contains the test examples
        :param query_based: bool, Is the dataset query-based?
        :param multi_doc: bool, Does each dataset instance source contain multiple documents
        :param dialogue_based: Is the dataset dialogue-based?
        """

        if not (train_set or test_set or validation_set):
            raise ValueError("Missing data for the dataset")

        self.is_dialogue_based = dialogue_based
        self.is_multi_document = multi_doc
        self.is_query_based = query_based

        # Load the data into their respective splits
        self._train_set = self._process_data(train_set, "Train")
        self._validation_set = self._process_data(validation_set, "Validation")
        self._test_set = self._process_data(test_set, "Test")

        dataset_name = None
        version = None

        self.dataset_name = "Custom"


    def _process_data(self, data: Dataset, split) -> Generator[SummInstance, None, None]:
        """
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        data_instances = []

        summaries_present = False
        for instance in data:
            
            if "source" in instance and instance["source"]:
                if self.is_dialogue_based or self.is_multi_document:
                    source: List = instance["source"]
                else:
                    source: str = instance["source"]
            else:
                raise TypeError("Missing source for a dataset instance")

            ## TODO: Ensure models can handle datasets with no summaries 
            summary = None
            if "summary" in instance and instance["summary"]:
                summaries_present = True
                summary: str = instance["summary"]

            query = None
            if self.is_query_based:
                if "query" in instance and instance["query"]:
                    query: str = instance["query"]
                else:
                    raise TypeError("Missing query for a query-based dataset")

            data_instances.append(SummInstance(source=source, summary=summary, query=query))

        if not summaries_present:
            print("\nATTENTION:", split, "split does not contain summaries.",
                  "Proceed if this is intended.\n")

        return (data for data in data_instances)