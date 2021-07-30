from datasets import *
from typing import Dict, List, Optional, Union


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
