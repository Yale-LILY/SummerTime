import datasets
from tqdm import tqdm
from datasets import Dataset

from typing import Optional, List, Tuple
from dataset.st_dataset import SummInstance, SummDataset


class HuggingfaceDataset(SummDataset):
    """
    A base class for all datasets currently supported by Huggingface
    """
    def __init__(self,
                 info_set: Dataset,
                 huggingface_page: str,
                 is_query_based: bool,
                 is_dialogue_based: bool,
                 is_multi_document: bool,
                 train_set: Optional[List[SummInstance]] = None,
                 dev_set: Optional[List[SummInstance]] = None,
                 test_set: Optional[List[SummInstance]] = None
                 ):
        """ Create dataset information from the huggingface Dataset class """
        
        super(HuggingfaceDataset, self).__init__(
            info_set.builder_name,
            info_set.description,
            citation=info_set.citation,
            homepage=info_set.homepage,
            huggingface_page=huggingface_page,
            is_query_based=is_query_based,
            is_dialogue_based=is_dialogue_based,
            is_multi_document=is_multi_document,
            train_set=train_set,
            dev_set=dev_set,
            test_set=test_set
        )


class CnndmDataset(HuggingfaceDataset):
    """
    The CNN/DM dataset
    """
    
    huggingface_page = "https://huggingface.co/datasets/cnn_dailymail"
    
    def __init__(self):
        # load the train, dev and test set from the huggingface datasets
        cnn_dataset = datasets.load_dataset('cnn_dailymail', '3.0.0')
        info_set = cnn_dataset['train']
        
        processed_train_set = CnndmDataset.process_cnndm_data(cnn_dataset['train'])
        processed_dev_set = CnndmDataset.process_cnndm_data(cnn_dataset['validation'])
        processed_test_set = CnndmDataset.process_cnndm_data(cnn_dataset['test'])
        
        super().__init__(info_set,
                         huggingface_page=CnndmDataset.huggingface_page,
                         is_query_based=False,
                         is_dialogue_based=False,
                         is_multi_document=False,
                         train_set=processed_train_set,
                         dev_set=processed_dev_set,
                         test_set=processed_test_set)
        
    @staticmethod
    def process_cnndm_data(data: Dataset) -> List[SummInstance]:
        for instance in tqdm(data):
            article: str = instance['article']
            highlights: str = instance['highlights']
            summ_instance = SummInstance(article, highlights)
            
            yield summ_instance



class MultinewsDataset(HuggingfaceDataset):
    """
    The Multi News dataset
    """
    
    huggingface_page = "https://huggingface.co/datasets/multi_news"
    
    def __init__(self):
        # load the train, dev and test set from the huggingface datasets
        multinews_dataset = datasets.load_dataset("multi_news")
        info_set = multinews_dataset['train']
        
        processed_train_set = MultinewsDataset.process_multinews_data(multinews_dataset['train'])
        processed_dev_set = MultinewsDataset.process_multinews_data(multinews_dataset['validation'])
        processed_test_set = MultinewsDataset.process_multinews_data(multinews_dataset['test'])
        
        super().__init__(info_set,
                         huggingface_page=MultinewsDataset.huggingface_page,
                         is_query_based=False,
                         is_dialogue_based=False,
                         is_multi_document=True,
                         train_set=processed_train_set,
                         dev_set=processed_dev_set,
                         test_set=processed_test_set)
        
    @staticmethod
    def process_multinews_data(data: Dataset) -> List[SummInstance]:
        for instance in tqdm(data):
            
            document: list = [doc for doc in instance['document'].split('|||||') if doc]  # removes the empty string generated
                                                                                          # since each doc ends with the delimiting token '|||||'
                                                                                          # the final doc creates an empty string
            summary: str = instance['summary']
            summ_instance = SummInstance(document, summary)
            
            yield summ_instance


class SamsumDataset(HuggingfaceDataset):
    """
    The SAMsum
    """
    
    huggingface_page = "https://huggingface.co/datasets/samsum"
    
    def __init__(self):
        # load the train, dev and test set from the huggingface datasets
        samsum_dataset = datasets.load_dataset('samsum')
        info_set = samsum_dataset['train']
        
        processed_train_set = SamsumDataset.process_samsum_data(samsum_dataset['train'])
        processed_dev_set = SamsumDataset.process_samsum_data(samsum_dataset['validation'])
        processed_test_set = SamsumDataset.process_samsum_data(samsum_dataset['test'])
        
        super().__init__(info_set,
                         huggingface_page=SamsumDataset.huggingface_page,
                         is_query_based=False,
                         is_dialogue_based=True,
                         is_multi_document=False,
                         train_set=processed_train_set,
                         dev_set=processed_dev_set,
                         test_set=processed_test_set)
        
    @staticmethod
    def process_samsum_data(data: Dataset) -> List[SummInstance]:
        for instance in tqdm(data):
            dialogue: str = instance['dialogue']
            summary: str = instance['summary']
            summ_instance = SummInstance(dialogue, summary)
            
            yield summ_instance

