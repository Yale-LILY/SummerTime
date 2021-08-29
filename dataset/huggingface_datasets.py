from tqdm.auto import tqdm
from typing import Optional, List, Tuple, Generator

from datasets import Dataset

from dataset.st_dataset import SummInstance, SummDataset, generate_train_dev_test_splits, concatenate_dataset_dicts


class HuggingfaceDataset(SummDataset):
    """
    A base class for all datasets currently supported by Huggingface
    """
    def __init__(self,
                 info_set: Dataset,
                 huggingface_page: str,
                 train_set: Optional[Generator[SummInstance, None, None]] = None,
                 dev_set: Optional[Generator[SummInstance, None, None]] = None,
                 test_set: Optional[Generator[SummInstance, None, None]] = None
                 ):
        """ Create dataset information from the huggingface Dataset class """
        
        super(HuggingfaceDataset, self).__init__(
            info_set.builder_name,
            info_set.description,
            citation=info_set.citation,
            homepage=info_set.homepage,
            huggingface_page=huggingface_page,
            train_set=train_set,
            dev_set=dev_set,
            test_set=test_set
        )


class CnndmDataset(HuggingfaceDataset):
    """
    The CNN/DM dataset
    """

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False
    
    huggingface_page = "https://huggingface.co/datasets/cnn_dailymail"
    
    def __init__(self):
        # Load the train, dev and test set from the huggingface datasets
        cnn_dataset = self.load_dataset_safe('cnn_dailymail', '3.0.0')
        info_set = cnn_dataset['train']
        
        processed_train_set = CnndmDataset.process_cnndm_data(cnn_dataset['train'])
        processed_dev_set = CnndmDataset.process_cnndm_data(cnn_dataset['validation'])
        processed_test_set = CnndmDataset.process_cnndm_data(cnn_dataset['test'])
        
        super().__init__(info_set,
                         huggingface_page=CnndmDataset.huggingface_page,
                         train_set=processed_train_set,
                         dev_set=processed_dev_set,
                         test_set=processed_test_set)
        
    @staticmethod
    def process_cnndm_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True, ascii=True):
            article: str = instance['article']
            highlights: str = instance['highlights']
            summ_instance = SummInstance(source=article, summary=highlights)
            
            yield summ_instance



class MultinewsDataset(HuggingfaceDataset):
    """
    The Multi News dataset
    """

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = True
    
    huggingface_page = "https://huggingface.co/datasets/multi_news"
    
    def __init__(self):
        # Load the train, dev and test set from the huggingface datasets
        multinews_dataset = self.load_dataset_safe("multi_news")
        info_set = multinews_dataset['train']
        
        processed_train_set = MultinewsDataset.process_multinews_data(multinews_dataset['train'])
        processed_dev_set = MultinewsDataset.process_multinews_data(multinews_dataset['validation'])
        processed_test_set = MultinewsDataset.process_multinews_data(multinews_dataset['test'])
        
        super().__init__(info_set,
                         huggingface_page=MultinewsDataset.huggingface_page,
                         train_set=processed_train_set,
                         dev_set=processed_dev_set,
                         test_set=processed_test_set)
        
    @staticmethod
    def process_multinews_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True, ascii=True):
            
            document: list = [doc for doc in instance['document'].split('|||||') if doc]  # removes the empty string generated
                                                                                          # since each doc ends with the delimiting token '|||||'
                                                                                          # the final doc creates an empty string
            summary: str = instance['summary']
            summ_instance = SummInstance(source=document, summary=summary)
            
            yield summ_instance


class SamsumDataset(HuggingfaceDataset):
    """
    The SAMsum Dataset
    """

    is_query_based = False
    is_dialogue_based = True
    is_multi_document = False
    
    huggingface_page = "https://huggingface.co/datasets/samsum"
    
    def __init__(self):
        # Load the train, dev and test set from the huggingface datasets
        samsum_dataset = self.load_dataset_safe('samsum')
        info_set = samsum_dataset['train']
        
        processed_train_set = SamsumDataset.process_samsum_data(samsum_dataset['train'])
        processed_dev_set = SamsumDataset.process_samsum_data(samsum_dataset['validation'])
        processed_test_set = SamsumDataset.process_samsum_data(samsum_dataset['test'])
        
        super().__init__(info_set,
                         huggingface_page=SamsumDataset.huggingface_page,
                         train_set=processed_train_set,
                         dev_set=processed_dev_set,
                         test_set=processed_test_set)
        
    @staticmethod
    def process_samsum_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True, ascii=True):
            dialogue: List = instance['dialogue'].split('\r\n')  # split each dialogue into a list of strings such as
                                                                 # ["speaker1 : utter..", "speaker2 : utter..."]
            summary: str = instance['summary']
            summ_instance = SummInstance(source=dialogue, summary=summary)
            
            yield summ_instance


class XsumDataset(HuggingfaceDataset):
    """
    The Xsum Dataset
    """
    
    huggingface_page = "https://huggingface.co/datasets/xsum"

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False
    
    def __init__(self):
        # Load the train, dev and test set from the huggingface datasets
        xsum_dataset = self.load_dataset_safe("xsum")
        info_set = xsum_dataset['train']
        
        processed_train_set = XsumDataset.process_xsum_data(xsum_dataset['train'])
        processed_dev_set = XsumDataset.process_xsum_data(xsum_dataset['validation'])
        processed_test_set = XsumDataset.process_xsum_data(xsum_dataset['test'])
        
        super().__init__(info_set,
                         huggingface_page=XsumDataset.huggingface_page,
                         train_set=processed_train_set,
                         dev_set=processed_dev_set,
                         test_set=processed_test_set)
        
    @staticmethod
    def process_xsum_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True, ascii=True):
            document: List = instance['document']
            summary: str = instance['summary']
            summ_instance = SummInstance(source=document, summary=summary)
            
            yield summ_instance



class PubmedqaDataset(HuggingfaceDataset):
    """
    The Pubmed QA dataset
    """

    is_query_based = True
    is_dialogue_based = False
    is_multi_document = False
    
    huggingface_page = "https://huggingface.co/datasets/pubmed_qa"
    
    def __init__(self, seed=None):
        # Load the train, dev and test set from the huggingface datasets
        pubmedqa_dataset = self.load_dataset_safe("pubmed_qa", "pqa_artificial")
        info_set = pubmedqa_dataset['train']

        # No dev and test splits provided; hence creating these splits from the train set 
        pubmedqa_split_dataset = generate_train_dev_test_splits(pubmedqa_dataset['train'], seed=seed)

        processed_train_set = PubmedqaDataset.process_pubmedqa_data(pubmedqa_split_dataset['train'])
        processed_dev_set = PubmedqaDataset.process_pubmedqa_data(pubmedqa_split_dataset['dev'])
        processed_test_set = PubmedqaDataset.process_pubmedqa_data(pubmedqa_split_dataset['test'])
        
        super().__init__(info_set,
                         huggingface_page=PubmedqaDataset.huggingface_page,
                         train_set=processed_train_set,
                         dev_set=processed_dev_set,
                         test_set=processed_test_set)
        
        
    @staticmethod
    def process_pubmedqa_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True, ascii=True):
            context: str = " ".join(instance["context"]["contexts"])
            answer: str = instance["long_answer"]
            query: str = instance["question"]
            summ_instance = SummInstance(source=context, summary=answer, query=query)
            
            yield summ_instance



class MlsumDataset(HuggingfaceDataset):
    """
    The MLsum Dataset - A multi-lingual dataset featuring 5 languages
    Includes 1.5 million news articles and their corresponding summaries

    "de" - German
    "es" - Spanish
    "fr" - French
    "ru" - Russian
    "tu" - Turkish
    """

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False
    
    huggingface_page = "https://huggingface.co/datasets/mlsum"
    supported_languages = ["de", "es", "fr", "ru", "tu"]

    mlsum_instantiation_guide = '''The languages supported for the Mlsum Dataset are: 
                de - German
                es - Spanish
                fr - French
                ru - Russian
                tu - Turkish
                    
                Examples to instantiate the dataset:
                1. Dataset with only one language
                   dataset = MlsumDataset({language_token})
                   dataset = MlsumDataset("es")
                   dataset = MlsumDataset("tu")...

                2. Dataset with a multiple languages
                   dataset = MlsumDataset({list of language_token})
                   dataset = MlsumDataset(["es","de"])
                   dataset = MlsumDataset(["es","de", "tu"])...

                3. Dataset with all supported languages (default)
                   dataset = MlsumDataset(all)
                   dataset = MlsumDataset() 
                   
                '''
    
    def __init__(self, languages="all"):
        
        print(MlsumDataset.mlsum_instantiation_guide)

        # Choose languages to download articles
        if languages == "all":
            selected_languages = MlsumDataset.supported_languages
        elif isinstance(languages, list):
            for language in languages:
                assert(MlsumDataset.is_supported(language))
            selected_languages = languages
        else:
            assert(MlsumDataset.is_supported(languages))
            selected_languages = [languages]
            
        # Concatenate selected languaeges into one dataset
        language_datasets = []
        for language in selected_languages:
            dataset = self.load_dataset_safe("mlsum", language)
            language_datasets.append(dataset)

        mlsum_dataset = concatenate_dataset_dicts(language_datasets)

        info_set = mlsum_dataset['train']
        
        # Load the train, dev and test set from the huggingface datasets
        processed_train_set = MlsumDataset.process_mlsum_data(mlsum_dataset['train'])
        processed_dev_set = MlsumDataset.process_mlsum_data(mlsum_dataset['validation'])
        processed_test_set = MlsumDataset.process_mlsum_data(mlsum_dataset['test'])
        
        super().__init__(info_set,
                         huggingface_page=MlsumDataset.huggingface_page,
                         train_set=processed_train_set,
                         dev_set=processed_dev_set,
                         test_set=processed_test_set)
        
    @staticmethod
    def process_mlsum_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True, ascii=True):
            article: List = instance['text']
            summary: str = instance['summary']
            summ_instance = SummInstance(source=article, summary=summary)
            
            yield summ_instance   

    @staticmethod
    def is_supported(language: str):
        if language not in MlsumDataset.supported_languages:
                print(MlsumDataset.mlsum_instantiation_guide)
                raise ValueError(f"The language(s): '{language}' entered is not supported. See above message for usage info")
        else:
            return True
