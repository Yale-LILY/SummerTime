import os
from tqdm import tqdm
from typing import Optional, List, Tuple, Generator

from datasets import Dataset, DatasetInfo

from dataset.st_dataset import SummInstance, SummDataset


# Set directory to load non_huggingface dataset scripts
FILE_DIRECTORY_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_NONHUGGINGFACE_DATASETS_PATH = os.path.join(FILE_DIRECTORY_PATH, "non_huggingface_datasets_builders")



# Huggingface Datasets

class CnndmDataset(SummDataset):
    """
    The CNN/DM dataset
    """

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False
    
    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/cnn_dailymail"
    
    def __init__(self):
        super().__init__(dataset_args=('cnn_dailymail', '3.0.0',))

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            article: str = instance['article']
            highlights: str = instance['highlights']
            summ_instance = SummInstance(source=article, summary=highlights)
            
            yield summ_instance



class MultinewsDataset(SummDataset):
    """
    The Multi News dataset
    """

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = True
    
    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/multi_news"
    
    def __init__(self):
        super().__init__(dataset_args=('multi_news',))
        

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            document: list = [doc for doc in instance['document'].split('|||||') if doc]  # removes the empty string generated
                                                                                          # since each doc ends with the delimiting token '|||||'
                                                                                          # the final doc creates an empty string
            summary: str = instance['summary']
            summ_instance = SummInstance(source=document, summary=summary)
            
            yield summ_instance


class SamsumDataset(SummDataset):
    """
    The SAMsum Dataset
    """

    is_query_based = False
    is_dialogue_based = True
    is_multi_document = False
    
    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/samsum"
    
    def __init__(self):
        super().__init__(dataset_args=('samsum',))


    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            dialogue: List = instance['dialogue'].split('\r\n')  # split each dialogue into a list of strings such as
                                                                 # ["speaker1 : utter..", "speaker2 : utter..."]
            summary: str = instance['summary']
            summ_instance = SummInstance(source=dialogue, summary=summary)
            
            yield summ_instance


class XsumDataset(SummDataset):
    """
    The Xsum Dataset
    """
    
    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/xsum"

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False
    
    def __init__(self):
        super().__init__(dataset_args=('xsum',))
        

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            document: List = instance['document']
            summary: str = instance['summary']
            summ_instance = SummInstance(source=document, summary=summary)
            
            yield summ_instance



class PubmedqaDataset(SummDataset):
    """
    The Pubmed QA dataset
    """

    is_query_based = True
    is_dialogue_based = False
    is_multi_document = False
    
    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/pubmed_qa"
    
    def __init__(self, seed=None):
        super().__init__(dataset_args=("pubmed_qa", "pqa_artificial",))
        
        
    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            context: str = " ".join(instance["context"]["contexts"])
            answer: str = instance["long_answer"]
            query: str = instance["question"]
            summ_instance = SummInstance(source=context, summary=answer, query=query)
            
            yield summ_instance



class MlsumDataset(SummDataset):
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
    
    huggingface_dataset = True
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
        super().__init__(dataset_args=(languages,))


    def _load_dataset_safe(self, languages):
        print(MlsumDataset.mlsum_instantiation_guide)

        # Choose languages to download articles
        if languages == "all":
            selected_languages = MlsumDataset.supported_languages
        elif isinstance(languages, list):
            for language in languages:
                assert(self.is_supported(language))
            selected_languages = languages
        else:
            assert(self.is_supported(languages))
            selected_languages = [languages]
            
        # Concatenate selected languaeges into one dataset
        language_datasets = []
        for language in selected_languages:
            dataset = super()._load_dataset_safe("mlsum", language,)
            
            language_datasets.append(dataset)

        mlsum_dataset = self._concatenate_dataset_dicts(language_datasets)
        
        return mlsum_dataset

        
    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            article: List = instance['text']
            summary: str = instance['summary']
            summ_instance = SummInstance(source=article, summary=summary)
            
            yield summ_instance   


    def is_supported(self, language: str):
        if language not in MlsumDataset.supported_languages:
                print(MlsumDataset.mlsum_instantiation_guide)
                raise ValueError(f"The language(s): '{language}' entered is not supported. See above message for usage info")
        else:
            return True




# Non-huggingface datasets

class ScisummnetDataset(SummDataset):
    """
    The SciSummNet dataset. As a dataset not included by huggingface, we need to do manually download, set basic
        information for the dataset
    """
    
    dataset_name = "ScisummNet"
    version = "1.1.0"
    description = "A summary of scientific papers should ideally incorporate the impact of the papers on the " \
                    "research community reflected by citations. To facilitate research in citation-aware scientific " \
                    "paper summarization (Scisumm), the CL-Scisumm shared task has been organized since 2014 for " \
                    "papers in the computational linguistics and NLP domain."

    is_dialogue_based = False
    is_multi_document = False
    is_query_based = False
   
    huggingface_dataset = False
    builder_script_path = os.path.join(BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py")
    
    def __init__(self, seed=None):   
        super().__init__()


    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            docs: List = [instance['document_xml'], instance['citing_sentences_annotated.json']]
            summary: str = instance['summary']
            summ_instance = SummInstance(source=docs, summary=summary)
            
            yield summ_instance



class SummscreenDataset(SummDataset):
    """
    The SummScreen dataset. As a dataset not included by huggingface, we need to do manually download, set basic
        information for the dataset
    """
      
    dataset_name = "Summscreen"
    version = "1.1.0"
    is_dialogue_based = True
    is_multi_document = False
    is_query_based = False

    huggingface_dataset = False
    builder_script_path = os.path.join(BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py")
    
    def __init__(self, seed=None):   
        super().__init__()


    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            transcript: List = instance['transcript']   #convert string into a list of string dialogues
            recap: str = instance['recap']
            summ_instance = SummInstance(source=transcript, summary=recap)

            yield summ_instance



class QMsumDataset(SummDataset):
    """
    QMSum Dataset
    """

    dataset_name = "QMsum"
    description = '''
    QMSum is a new human-annotated benchmark for query-based multi-domain meeting summarization task,
    which consists of 1,808 query-summary pairs over 232 meetings in multiple domains.
    '''

    is_dialogue_based = True
    is_multi_document = False
    is_query_based = True

    huggingface_dataset = False
    builder_script_path = os.path.join(BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py")
    
    
    def __init__(self):   
        super().__init__()


    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            for query_set in instance['general_query_list'] + instance['specific_query_list']:
                meeting: List = [utterance['speaker'] + " : " + utterance['content']\
                                for utterance in instance['meeting_transcripts']]
                query: str = query_set['query']
                summary: str = query_set['answer']    
                summ_instance = SummInstance(source=meeting, summary=summary, query=query)

            yield summ_instance



class ArxivDataset(SummDataset):
    """
    The Arxiv Dataset
    """    
    dataset_name = "Arxiv_longsummarization"
    description = '''
    A summarization dataset comprised of pairs of scientific papers. 
    The dataset provides a challenging testbed for abstractive summarization. 
    It contains papers and their abstracts.
    '''

    is_dialogue_based = False
    is_multi_document = False
    is_query_based = False

    huggingface_dataset = False
    builder_script_path = os.path.join(BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py")
    
    def __init__(self):
    
        print("*****************",\
              "***Attention***",\
              "This dataset is quite large (approx 5Gb and will need about 15 Gb for the extraction process",\
              "Cancel/interrupt the download if size and time constraints will not be met",\
              "*****************", sep="\n")
          
        super().__init__()


    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data):
            article: List = instance['article_text']
            abstract: str = " ".join(instance['abstract_text'])
            summ_instance = SummInstance(source=article, summary=abstract)

            yield summ_instance
