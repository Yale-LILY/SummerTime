import tempfile
import zipfile
import os
import json
import random
import gdown
from urllib.request import urlretrieve, urlopen
from typing import Optional, List, Tuple
# from st_dataset import SummInstance, SummDataset
from dataset.st_dataset import SummInstance, SummDataset

import ast
import datasets
from datasets import Dataset
from tqdm import tqdm


BASE_NONHUGGINGFACE_DATASETS_PATH = os.path.join(os.getcwd(), "dataset", "non_huggingface_datasets_builders")


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

                    
    builder_script_path = os.path.join(BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py")
    
    def __init__(self):   


        # Load dataset
        scisummnet_dataset = datasets.load_dataset(path=ScisummnetDataset.builder_script_path)
        info_set = scisummnet_dataset["train"]
        
        #  Process the train, dev and test se
        processed_dataset = ScisummnetDataset.process_scisummnet_data(scisummnet_dataset["train"])
        
        # Randomize and split data into train, dev and test sets
        processed_dataset = list(processed_dataset)
        random.shuffle(processed_dataset)
        split_1 = int(0.8 * len(processed_dataset))
        split_2 = int(0.9 * len(processed_dataset))

        processed_train_set = processed_dataset[:split_1]
        processed_dev_set = processed_dataset[split_1:split_2]
        processed_test_set = processed_dataset[split_2:]

        super().__init__(ScisummnetDataset.dataset_name,
                         ScisummnetDataset.description,
                         is_dialogue_based=False,
                         is_multi_document=False,
                         is_query_based=False,
                         train_set=processed_train_set,  
                         dev_set=processed_dev_set, 
                         test_set=processed_test_set, 
                         )

    @staticmethod
    def process_scisummnet_data(data: Dataset) -> List[SummInstance]:
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

    builder_script_path = os.path.join(BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py")
    
    def __init__(self):
        
        # Load dataset
        summscreen_dataset = datasets.load_dataset(path=SummscreenDataset.builder_script_path)
        info_set = summscreen_dataset["train"]
        
        #  Process the train, dev and test se
        processed_train_set = SummscreenDataset.process_summscreen_data(summscreen_dataset["train"])
        processed_dev_set = SummscreenDataset.process_summscreen_data(summscreen_dataset["validation"]) 
        processed_test_set = SummscreenDataset.process_summscreen_data(summscreen_dataset["test"])
        
        #  Process the train, dev and test set and replace the last three args in __init__() below
        dataset_name = "SummScreen_fd+tms_tokenized"
        description = "A summarization dataset comprised of pairs of TV series transcripts and human written recaps. \
                        The dataset provides a challenging testbed for abstractive summarization. \
                        It contains transcripts from FoeverDreaming (fd) and TVMegaSite.\
                        The current version being used is the one where the transcripts have already been tokenized."
        super().__init__(dataset_name,
                         description,
                         is_dialogue_based=True,
                         is_multi_document=False,
                         is_query_based=False,
                         train_set=processed_train_set,  
                         dev_set=processed_dev_set, 
                         test_set=processed_test_set, 
                         )
    
    @staticmethod
    def process_summscreen_data(data: Dataset) -> List[SummInstance]:
        processed_set = []
        for instance in tqdm(data):
            transcript: List = instance['transcript']   #convert string into a list of string dialogues
            recap: str = instance['recap']
            summ_instance = SummInstance(source=transcript, summary=recap)
            processed_set.append(summ_instance)
            
        return processed_set




class QMsumDataset(SummDataset):
    """
    QMSum Dataset
    """

    dataset_name = "QMsum"
    description = '''
    QMSum is a new human-annotated benchmark for query-based multi-domain meeting summarization task,
    which consists of 1,808 query-summary pairs over 232 meetings in multiple domains.
    '''
    version = "1.0.0"

    builder_script_path = os.path.join(BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py")
    
    def __init__(self):
        
        # Load dataset
        qmsum_dataset = datasets.load_dataset(path=QMsumDataset.builder_script_path)
        info_set = qmsum_dataset["train"]

        # Extract the dataset entries from folders and load into dataset
        processed_train_set = QMsumDataset.process_qmsum_data(qmsum_dataset["train"])
        processed_dev_set = QMsumDataset.process_qmsum_data(qmsum_dataset["validation"])
        processed_test_set = QMsumDataset.process_qmsum_data(qmsum_dataset["test"])

        
        #  Process the train, dev and test set and replace the last three args in __init__() below
        super().__init__(QMsumDataset.dataset_name,
                         QMsumDataset.description,
                         is_dialogue_based=True,
                         is_multi_document=False,
                         is_query_based=True,
                         train_set=processed_train_set,  
                         dev_set=processed_dev_set, 
                         test_set=processed_test_set, 
                         )
        

    @staticmethod
    def process_qmsum_data(data: Dataset) -> List[SummInstance]:
        processed_set = []
        for instance in tqdm(data):
            for query_set in instance['general_query_list'] + instance['specific_query_list']:
                meeting: List = [utterance['speaker'] + " : " + utterance['content']\
                                for utterance in instance['meeting_transcripts']]
                query: str = query_set['query']
                summary: str = query_set['answer']    
                summ_instance = SummInstance(source=meeting, summary=summary, query=query)
                processed_set.append(summ_instance)


        return processed_set




class ArxivDataset(SummDataset):
    """
    The Arxiv Dataset
    """
    
    download_link = 'https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip'
    
    def __init__(self):
    
        print("*****************\n",\
              "***Attention***\n",\
              "This dataset is quite large (approx 5Gb and will need about 15 Gb for the extraction process\n",\
              "Cancel/interrupt the download if size and time constraints will not be met\n",\
              "*****************", sep="")
        
        # download and unzip the dataset in the temp directory
        tmp_dir = tempfile.TemporaryDirectory()

        zip_path = os.path.join(tmp_dir.name, 'arxiv.zip')
        gdown.download(ArxivDataset.download_link, zip_path, quiet=False) 

        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmp_dir.name)

        # Read from individual folders for the article and the summary
        extraction_path = os.path.join(tmp_dir.name)


        # Extract the dataset entries from folders and load into dataset
        processed_train_set = ArxivDataset.process_arxiv_data(os.path.join(extraction_path, 'train.txt')) 
        processed_dev_set = ArxivDataset.process_arxiv_data(os.path.join(extraction_path, 'val.txt')) 
        processed_test_set = ArxivDataset.process_arxiv_data(os.path.join(extraction_path, 'test.txt')) 

        tmp_dir.cleanup()

        
        #  Process the train, dev and test set and replace the last three args in __init__() below
        dataset_name = "Arxiv_longsummarization_dataset"
        description = "A summarization dataset comprised of pairs of scientific papers. \
                        The dataset provides a challenging testbed for abstractive summarization. \
                        It contains papers and their abstracts. "
        super().__init__(dataset_name,
                         description,
                         is_dialogue_based=False,
                         is_multi_document=False,
                         is_query_based=False,
                         train_set=processed_train_set,  
                         dev_set=processed_dev_set, 
                         test_set=processed_test_set, 
                         )

    @staticmethod
    def process_arxiv_data(file_path: str) -> List[SummInstance]:

        entries_set = []

        infile = open(file_path, 'r')
        for line in infile:
            data = json.loads(line)
            entries_set.append(data)
        infile.close()

        processed_set = []
        for instance in entries_set:
            article: List = instance['article_text']
            abstract: str = instance['abstract_text']
            summ_instance = SummInstance(source=article, summary=abstract)
            processed_set.append(summ_instance)

        return processed_set
