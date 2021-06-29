import tempfile
import zipfile
import os
import json
import random
import gdown
from urllib.request import urlretrieve, urlopen
from typing import Optional, List, Tuple
from dataset.st_dataset import SummInstance, SummDataset


class ScisummnetDataset(SummDataset):
    """
    The SciSummNet dataset. As a dataset not included by huggingface, we need to do manually download, set basic
        information for the dataset
    """
    
    download_link = "https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip"
    
    def __init__(self):
    
        # Download and unzip the dataset in the temp directory
        tmp_dir = tempfile.TemporaryDirectory()
        zip_path, _ = urlretrieve(ScisummnetDataset.download_link)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmp_dir.name)
            
        # Read from individual folders for the article and the summary
        extraction_path = os.path.join(tmp_dir.name, 'scisummnet_release1.1__20190413', 'top1000_complete')

        # Extract the dataset entries from folders and load into dataset
        scisumm_ds = []
        for folder in os.listdir(extraction_path):
            entry = {}
            entry['entry_number'] = folder

            doc_xml_path = os.path.join(extraction_path, folder, 'Documents_xml', folder + ".xml")
            with open (doc_xml_path, "r", encoding='utf-8') as f:
                entry['document_xml'] = f.read()

            cite_annot_path = os.path.join(extraction_path, folder, 'citing_sentences_annotated.json')
            with open (cite_annot_path, "r", encoding='utf-8') as f:
                entry['citing_sentences_annotated.json'] = json.load(f)

            summary_path = os.path.join(extraction_path, folder, 'summary', folder + ".gold.txt")
            with open (summary_path, "r", encoding='utf-8') as f:
                entry['summary'] = f.read()

            # Create a Summarization instance for every unit and add to the dataset
            summ_entry_instance = SummInstance(source = [entry['document_xml'], entry['citing_sentences_annotated.json']],\
                                               summary = entry['summary'])
            scisumm_ds.append(summ_entry_instance)

        tmp_dir.cleanup()


        # Randomize and split data into train, dev and test sets
        random.shuffle(scisumm_ds)
        split_1 = int(0.8 * len(scisumm_ds))
        split_2 = int(0.9 * len(scisumm_ds))

        processed_train_set = scisumm_ds[:split_1]
        processed_dev_set = scisumm_ds[split_1:split_2]
        processed_test_set = scisumm_ds[split_2:]

        
        #  Process the train, dev and test set and replace the last three args in __init__() below
        dataset_name = "ScisummNet_v1.1"
        description = "A summary of scientific papers should ideally incorporate the impact of the papers on the " \
                      "research community reflected by citations. To facilitate research in citation-aware scientific " \
                      "paper summarization (Scisumm), the CL-Scisumm shared task has been organized since 2014 for " \
                      "papers in the computational linguistics and NLP domain."
        super().__init__(dataset_name,
                         description,
                         is_dialogue_based=False,
                         is_multi_document=False,
                         is_query_based=False,
                         train_set=processed_train_set,  
                         dev_set=processed_dev_set, 
                         test_set=processed_test_set, 
                         )
        


class SummscreenDataset(SummDataset):
    """
    The SummScreen dataset. As a dataset not included by huggingface, we need to do manually download, set basic
        information for the dataset
    """
    
    download_link = 'https://drive.google.com/uc?id=1BvdIllGBo9d2-bzXQRzWuJXB04XPVmfF'
    
    def __init__(self):
    
        
        # download and unzip the dataset in the temp directory
        tmp_dir = tempfile.TemporaryDirectory()

        zip_path = os.path.join(tmp_dir.name, 'sumscreen.zip')
        gdown.download(SummscreenDataset.download_link, zip_path, quiet=False) 

        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmp_dir.name)

        # Read from individual folders for the article and the summary
        extraction_path = os.path.join(tmp_dir.name, 'SummScreen')


        # Extract the dataset entries from folders and load into dataset
        processed_train_set = SummscreenDataset.process_summscreen_data(os.path.join(extraction_path, 'ForeverDreaming', 'fd_train.json')) 
        processed_train_set += SummscreenDataset.process_summscreen_data(os.path.join(extraction_path, 'TVMegaSite', 'tms_train.json'))
        processed_dev_set = SummscreenDataset.process_summscreen_data(os.path.join(extraction_path, 'ForeverDreaming', 'fd_dev.json')) 
        processed_dev_set += SummscreenDataset.process_summscreen_data(os.path.join(extraction_path, 'TVMegaSite', 'tms_dev.json'))
        processed_test_set = SummscreenDataset.process_summscreen_data(os.path.join(extraction_path, 'ForeverDreaming', 'fd_test.json')) 
        processed_test_set += SummscreenDataset.process_summscreen_data(os.path.join(extraction_path, 'TVMegaSite', 'tms_test.json'))

        tmp_dir.cleanup()

        
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
    def process_summscreen_data(file_path: str) -> List[SummInstance]:

        entries_set = []

        infile = open(file_path, 'r')
        for line in infile:
            processed_line = line.replace("@@ ", "")
            data = json.loads(processed_line)
            entries_set.append(data)
        infile.close()

        processed_set = []
        for instance in entries_set:
            transcript: List = instance['Transcript']
            recap: str = instance['Recap'][0]               # Recap is a single string in list
            summ_instance = SummInstance(source=transcript, summary=recap)
            processed_set.append(summ_instance)

        return processed_set




class QMsumDataset(SummDataset):
    """
    QMSum is a new human-annotated benchmark for query-based multi-domain meeting summarization task, \
        which consists of 1,808 query-summary pairs over 232 meetings in multiple domains.
    """
    
    download_link = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/"
    
    def __init__(self):

        # Extract the dataset entries from folders and load into dataset
        processed_train_set = QMsumDataset.process_qmsum_data("train")
        processed_dev_set = QMsumDataset.process_qmsum_data("val")
        processed_test_set = QMsumDataset.process_qmsum_data("test")

        
        #  Process the train, dev and test set and replace the last three args in __init__() below
        dataset_name = "QMsum"
        description = "QMSum is a new human-annotated benchmark for query-based multi-domain meeting summarization task, \
                        which consists of 1,808 query-summary pairs over 232 meetings in multiple domains."
        super().__init__(dataset_name,
                         description,
                         is_dialogue_based=True,
                         is_multi_document=False,
                         is_query_based=True,
                         train_set=processed_train_set,  
                         dev_set=processed_dev_set, 
                         test_set=processed_test_set, 
                         )
        

    @staticmethod
    def process_qmsum_data(split: str) -> List[SummInstance]:

        data_path, _ = urlretrieve(QMsumDataset.download_link + split + ".jsonl")
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))

        processed_set = []
        for instance in data:
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
              "*****************")
        
        # download and unzip the dataset in the temp directory
        tmp_dir = tempfile.TemporaryDirectory()

        zip_path = os.path.join(tmp_dir.name, 'arxiv.zip')
        gdown.download(ArxivDataset.download_link, zip_path, quiet=False) 

        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmp_dir.name)

        # Read from individual folders for the article and the summary
        extraction_path = os.path.join(tmp_dir.name, 'Arxiv')


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


ArxivDataset()