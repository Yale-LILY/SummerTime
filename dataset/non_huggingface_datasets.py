import tempfile
import zipfile
import os
import json
import random
from urllib.request import urlretrieve
from dataset.st_dataset import SummInstance, SummDataset


class ScisummnetDataset(SummDataset):
    """
    The SciSummNet dataset. As a dataset not included by huggingface, we need to do manually download, set basic
        information for the dataset
    """

    is_dialogue_based = False
    is_multi_document = False
    is_query_based = False
    download_link = "https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip"
    
    def __init__(self):
    
        # download and unzip the dataset in the temp directory
        tmp_dir = tempfile.TemporaryDirectory()
        zip_path, _ = urlretrieve(ScisummnetDataset.download_link)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmp_dir.name)
            
        # TODO Murori: read from individual folders for the article and the summary
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

            # scisumm_ds.append(entry)    # add as individual items instead of a Summarization instance


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
                         train_set=processed_train_set,  
                         dev_set=processed_dev_set, 
                         test_set=processed_test_set, 
                         )
        

if __name__ == '__main__':
    print(ScisummnetDataset().train_set[:20])
