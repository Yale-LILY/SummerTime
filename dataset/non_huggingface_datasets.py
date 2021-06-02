import tempfile
import zipfile
import os
from urllib.request import urlretrieve
from dataset.st_dataset import SummInstance, SummDataset


class Scisummnet(SummDataset):
    """
    The SciSummNet dataset. As a dataset not included by huggingface, we need to do manually download, set basic
        information for the dataset
    """
    
    download_link = "https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip"
    
    def __init__(self):
    
        # download and unzip the dataset in the temp directory
        tmp_dir = tempfile.TemporaryDirectory()
        zip_path, _ = urlretrieve(Scisummnet.download_link)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmp_dir.name)
            
        # TODO Murori: read from individual folders for the article and the summary
        extraction_path = os.path.join(tmp_dir.name, 'scisummnet_release1.1__20190413/Dataset_Documentation.txt')
        
        # TODO Murori: process the train, dev and test set and replace the last three args in __init__() below
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
                         train_set=None,  # TODO
                         dev_set=None,  # TODO
                         test_set=None,  # TODO
                         )
        

if __name__ == '__main__':
    Scisummnet()
