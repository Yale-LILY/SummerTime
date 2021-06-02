from dataset.huggingface_datasets import CnndmDataset
from dataset.non_huggingface_datasets import Scisummnet

SUPPORTED_HF_DATASETS = [CnndmDataset]
SUPPORTED_NON_HF_DATASETS = [Scisummnet]
SUPPORTED_SUMM_DATASETS = SUPPORTED_HF_DATASETS + SUPPORTED_NON_HF_DATASETS


def list_all_datasets():
    all_dataset_tuples = []
    for ds in SUPPORTED_SUMM_DATASETS:
        ds_obj = ds()
        all_dataset_tuples.append((ds, ds_obj.description))
    
    return all_dataset_tuples
