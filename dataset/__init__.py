from dataset.dataset_loaders import CnndmDataset, MultinewsDataset, SamsumDataset, XsumDataset, PubmedqaDataset, MlsumDataset,\
                                         ScisummnetDataset, SummscreenDataset, QMsumDataset, ArxivDataset


SUPPORTED_SUMM_DATASETS  = [CnndmDataset, MultinewsDataset, SamsumDataset, XsumDataset, PubmedqaDataset, MlsumDataset,\
                            ScisummnetDataset, SummscreenDataset, QMsumDataset, ArxivDataset]


def list_all_datasets():
    all_datasets = []
    for ds in SUPPORTED_SUMM_DATASETS:
        ds_obj = ds()
        all_datasets.append(ds_obj.dataset_name)
    
    return all_datasets


def list_all_datasets_detailed():
    all_dataset_dict = {}
    for ds in SUPPORTED_SUMM_DATASETS:
        ds_obj = ds()
        all_dataset_dict[ds_obj.dataset_name] = ds_obj.description
    
    return all_dataset_dict
