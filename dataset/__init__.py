from dataset.dataset_loaders import (
    CnndmDataset,
    MultinewsDataset,
    SamsumDataset,
    XsumDataset,
    PubmedqaDataset,
    MlsumDataset,
    ScisummnetDataset,
    SummscreenDataset,
    QMsumDataset,
    ArxivDataset,
)


SUPPORTED_SUMM_DATASETS = [
    CnndmDataset,
    MultinewsDataset,
    SamsumDataset,
    XsumDataset,
    PubmedqaDataset,
    MlsumDataset,
    ScisummnetDataset,
    SummscreenDataset,
    QMsumDataset,
    ArxivDataset,
]


def list_all_datasets():
    all_datasets = []
    for ds in SUPPORTED_SUMM_DATASETS:
        dataset_description = ds.generate_basic_description()

        all_datasets.append((ds.dataset_name, dataset_description))

    return all_datasets