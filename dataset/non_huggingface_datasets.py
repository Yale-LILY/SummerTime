import os
from tqdm.auto import tqdm
from typing import List, Generator

from datasets import Dataset

from dataset.st_dataset import SummInstance, SummDataset, generate_train_dev_test_splits

# Set directory to load non_huggingface dataset scripts
FILE_DIRECTORY_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_NONHUGGINGFACE_DATASETS_PATH = os.path.join(
    FILE_DIRECTORY_PATH, "non_huggingface_datasets_builders"
)


class ScisummnetDataset(SummDataset):
    """
    The SciSummNet dataset. As a dataset not included by huggingface, we need to do manually download, set basic
        information for the dataset
    """

    dataset_name = "ScisummNet"
    version = "1.1.0"
    description = (
        "A summary of scientific papers should ideally incorporate the impact of the papers on the "
        "research community reflected by citations. To facilitate research in citation-aware scientific "
        "paper summarization (Scisumm), the CL-Scisumm shared task has been organized since 2014 for "
        "papers in the computational linguistics and NLP domain."
    )

    is_dialogue_based = False
    is_multi_document = False
    is_query_based = False

    builder_script_path = os.path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self, seed=None):

        # Load dataset
        scisummnet_dataset = self.load_dataset_safe(
            path=ScisummnetDataset.builder_script_path
        )

        # No dev and test splits provided; hence creating these splits from the train set
        scisummnet_split_dataset = generate_train_dev_test_splits(
            scisummnet_dataset["train"], seed=seed
        )

        processed_train_set = ScisummnetDataset.process_scisummnet_data(
            scisummnet_split_dataset["train"]
        )
        processed_dev_set = ScisummnetDataset.process_scisummnet_data(
            scisummnet_split_dataset["dev"]
        )
        processed_test_set = ScisummnetDataset.process_scisummnet_data(
            scisummnet_split_dataset["test"]
        )

        super().__init__(
            ScisummnetDataset.dataset_name,
            ScisummnetDataset.description,
            train_set=processed_train_set,
            dev_set=processed_dev_set,
            test_set=processed_test_set,
        )

    @staticmethod
    def process_scisummnet_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True):
            docs: List = [
                instance["document_xml"],
                instance["citing_sentences_annotated.json"],
            ]
            summary: str = instance["summary"]
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

    builder_script_path = os.path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self):

        # Load dataset
        summscreen_dataset = self.load_dataset_safe(
            path=SummscreenDataset.builder_script_path
        )

        #  Process the train, dev and test se
        processed_train_set = SummscreenDataset.process_summscreen_data(
            summscreen_dataset["train"]
        )
        processed_dev_set = SummscreenDataset.process_summscreen_data(
            summscreen_dataset["validation"]
        )
        processed_test_set = SummscreenDataset.process_summscreen_data(
            summscreen_dataset["test"]
        )

        #  Process the train, dev and test set and replace the last three args in __init__() below
        dataset_name = "SummScreen_fd+tms_tokenized"
        description = "A summarization dataset comprised of pairs of TV series transcripts and human written recaps. \
                        The dataset provides a challenging testbed for abstractive summarization. \
                        It contains transcripts from FoeverDreaming (fd) and TVMegaSite.\
                        The current version being used is the one where the transcripts have already been tokenized."
        super().__init__(
            dataset_name,
            description,
            train_set=processed_train_set,
            dev_set=processed_dev_set,
            test_set=processed_test_set,
        )

    @staticmethod
    def process_summscreen_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True):
            transcript: List = instance[
                "transcript"
            ]  # convert string into a list of string dialogues
            recap: str = instance["recap"]
            summ_instance = SummInstance(source=transcript, summary=recap)

            yield summ_instance


class QMsumDataset(SummDataset):
    """
    QMSum Dataset
    """

    dataset_name = "QMsum"
    description = """
    QMSum is a new human-annotated benchmark for query-based multi-domain meeting summarization task,
    which consists of 1,808 query-summary pairs over 232 meetings in multiple domains.
    """

    is_dialogue_based = True
    is_multi_document = False
    is_query_based = True

    builder_script_path = os.path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self):

        # Load dataset
        qmsum_dataset = self.load_dataset_safe(path=QMsumDataset.builder_script_path)

        # Extract the dataset entries from folders and load into dataset
        processed_train_set = QMsumDataset.process_qmsum_data(qmsum_dataset["train"])
        processed_dev_set = QMsumDataset.process_qmsum_data(qmsum_dataset["validation"])
        processed_test_set = QMsumDataset.process_qmsum_data(qmsum_dataset["test"])

        #  Process the train, dev and test set and replace the last three args in __init__() below
        super().__init__(
            QMsumDataset.dataset_name,
            QMsumDataset.description,
            train_set=processed_train_set,
            dev_set=processed_dev_set,
            test_set=processed_test_set,
        )

    @staticmethod
    def process_qmsum_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True):
            for query_set in (
                instance["general_query_list"] + instance["specific_query_list"]
            ):
                meeting: List = [
                    utterance["speaker"] + " : " + utterance["content"]
                    for utterance in instance["meeting_transcripts"]
                ]
                query: str = query_set["query"]
                summary: str = query_set["answer"]
                summ_instance = SummInstance(
                    source=meeting, summary=summary, query=query
                )

            yield summ_instance


class ArxivDataset(SummDataset):
    """
    The Arxiv Dataset
    """

    dataset_name = "Arxiv_longsummarization"
    description = """
    A summarization dataset comprised of pairs of scientific papers.
    The dataset provides a challenging testbed for abstractive summarization.
    It contains papers and their abstracts.
    """

    is_dialogue_based = False
    is_multi_document = False
    is_query_based = False

    builder_script_path = os.path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self):

        print(
            "*****************",
            "***Attention***",
            "This dataset is quite large (approx 5Gb and will need about 15 Gb for the extraction process",
            "Cancel/interrupt the download if size and time constraints will not be met",
            "*****************",
            sep="\n",
        )

        # Load dataset
        arxiv_dataset = self.load_dataset_safe(path=ArxivDataset.builder_script_path)

        # Extract the dataset entries from folders and load into dataset
        processed_train_set = ArxivDataset.process_arxiv_data(arxiv_dataset["train"])
        processed_dev_set = ArxivDataset.process_arxiv_data(arxiv_dataset["validation"])
        processed_test_set = ArxivDataset.process_arxiv_data(arxiv_dataset["test"])

        #  Process the train, dev and test set and replace the last three args in __init__() below
        super().__init__(
            ArxivDataset.dataset_name,
            ArxivDataset.description,
            train_set=processed_train_set,
            dev_set=processed_dev_set,
            test_set=processed_test_set,
        )

    @staticmethod
    def process_arxiv_data(data: Dataset) -> Generator[SummInstance, None, None]:
        for instance in tqdm(data, position=0, leave=True):
            article: List = instance["article_text"]
            abstract: str = " ".join(instance["abstract_text"])
            summ_instance = SummInstance(source=article, summary=abstract)

            yield summ_instance
