from os import path
from tqdm import tqdm
from typing import List, Generator, Optional, Union

from datasets import Dataset

from dataset.st_dataset import SummInstance, SummDataset


# Set directory to load non_huggingface dataset scripts
FILE_DIRECTORY_PATH = path.dirname(path.realpath(__file__))
BASE_NONHUGGINGFACE_DATASETS_PATH = path.join(
    FILE_DIRECTORY_PATH, "non_huggingface_datasets_builders"
)


# Huggingface Datasets


class CnndmDataset(SummDataset):
    """
    The CNN/DM dataset
    """

    dataset_name = "CNN/DailyMail"

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/cnn_dailymail"

    def __init__(self, cache_dir: str = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ("cnn_dailymail", "3.0.0",)
        dataset_kwargs = {"cache_dir":cache_dir}
        super().__init__(dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        for instance in tqdm(data):
            article: str = instance["article"]
            highlights: str = instance["highlights"]
            summ_instance = SummInstance(source=article, summary=highlights)

            yield summ_instance


class MultinewsDataset(SummDataset):
    """
    The Multi News dataset
    """

    dataset_name = "Multinews"

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = True

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/multi_news"

    def __init__(self, cache_dir: str = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ("multi_news",)
        dataset_kwargs = {"cache_dir":cache_dir}
        super().__init__(dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        for instance in tqdm(data):
            document: list = [
                doc for doc in instance["document"].split("|||||") if doc
            ]  # removes the empty string generated
            # since each doc ends with the delimiting token '|||||'
            # the final doc creates an empty string
            summary: str = instance["summary"]
            summ_instance = SummInstance(source=document, summary=summary)

            yield summ_instance


class SamsumDataset(SummDataset):
    """
    The SAMsum Dataset
    """

    dataset_name = "Samsum"

    is_query_based = False
    is_dialogue_based = True
    is_multi_document = False

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/samsum"

    def __init__(self, cache_dir: str = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ("samsum",)
        dataset_kwargs = {"cache_dir":cache_dir}
        super().__init__(dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        for instance in tqdm(data):
            dialogue: List = instance["dialogue"].split(
                "\r\n"
            )  # split each dialogue into a list of strings such as
            # ["speaker1 : utter..", "speaker2 : utter..."]
            summary: str = instance["summary"]
            summ_instance = SummInstance(source=dialogue, summary=summary)

            yield summ_instance


class XsumDataset(SummDataset):
    """
    The Xsum Dataset
    """

    dataset_name = "Xsum"

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/xsum"

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False

    def __init__(self, cache_dir: str = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ("xsum",)
        dataset_kwargs = {"cache_dir":cache_dir}
        super().__init__(dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        for instance in tqdm(data):
            document: List = instance["document"]
            summary: str = instance["summary"]
            summ_instance = SummInstance(source=document, summary=summary)

            yield summ_instance


class PubmedqaDataset(SummDataset):
    """
    The Pubmed QA dataset
    """

    dataset_name = "Pubmedqa"

    is_query_based = True
    is_dialogue_based = False
    is_multi_document = False

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/pubmed_qa"

    def __init__(self, cache_dir: str = None, seed: int = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        :param seed: Optional, a seed for the random generator used for making the train and val splits
        """
        dataset_args = ("pubmed_qa", "pqa_artificial",)
        dataset_kwargs = {"cache_dir":cache_dir}
        super().__init__(dataset_args=dataset_args, dataset_kwargs=dataset_kwargs, splitseed=seed)

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
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

    dataset_name = "MlSum"

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/mlsum"
    supported_languages = ["de", "es", "fr", "ru", "tu"]

    mlsum_instantiation_guide = """The languages supported for the Mlsum Dataset are:
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
                """

    def __init__(self, languages: Union[str, List[str]] = "all", cache_dir: str =None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param languages: Optional, a str or a list[str] specifying languages to be included
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ()
        dataset_kwargs = {"cache_dir":cache_dir, "languages":languages}
        super().__init__(dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)

    def _load_dataset_safe(self, args, kwargs):
        """
        Overrides the parent class method
        Method loads multiple datasets of different languages provided in :param languages:
            It then concatenates these datasets into one combined dataset
        :rtype: datasetDict containing the combined dataset
        :param languages: Optional, either a string or list of strings specifying the languages
            to load
        """
        print(MlsumDataset.mlsum_instantiation_guide)

        languages = kwargs["languages"]
        kwargs.pop('languages', None)

        # Choose languages to download articles
        if languages == "all":
            selected_languages = MlsumDataset.supported_languages
        elif isinstance(languages, list):
            for language in languages:
                assert self.is_supported(language)
            selected_languages = languages
        else:
            assert self.is_supported(languages)
            selected_languages = [languages]

        # Concatenate selected languaeges into one dataset
        language_datasets = []
        for language in selected_languages:
            dataset_args = ("mlsum", language, *args)
            dataset_kwargs = kwargs
            dataset = super()._load_dataset_safe(dataset_args, dataset_kwargs)

            language_datasets.append(dataset)

        mlsum_dataset = self._concatenate_dataset_dicts(language_datasets)

        return mlsum_dataset

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        for instance in tqdm(data):
            article: List = instance["text"]
            summary: str = instance["summary"]
            summ_instance = SummInstance(source=article, summary=summary)

            yield summ_instance

    def is_supported(self, language: str):
        """
        Checks whether the requested langues is supported
        :param language: string containing the requested language
        :rtype bool:
        """
        if language not in MlsumDataset.supported_languages:
            print(MlsumDataset.mlsum_instantiation_guide)
            raise ValueError(
                f"The language(s): '{language}' entered is not supported. See above message for usage info"
            )
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
    description = (
        "A summary of scientific papers should ideally incorporate the impact of the papers on the "
        "research community reflected by citations. To facilitate research in citation-aware scientific "
        "paper summarization (Scisumm), the CL-Scisumm shared task has been organized since 2014 for "
        "papers in the computational linguistics and NLP domain."
    )

    is_dialogue_based = False
    is_multi_document = False
    is_query_based = False

    builder_script_path = path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self, cache_dir: str = None, seed: int = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        :param seed: Optional, a seed for the random generator used for making the train and val splits
        """
        dataset_kwargs = {"cache_dir":cache_dir,
                          "path":self.builder_script_path}
        super().__init__(dataset_kwargs=dataset_kwargs, splitseed=seed)

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        for instance in tqdm(data):
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

    builder_script_path = path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self, cache_dir: str = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_kwargs = {"cache_dir":cache_dir,
                          "path":self.builder_script_path}
        super().__init__(dataset_kwargs=dataset_kwargs)

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        for instance in tqdm(data):
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

    builder_script_path = path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self, cache_dir: str = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_kwargs = {"cache_dir":cache_dir,
                          "path":self.builder_script_path}
        super().__init__(dataset_kwargs=dataset_kwargs)

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        for instance in tqdm(data):
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

    builder_script_path = path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self, cache_dir: str = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        print(
            "*****************",
            "***Attention***",
            "This dataset is quite large (approx 5Gb and will need about 15 Gb for the extraction process",
            "Cancel/interrupt the download if size and time constraints will not be met",
            "*****************",
            sep="\n",
        )
        dataset_kwargs = {"cache_dir":cache_dir,
                          "path":self.builder_script_path}
        super().__init__(dataset_kwargs=dataset_kwargs)

    def _process_data(self, data: Dataset) -> Generator[SummInstance, None, None]:
        """
        Overrides the SummDataset '_process_data()' method
        This method processes the data contained in the dataset
            and puts each data instance into a SummInstance object
        :param dataset: a train/validation/test dataset
        :rtype: a generator yielding SummInstance objects
        """
        for instance in tqdm(data):
            article: List = instance["article_text"]
            abstract: str = " ".join(instance["abstract_text"])
            summ_instance = SummInstance(source=article, summary=abstract)

            yield summ_instance
