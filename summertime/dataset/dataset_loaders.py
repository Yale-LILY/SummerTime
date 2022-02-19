from os import path
from tqdm import tqdm
from typing import List, Generator, Optional, Union

from datasets import Dataset

from summertime.dataset.st_dataset import SummInstance, SummDataset


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
    description = "The dataset contains news articles from CNN and Daily Mail. Version 1.0.0 of it was originally developed for reading comprehension and abstractive question answering, then the extractive and abstractive summarization annotations were added in version 2.0.0 and 3.0.0, respectively"

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/cnn_dailymail"

    def __init__(self, cache_dir: Optional[str] = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = (
            "cnn_dailymail",
            "3.0.0",
        )
        dataset_kwargs = {"cache_dir": cache_dir}
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
    description = "This is a large-scale multi-document summarization dataset which contains news articles from the site newser.com with corresponding human-written summaries. Over 1,500 sites, i.e. news sources, appear as source documents, which is higher than the other common news datasets."

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = True

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/multi_news"

    def __init__(self, cache_dir: Optional[str] = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ("multi_news",)
        dataset_kwargs = {"cache_dir": cache_dir}
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
    description = "This  is a dataset with chat dialogues corpus, and human-annotated abstractive  summarizations. In the SAMSum corpus, each dialogue is written by one person. After collecting all the dialogues, experts write a single summary for each dialogue."

    is_query_based = False
    is_dialogue_based = True
    is_multi_document = False

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/samsum"

    def __init__(self, cache_dir: Optional[str] = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ("samsum",)
        dataset_kwargs = {"cache_dir": cache_dir}
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
            # split each dialogue into a list of strings such as ["speaker1 : utter..", "speaker2 : utter..."]
            dialogue: List = instance["dialogue"].split("\r\n")

            summary: str = instance["summary"]
            summ_instance = SummInstance(source=dialogue, summary=summary)
            summ_instance.ensure_dialogue_format()

            yield summ_instance


class XsumDataset(SummDataset):
    """
    The Xsum Dataset
    """

    dataset_name = "Xsum"
    description = "This  is a news summarization dataset for generating a one-sentence summary aiming to answer the question “What is the article about?”. It consists of real-world articles and corresponding one-sentence summarization from British Broadcasting Corporation (BBC)."

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/xsum"

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False

    def __init__(self, cache_dir: Optional[str] = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ("xsum",)
        dataset_kwargs = {"cache_dir": cache_dir}
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
    description = "This is a question answering dataset on the biomedical domain. Every QA instance contains a short answer and a long answer, latter of which can also be used for query-based summarization."

    is_query_based = True
    is_dialogue_based = False
    is_multi_document = False

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/pubmed_qa"

    def __init__(self, cache_dir: Optional[str] = None, seed: int = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        :param seed: Optional, a seed for the random generator used for making the train and val splits
        """
        dataset_args = (
            "pubmed_qa",
            "pqa_artificial",
        )
        dataset_kwargs = {"cache_dir": cache_dir}
        super().__init__(
            dataset_args=dataset_args, dataset_kwargs=dataset_kwargs, splitseed=seed
        )

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
    description = "This is a large-scale multilingual summarization dataset. It contains over 1.5M news articles in five languages, namely French, German, Spanish, Russian, and Turkish."

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False
    is_multilingual = True

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

    def __init__(
        self, languages: Union[str, List[str]] = "all", cache_dir: Optional[str] = None
    ):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param languages: Optional, a str or a list[str] specifying languages to be included
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ()
        dataset_kwargs = {"cache_dir": cache_dir, "languages": languages}
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
        kwargs.pop("languages", None)

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


class XlsumDataset(SummDataset):
    """
    The XLSum dataset - A massively multilingual dataset including 45 languages
    Contains 1.35 million article-summary pairs from BBC in the following languages:

    """

    dataset_name = "XLSum"
    description = "A massively multilingual dataset including 45 languages. Contains 1.35 million article-summary pairs from BBC."

    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False
    is_multilingual = True

    huggingface_dataset = True
    huggingface_page = "https://huggingface.co/datasets/csebuetnlp/xlsum"
    supported_languages = [
        "oromo",
        "french",
        "amharic",
        "arabic",
        "azerbaijani",
        "bengali",
        "burmese",
        "chinese_simplified",
        "chinese_traditional",
        "welsh",
        "english",
        "kirundi",
        "gujarati",
        "hausa",
        "hindi",
        "igbo",
        "indonesian",
        "japanese",
        "korean",
        "kyrgyz",
        "marathi",
        "spanish",
        "scottish_gaelic",
        "nepali",
        "pashto",
        "persian",
        "pidgin",
        "portuguese",
        "punjabi",
        "russian",
        "serbian_cyrillic",
        "serbian_latin",
        "sinhala",
        "somali",
        "swahili",
        "tamil",
        "telugu",
        "thai",
        "tigrinya",
        "turkish",
        "ukrainian",
        "urdu",
        "uzbek",
        "vietnamese",
        "yoruba",
    ]

    instantiation_guide = """The languages supported for the XLSum dataset are:
                ['oromo', 'french', 'amharic', 'arabic', 'azerbaijani',
                'bengali', 'burmese', 'chinese_simplified', 'chinese_traditional', 'welsh',
                'english', 'kirundi', 'gujarati', 'hausa', 'hindi',
                'igbo', 'indonesian', 'japanese', 'korean', 'kyrgyz',
                'marathi', 'spanish', 'scottish_gaelic', 'nepali', 'pashto',
                'persian', 'pidgin', 'portuguese', 'punjabi', 'russian',
                'serbian_cyrillic', 'serbian_latin', 'sinhala', 'somali', 'swahili',
                'tamil', 'telugu', 'thai', 'tigrinya', 'turkish',
                'ukrainian', 'urdu', 'uzbek', 'vietnamese', 'yoruba']

                Instantiate either with "all" to use all languages or a list of languages to use.
                Examples:

                    xlsum_dataset = XlsumDataset("all")
                    xlsum_dataset = XlsumDataset(["english", "french"])
                    xlsum_dataset = XlsumDataset("english")
                """

    def __init__(
        self, languages: Union[str, List[str]], cache_dir: Optional[str] = None
    ):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param languages: Optional, a str or a list[str] specifying languages to be included
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ()
        dataset_kwargs = {"cache_dir": cache_dir, "languages": languages}
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
        print(self.instantiation_guide)

        languages = kwargs["languages"]
        kwargs.pop("languages", None)

        # Choose languages to load
        if languages == "all":
            selected_languages = self.supported_languages
        elif isinstance(languages, list):
            for language in languages:
                assert self.is_supported(language)
            selected_languages = languages
        else:
            assert self.is_supported(languages)
            selected_languages = [languages]

        # Concatenate selected languages into one dataset
        language_datasets = []
        for language in selected_languages:
            dataset_args = ("csebuetnlp/xlsum", language)
            dataset_kwargs = kwargs
            dataset = super()._load_dataset_safe(dataset_args, dataset_kwargs)

            language_datasets.append(dataset)

        xlsum_dataset = self._concatenate_dataset_dicts(language_datasets)

        return xlsum_dataset

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
        Checks whether the requested language is supported
        :param language: string containing the requested language
        :rtype bool:
        """
        if language not in self.supported_languages:
            print(self.instantiation_guide)
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
    description = "This is a human-annotated dataset made for citation-aware scientific paper summarization (Scisumm). It contains over 1,000 papers in the ACL anthology network as well as their citation networks and their manually labeled summaries."

    is_dialogue_based = False
    is_multi_document = False
    is_query_based = False

    builder_script_path = path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self, cache_dir: Optional[str] = None, seed: int = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        :param seed: Optional, a seed for the random generator used for making the train and val splits
        """
        dataset_kwargs = {"cache_dir": cache_dir, "path": self.builder_script_path}
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
    description = "This dataset  consists of community contributed transcripts of television show episodes from The TVMegaSite, Inc. (TMS) and ForeverDream (FD). The summary of each transcript is the recap from TMS, or a recap of the FD shows from Wikipedia and TVMaze"

    version = "1.1.0"
    is_dialogue_based = True
    is_multi_document = False
    is_query_based = False

    builder_script_path = path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self, cache_dir: Optional[str] = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_kwargs = {"cache_dir": cache_dir, "path": self.builder_script_path}
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

    def __init__(self, cache_dir: Optional[str] = None):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_kwargs = {"cache_dir": cache_dir, "path": self.builder_script_path}
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
    description = "This dataset is extracted from research papers for abstractive summarization of single, longer-form documents. For each research paper from arxiv.org, its abstract is used as ground-truth summaries."

    is_dialogue_based = False
    is_multi_document = False
    is_query_based = False

    builder_script_path = path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    def __init__(self, cache_dir: Optional[str] = None):
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
        dataset_kwargs = {"cache_dir": cache_dir, "path": self.builder_script_path}
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


class MassivesummDataset(SummDataset):
    """
    The Massivesumm Dataset
    """

    dataset_name = "Massivesumm"
    description = "This dataset is composed of articles and their summaries crawled from over 300 news platforms online. Includes data from over 12 million articles spread across 92 languages."

    is_dialogue_based = False
    is_multi_document = False
    is_query_based = False
    is_multilingual = True

    huggingface_dataset = False

    builder_script_path = path.join(
        BASE_NONHUGGINGFACE_DATASETS_PATH, dataset_name.lower() + ".py"
    )

    instantiation_guide = """The languages supported for the MassiveSumm dataset are:
                ["afrikaans", "amharic", "arabic", "assamese", "aymara",
                "azerbaijani", "bambara", "bengali", "tibetan", "bosnian",
                "bulgarian", "catalan", "czech", "welsh", "danish", "german",
                "greek", "english", "esperanto", "persian", "filipino", "french",
                "fulah", "irish", "gujarati", "haitian", "hausa", "hebrew",
                "hindi", "croatian", "hungarian", "armenian","igbo", "indonesian",
                "icelandic", "italian", "japanese", "kannada", "georgian", "khmer",
                "kinyarwanda", "kyrgyz", "korean", "kurdish", "lao", "latvian",
                "lingala", "lithuanian", "malayalam", "marathi", "macedonian",
                "malagasy", "mongolian", "burmese", "south ndebele", "nepali",
                "dutch", "oriya", "oromo", "punjabi", "polish", "portuguese",
                "dari", "pashto", "romanian", "rundi", "russian", "sinhala",
                "slovak", "slovenian", "shona", "somali", "spanish", "albanian",
                "serbian", "swahili", "swedish", "tamil", "telugu", "tetum",
                "tajik", "thai", "tigrinya", "turkish", "ukrainian", "urdu",
                "uzbek", "vietnamese", "xhosa", "yoruba", "yue chinese",
                "chinese", "bislama", "gaelic"]

                Currently only supports initialization for a single language.
                The following languages are not currently working, but will be ready soon:
                ["afrikaans", "assamese", "aymara", "catalan", "esperanto", "filipino", "hebrew",
                "croatian", "kinyarwanda", "dutch", "romanian", "swedish", "tetum", "tajik"]
                """
    supported_languages = [
        # "afrikaans",
        "amharic",
        "arabic",
        # "assamese",
        # "aymara",
        "azerbaijani",
        "bambara",
        "bengali",
        "tibetan",
        "bosnian",
        "bulgarian",
        # "catalan",
        "czech",
        "welsh",
        "danish",
        "german",
        "greek",
        "english",
        # "esperanto",
        "persian",
        # "filipino",
        "french",
        "fulah",
        "irish",
        "gujarati",
        "haitian",
        "hausa",
        # "hebrew",
        "hindi",
        # "croatian",
        "hungarian",
        "armenian",
        "igbo",
        "indonesian",
        "icelandic",
        "italian",
        "japanese",
        "kannada",
        "georgian",
        "khmer",
        # "kinyarwanda",
        "kyrgyz",
        "korean",
        "kurdish",
        "lao",
        "latvian",
        "lingala",
        "lithuanian",
        "malayalam",
        "marathi",
        "macedonian",
        "malagasy",
        "mongolian",
        "burmese",
        "south ndebele",
        "nepali",
        # "dutch",
        "oriya",
        "oromo",
        "punjabi",
        "polish",
        "portuguese",
        "dari",
        "pashto",
        # "romanian",
        "rundi",
        "russian",
        "sinhala",
        "slovak",
        "slovenian",
        "shona",
        "somali",
        "spanish",
        "albanian",
        "serbian",
        "swahili",
        # "swedish",
        "tamil",
        "telugu",
        # "tetum",
        # "tajik",
        "thai",
        "tigrinya",
        "turkish",
        "ukrainian",
        "urdu",
        "uzbek",
        "vietnamese",
        "xhosa",
        "yoruba",
        "yue chinese",
        "chinese",
        "bislama",
        "gaelic",
    ]

    def __init__(
        self, languages: Union[str, List[str]], cache_dir: Optional[str] = None
    ):
        """Create dataset information from the huggingface Dataset class
        :rtype: object
        :param languages: Optional, a str or a list[str] specifying languages to be included
        :param cache_dir: Optional, a str specifying where to download/load the dataset to/from
        """
        dataset_args = ()
        dataset_kwargs = {
            "name": languages,
            "cache_dir": cache_dir,
            "path": self.builder_script_path,
        }

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
            summary: str = instance["summary"]
            summ_instance = SummInstance(source=article, summary=summary)

            yield summ_instance
