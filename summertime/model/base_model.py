from typing import List, Union


class SummModel:
    """
    Base model class for SummerTime
    """

    # static variables
    model_name = "None"
    is_extractive = False
    is_neural = False
    is_query_based = False
    is_dialogue_based = False
    is_multi_document = False
    is_multilingual = False

    def __init__(
        self,
        trained_domain: str = None,
        max_input_length: int = None,
        max_output_length: int = None,
    ):
        self.trained_domain = trained_domain
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def summarize(
        self, corpus: Union[List[str], List[List[str]]], queries: List[str] = None
    ) -> List[str]:
        """
        All summarization models should have this function

        :param corpus: each string in the list is a source document to be summarized; if the model is multi-document or
            dialogue summarization model, then each instance contains a list of documents/utterances
        :param queries: a list of queries if this is a query-based model
        :return: a list of generated summaries
        """
        raise NotImplementedError(
            "The base class for models shouldn't be instantiated!"
        )

    @classmethod
    def assert_summ_input_type(
        cls, corpus: Union[List[str], List[List[str]]], queries: Union[List[str], None]
    ):
        """
        Verifies that type of input corpus or queries for summarization align with the model type.
        """
        raise NotImplementedError(
            "The base class for models shouldn't be instantiated!"
        )

    @classmethod
    def assert_summ_input_language(
        cls, corpus: Union[List[str], List[List[str]]], queries: Union[List[str], None]
    ) -> str:
        """
        Verifies that language of input corpus and queries for summarization align with the model type.
        Returns the ISO-639 language tag of the input corpus as a string.
        """
        raise NotImplementedError(
            "The base class for models shouldn't be instantiated!"
        )

    @classmethod
    def show_capability(cls) -> None:
        """
        Use concise language to show the strength and weakness for each model. Try not to use NLP terminologies
        """
        raise NotImplementedError(
            "The base class for models shouldn't be instantiated!"
        )

    @classmethod
    def generate_basic_description(cls) -> str:
        """
        Automatically generate the basic description string based on the attributes
        """
        extractive_abstractive = "extractive" if cls.is_extractive else "abstractive"
        neural = "neural" if cls.is_neural else "non-neural"

        basic_description = (
            f"{cls.model_name} is a"
            f"{'query-based' if cls.is_query_based else ''} "
            f"{extractive_abstractive}, {neural} model for summarization."
        )
        if cls.is_multi_document or cls.is_dialogue_based:
            basic_description += (
                f"It can handle {'multi-document' if cls.is_multi_document else ''} "
                f"{'dialogue' if cls.is_dialogue_based else ''} textual data."
            )

        return basic_description

    # TODO nick: implement this function eventually!
    # @classmethod
    # def show_supported_languages(cls) -> str:
    #     """
    #     Returns a list of supported languages for summarization.
    #     """
    #     raise NotImplementedError(
    #         "The base class for models shouldn't be instantiated!"
    #     )


class SummPipeline(SummModel):
    """
    A basic wrapper class for SummModel for handling different aspects of more complicated
    summarization tasks, such as query-based, dialouge, multi-docs, etc
    """

    def __init__(
        self, trained_domain: str, max_input_length: int, max_output_length: int
    ):
        super().__init__(
            trained_domain=trained_domain,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )
