from typing import List, Union


class SummModel:
    """
    Base model class for SummerTime
    """
    def __init__(self,
                 model_name: str,
                 is_extractive: bool,
                 is_neural: bool,
                 is_query_based: bool=False,
                 is_dialogue_based: bool=False,
                 is_multi_document: bool=False,
                 trained_domain: str=None,
                 max_input_length: int=None,
                 max_output_length: int=None,
                 ):
        self.model_name = model_name
        self.is_extractive = is_extractive
        self.is_neural = is_neural
        self.is_query_based = is_query_based
        self.is_dialogue_based = is_dialogue_based
        self.is_multi_document = is_multi_document
        self.trained_domain = trained_domain
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def summarize(self, corpus: Union[List[str], List[List[str]]]) -> List[str]:
        """
        All summarization models should have this function
        
        :param corpus: each string in the list is a source document to be summarized; if the model is multi-document
            summarization model, then each instance contains a list of documents
        :return: a list of generated summaries
        """
        raise NotImplementedError("The base class for models shouldn't be instantiated!")

    def show_capability(self) -> None:
        """
        Use concise language to show the strength and weakness for each model. Try not to use NLP terminologies
        """
        raise NotImplementedError("The base class for models shouldn't be instantiated!")
    
    def generate_basic_description(self) -> str:
        """
        Automatically generate the basic description string based on the attributes
        """
        extractive_abstractive = "extractive" if self.is_extractive else "abstractive"
        neural = "neural" if self.is_neural else "non-neural"
        
        basic_description = f"{self.model_name} is a {extractive_abstractive}, {neural} model for summarization."
        if self.is_multi_document or self.is_dialogue_based:
            basic_description += f"It can handle {'multi-document' if self.is_multi_document else ''} " \
                                 f"{'dialogue' if self.is_dialogue_based else ''} textual data."
        if self.trained_domain:
            basic_description += f"It is trained on {self.trained_domain} data."
        if self.max_input_length:
            basic_description += f"The maximum input length for {self.model_name} is {self.max_output_length}. "
        if self.max_output_length:
            basic_description += f"The maximum output length is {self.max_output_length}. "
            
        return basic_description
    
    @staticmethod
    def list_all_models():
        # TODO ansongn:
        pass
