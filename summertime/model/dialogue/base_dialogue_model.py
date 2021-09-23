import re

from typing import List, Union
from itertools import chain
from summertime.model.base_model import SummModel


class DialogueSummModel(SummModel):

    is_dialogue_based = True

    def __init__(
        self,
        trained_domain: str = None,
        max_input_length: int = None,
        max_output_length: int = None,
    ):
        super(DialogueSummModel, self).__init__(
            trained_domain=trained_domain,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )

    @classmethod
    def assert_summ_input_type(
        cls, corpus: Union[List[str], List[List[str]]], queries: Union[List[str], None]
    ):
        """each instance must be a list of \"speaker : utterance\" """
        assert all([isinstance(instance, list) for instance in corpus])

        pattern = re.compile(r"\w+\s:\s\w+")
        assert all(
            [pattern.match(instance) for instance in chain.from_iterable(corpus)]
        ), 'each instance must be a list of "[speaker] : [utterance]", the ":" is essential'
