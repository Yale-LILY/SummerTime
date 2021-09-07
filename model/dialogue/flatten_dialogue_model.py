from model.base_model import SummModel

class FlattenDialogueModel(SummModel):
    def __init__(self, trained_domain: str, max_input_length: int, max_output_length: int):
        super().__init__(trained_domain=trained_domain, max_input_length=max_input_length, max_output_length=max_output_length)


    