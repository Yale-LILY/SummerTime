from transformers import LongformerTokenizer, EncoderDecoderModel
from .base_model import SummModel

class LongformerModel(SummModel):
    
    # static variables
    model_name = "LONGFORMER"
    is_extractive = False
    is_neural = False

    def __init__(self):
        super(LongformerModel, self).__init__()

        self.model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
        self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    def summarize(self, corpus, queries=None):
        input_ids = self.tokenizer(corpus, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids)

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    @classmethod
    def show_capability(cls) -> None:
        basic_description = cls.generate_basic_description()
        more_details = ("A Longformer2Roberta model finetuned on CNN-DM dataset for summarization.\n\n"
                "Strengths:\n - Correctly handles longer (> 2000 tokens) corpus.\n\n"
                "Weaknesses:\n - Less accurate on contexts outside training domain.\n\n"
                "Initialization arguments:\n "
                " - `corpus`: Unlabelled corpus of documents.\n")
        print(f"{basic_description} \n {'#'*20} \n {more_details}")
