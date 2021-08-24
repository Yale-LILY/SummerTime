from transformers import LongformerTokenizer, EncoderDecoderModel
from .base_single_doc_model import SingleDocSummModel


class LongformerModel(SingleDocSummModel):

    # static variables
    model_name = "Longformer"
    is_extractive = False
    is_neural = True

    def __init__(self):
        super(LongformerModel, self).__init__()

        self.model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
        self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    def summarize(self, corpus, queries=None):
        self.assert_summ_input_type(corpus, queries)

        summaries = list(map(lambda doc: self.summarize_single(doc), corpus))

        return summaries
    
    def summarize_single(self, document):
        # Tokenizes document and returns PyTorch torch.Tensor object with length attribute
        tokenized_sequence = self.tokenizer(document, return_tensors="pt", return_length=True, truncation=True, max_length=4096)
        print(f"Longformer model: processing document of {tokenized_sequence.length} tokens")
        input_ids = tokenized_sequence.input_ids
        # output_ids is tensor with one layer: output_ids[0] extracts tensor layer for decoding
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
