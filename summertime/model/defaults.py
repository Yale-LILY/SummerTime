from .single_doc import PegasusModel


class summarizer(PegasusModel):
    def __init__(self, device="cpu"):
        super(summarizer, self).__init__(device)

    def show_capability(self):
        print("Pegasus is the default single-document summarization model.")
        super(summarizer, self).show_capability()
