from .pegasus_model import pegasus

class summarizer(pegasus):
    def __init__(self, device='cpu'):
        super(summarizer, self).__init__(device)

    def show_capability(self):
        print('Pegasus is the default singe-document summarization model.')
        super(summarizer, self).show_capability()
