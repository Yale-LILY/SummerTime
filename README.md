# SummerTime

A library to help users choose appropriate summarization tools based on their specific tasks or needs. Includes models, evaluation metrics, and datasets.

## Installation and setup

Create and activate a new `conda` environment:
```bash
conda create -n st python=3.7
conda activate st
```

`pip` dependencies for local demo:
```bash
pip install -r requirements.txt
```

Start demo Jupyter notebook from localhost:
```bash
jupyter notebook demo.ipynb
```

## Quick Start
Imports model, initializes default model, and summarizes sample documents.
```python
import model as st_model

model = st_model.summarizer()
documents = [
    """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. 
    The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected 
    by the shutoffs which were expected to last through at least midday tomorrow."""
]
model.summarize(documents)

# ["California's largest electricity provider has turned off power to hundreds of thousands of customers."]
```
See `demo.ipynb` demo Jupyter notebook for more examples.

## Models
Import and initialization:
```python
import model as st_model

default_model = std_model.summarizer()
bart_model = std_model.bart_model.BartModel()
pegasus_model = std_model.pegasus_model.PegasusModel()
lexrank_model = std_model.lexrank_model.LexRankModel()
textrank_model = st_model.textrank_model.TextRankModel()
```

All models can be initialized with the following optional options:
```python
def __init__(self,
         trained_domain: str=None,
         max_input_length: int=None,
         max_output_length: int=None,
         ):
```

All models implement the following methods:
```python
def summarize(self,
  corpus: Union[List[str], List[List[str]]],
  queries: List[str]=None) -> List[str]:

def show_capability(cls) -> None:

def generate_basic_description(cls) -> str:
```

## Evaluation
Import and initialization:
```python
import eval as st_eval

bert_eval = st_eval.bertscore()
bleu_eval = st_eval.bleu_eval()
rouge_eval = st_eval.rouge()
rougewe_eval = st_eval.rougewe()
```

All evaluation metrics can be initialized with the following optional arguments:
```python
def __init__(self, metric_name):
```

All evaluation metric objects implement the following methods:
```python
def evaluate(self, model, data):

def get_dict(self, keys):
```


## Datasets
Import and initialization:
```python
import dataset.stdatasets as st_data
```
