# SummerTime - A summarization library

A library to help users choose appropriate summarization tools based on their specific tasks or needs. Includes models, evaluation metrics, and datasets.

Check out our midway showcase notebook on Colab to give SummerTime a try        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yale-LILY/SummerTime/blob/murori/midway-notebook/SummerTime_midway_showcase.ipynb)



## Installation and setup

#### Create and activate a new `conda` environment:
```bash
conda create -n summertime python=3.7
conda activate summertime
```

#### `pip` dependencies for local demo:
```bash
pip install -r requirements.txt
```
#### Setup ROUGE
```bash
!export ROUGE_HOME=/usr/local/lib/python3.7/dist-packages/summ_eval/ROUGE-1.5.5/
!pip install -U  git+https://github.com/bheinzerling/pyrouge.git
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

Also, please run our colab notebook for a more hands-on demo and more examples. 


Check out our midway showcase notebook on Colab to give SummerTime a try        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yale-LILY/SummerTime/blob/murori/midway-notebook/SummerTime_midway_showcase.ipynb)




## Models
### Supported Models
SummerTime supports different models (e.g., TextRank, BART, Longformer) as well as model wrappers for more complex summariztion tasks (e.g., JointModel for multi-doc summarzation, BM25 retrieval for query-based summarization).

```python
# Single doc models - BartModel, LexRankModel, LongformerModel, PegasusModel, TextRankModel
# Multi doc models - MultiDocJointModel, MultiDocSeparateModel
# Dialogue model - HMNetModel
# Query based model - TFIDFSummModel, BM25SummModel

# Default model - summarizer
```

To see all supported models, run:

from model import SUPPORTED_SUMM_MODELS
```python
print(SUPPORTED_SUMM_MODELS)
```

### Import and initialization:
```python
import model as st_model

# To use a default model
default_model = st_model.summarizer()    

# Or a specific model
bart_model = st_model.BartModel()
pegasus_model = st_model.PegasusModel()
lexrank_model = st_model.LexRankModel()
textrank_model = st_model.TextRankModel()
```

Users can easily access documentation to assist with model selection
```python
sample_model.show_capability()
pegasus_model.show_capability()
textrank_model.show_capability()
```

To use a model for summarization, simply run:
```python
documents = [
    """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. 
    The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected 
    by the shutoffs which were expected to last through at least midday tomorrow."""
]

sample_model.summarize(documents)
# or 
pegasus_model.summarize(documents)
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
```


## Datasets
### Datasets supported
SummerTime supports different summarization datasets across different domains (e.g., CNNDM dataset - news article corpus, Samsum - dialogue corpus, QM-Sum - query-based dialogue corpus, MultiNews - multi-document corpus, ML-sum - multi-lingual corpus, PubMedQa - Medical domain, Arxiv - Science papers domain, among others.


```python
News article summary datasets - CNNDM corpus, XSum corpus 
Dialogue-based dataset - Samsum corpus, QM-Sum - 
Query-based dialogue corpus, MultiNews - 
multi-document corpus, ML-sum - 
multi-lingual corpus, PubMedQa - 
Medical domain, Arxiv - 
Science papers
```
| Datasets      | Single doc    | Multi doc |  Dialogue-based |  Query-based |  Multi-lingual  | News articles | Scientific papers  |   Medical papers |   TV transcripts   | Meetings domain    |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |



### Import and initialization:
```python
import dataset.stdatasets as st_data
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



## Contributors
This repository is built by the [LILY Lab](https://yale-lily.github.io/) at Yale University, led by Prof. [Dragomir Radev](https://cpsc.yale.edu/people/dragomir-radev). The main contributors are [Ansong Ni](https://niansong1996.github.io), Zhangir Azerbayev, Troy Feng, Murori Mutuma and Yusen Zhang (Penn State). For comments and question, please open an issue.
