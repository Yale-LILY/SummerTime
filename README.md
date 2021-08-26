# SummerTime

A library to help users choose appropriate summarization tools based on their specific tasks or needs. Includes models, evaluation metrics, and datasets.



## Installation and setup

#### Create and activate a new `conda` environment:
```bash
conda create -n st python=3.7
conda activate st
```

#### `pip` dependencies for local demo:
```bash
pip install -r requirements.txt
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

Also, please run `demo.ipynb` demo Jupyter notebook for more examples. To start demo Jupyter notebook on localhost:
```bash
jupyter notebook demo.ipynb
```



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


## Tests
Our continuous integration system is provided through [Travis CI](https://travis-ci.org/). When any pull request is created or updated, the repository's unit tests will be run as build jobs on a Travis server for that pull request. Build jobs will either pass or fail within 5-20 minutes, and build statuses are visible on the Github UI and on Travis CI. Please ensure that the most recent commit in pull requests pass all checks (i.e. all Travis builds run to completion) before merging, or request a review. To skip a Travis build on any particular commit, append `[skip travis]` to the commit message (see [here](https://docs.travis-ci.com/user/customizing-the-build/#skipping-a-build)). Note that PRs with the substring `/no-ci/` anywhere in the branch name will not be built.


## Contributors
This repository is built by the [LILY Lab](https://yale-lily.github.io/) at Yale University, led by Prof. [Dragomir Radev](https://cpsc.yale.edu/people/dragomir-radev). The main contributors are [Ansong Ni](https://niansong1996.github.io), Zhangir Azerbayev, Troy Feng, Murori Mutuma and Yusen Zhang (Penn State). For comments and question, please open an issue.
