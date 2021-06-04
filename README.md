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


## For contributors
### Pull requests
Create a pull request and name it `[your_gh_username]/[your_branch_name]`. If needed, resolve your own branch's merge conflicts with `main`. Do not push directly to `main`.

### Code formatting
If you haven't already, install `black` and `flake8`:
```bash
pip install black
pip install flake8
```

Before pushing commits or merging branches, run the following commands from the project root:
```bash
black .
flake8 .
```
Or if you would like to run lint only on specific files:
```bash
black path/to/specific/file.py
flake8 path/to/specific/file.py
```
Ensure that `black` reformats all changed files and that `flake8` does not print any warnings. If you would like to override any of the preferences or practices enforced by `black` or `flake8`, please leave a comment in your PR for any lines of code that generate warning or error logs. Do not directly edit linting config files such as `setup.cfg`.

See the [`black` docs](https://black.readthedocs.io/en/stable/index.html) and [`flake8` docs](https://flake8.pycqa.org/en/latest/user/index.html) for documentation on installation and advanced usage. In particular:
- `black [file.py] --diff` to preview changes as diffs instead of directly making changes
- `black [file.py] --check` to preview changes with status codes instead of directly making changes
- `git diff -u | flake8 --diff` to only run `flake8` on working branch changes