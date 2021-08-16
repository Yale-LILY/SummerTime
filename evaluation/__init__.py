import site
import os

# needed so that rouge works
package_path = site.getsitepackages()[0]
os.environ["ROUGE_HOME"] = package_path + '/summ_eval/ROUGE-1.5.5/'

from .rouge_metric import Rouge
from .bertscore_metric import BertScore
from .rougewe_metric import RougeWe
from .bleu_metric import Bleu
from .model_selector import ModelSelector
from .meteor_metric import Meteor

SUPPORTED_EVALUATION_METRICS = [BertScore, Bleu, Rouge, RougeWe, Meteor]
