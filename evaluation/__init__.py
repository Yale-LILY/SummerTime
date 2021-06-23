from .rouge_metric import Rouge
from .bertscore_metric import BertScore
from .rougewe_metric import RougeWe
from .bleu_metric import Bleu
from .meteor_metric import Meteor

SUPPORTED_EVALUATION_METRICS = [BertScore, Bleu, Rouge, RougeWe, Meteor]
