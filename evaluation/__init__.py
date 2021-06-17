from .rouge_metric import Rouge
from .bertscore_metric import BertScore
from .rougewe_metric import RougeWe
from .bleu_metric import Bleu

SUPPORTED_EVALUATION_METRICS = [BertScore, Bleu, Rouge]
