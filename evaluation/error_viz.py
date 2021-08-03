import matplotlib.pyplot as plt
import itertools
from typing import Tuple, Generator

def scatter(models: Tuple,
            generator: Generator,
            metrics: Tuple,
            keys: Tuple,
            max_instances : int = -1):

    lexical_metric = metrics[0]
    semantic_metric = metrics[1]
    lexical_key = keys[0]
    semantic_key = keys[1]

    tiny_generator = itertools.islice(generator, max_instances)

    model0_lexical = []
    model1_lexical = []
    model0_semantic = []
    model1_semantic = []
    for instance in tiny_generator:
        model0_summ = models[0].summarize([instance.source])
        model1_summ = models[1].summarize([instance.source])

        summary = [instance.summary]

        model0_lexical.append(lexical_metric.evaluate(model0_summ, summary)[lexical_key])

        model1_lexical.append(lexical_metric.evaluate(model1_summ, summary)[lexical_key])

        model0_semantic.append(semantic_metric.evaluate(model0_summ, summary)[semantic_key])

        model1_semantic.append(semantic_metric.evaluate(model1_summ, summary)[semantic_key])

    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.scatter(model0_lexical, model0_semantic, label=models[0].model_name)
    ax.scatter(model1_lexical, model1_semantic, label=models[1].model_name)
    ax.legend(loc=(1.2, .5))

    plt.xlabel('Lexical ({})'.format(lexical_metric.metric_name), fontsize=12, color='grey')
    plt.ylabel('Semantic ({})'.format(semantic_metric.metric_name), fontsize=12, color='grey')

    ax.text(-.3, -.2, 'Hallucination', fontsize=15)
    ax.text(-.3, 1.2, 'Abstraction', fontsize=15)
    ax.text(1, 1.2, "Extraction", fontsize=15)
    ax.text(1, -.2, 'Misinterpretation', fontsize=15)

    plt.savefig('scatter.pdf', bbox_inches='tight')

    plt.show()
