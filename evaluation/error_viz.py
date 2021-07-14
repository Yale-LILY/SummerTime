import matplotlib.pyplot as plt

def scatter(models,
            generator,
            lexical_metric,
            lexical_key : str,
            semantic_metric,
            semantic_key : str ,
            max_instances : int = -1):
            
    tiny_generator = itertools.islice(generator, max_instances)

    model0_lexical = []
    model1_lexical = []
    model1_semantic = []
    model2_semantic = []
    for instance in tiny_generator:
        model0_summ = model[0].summarize([instance.source])
        model1_summ = model[1].summarize([instance.source])

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

    ax.scatter(pegasus_meteors, pegasus_bertscores, label='Pegasus')
    ax.scatter(bart_meteors, bart_bertscores, label='BART')
    ax.legend(loc=(1.2, .5))

    plt.xlabel('Lexical (Meteor)', fontsize=12, color='grey')
    plt.ylabel('Semantic (BERTScore)', fontsize=12, color='grey')

    ax.text(-.3, -.2, 'Hallucination', fontsize=15)
    ax.text(-.3, 1.2, 'Abstraction', fontsize=15)
    ax.text(1, 1.2, "Extraction", fontsize=15)
    ax.text(1, -.2, 'Misinterpretation', fontsize=15)

    plt.savefig('scatter.pdf', bbox_inches='tight')

    plt.show()
