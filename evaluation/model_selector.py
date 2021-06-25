import pandas as pd
def model_selector(models,
                   dataset,
                   evaluation_metrics) -> pd.DataFrame:

    table = pd.DataFrame()

    sources = [instance.source for instance in dataset.test_set]
    targets = [instance.summary for instance in dataset.test_set]

    for model in models:
        inputs = model.summarize(sources)
        for metric in evaluation_metrics:
            score_dict = metric.evaluate(sources, targets)
            for key in score_dict:
                table[model.model_name][metric.metric_name] = score_dict[key]

    return table
