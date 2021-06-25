import pandas as pd
import itertools
def model_selector(models,
                   generator,
                   evaluation_metrics,
                   max_instances) -> pd.DataFrame:
    store_data = []

    for model in models:
        row_data = {"model name": model.model_name}
        for metric in evaluation_metrics:
            tiny_generator = itertools.islice(generator, max_instances)

            test_score_dict = metric.evaluate(['test'], ['test'])
            acc_score_dict = {key: 0 for key in test_score_dict}
            length = 0

            for instance in tiny_generator:
                input = model.summarize([instance.source])
                score_dict = metric.evaluate(input, [instance.summary])
                acc_score_dict = {key: acc_score_dict[key] + score_dict[key] for key in acc_score_dict}
                length += 1
            print(model.model_name, metric.metric_name, length)
            total_score_dict = {key: acc_score_dict[key]/length for key in acc_score_dict}

            for key in total_score_dict:
                #table[model.model_name][metric.metric_name] = total_score_dict[key]
                row_data[key] = total_score_dict[key]
        store_data.append(row_data)

    return pd.DataFrame(store_data)
