import pandas as pd
import itertools
import sys
from .plotutils.radar import make_radar_plot

def model_selector(models,
                   generator,
                   evaluation_metrics,
                   max_instances = -1) -> pd.DataFrame:
    store_data = {}

    if max_instances == -1:
        tiny_generator = generator
    else:
        tiny_generator = itertools.islice(generator, max_instances)

    tiny_generators = itertools.tee(tiny_generator, len(models)*len(evaluation_metrics))

    num_passes = 0 # index for tiny_generators

    for model in models:
        store_data[model.model_name] = {}
        for metric in evaluation_metrics:

            test_score_dict = metric.evaluate(['test'], ['test'])
            acc_score_dict = {key: 0 for key in test_score_dict}
            length = 0

            for instance in tiny_generators[num_passes]:
                input = model.summarize([instance.source])
                score_dict = metric.evaluate(input, [instance.summary])
                acc_score_dict = {key: acc_score_dict[key] + score_dict[key] for key in acc_score_dict}
                length += 1
            #print(model.model_name, metric.metric_name, length)
            total_score_dict = {key: acc_score_dict[key]/length for key in acc_score_dict}

            for key in total_score_dict:
                store_data[model.model_name][key] = total_score_dict[key]

            num_passes += 1

    return pd.DataFrame.from_dict(store_data, orient='index')

def smart_model_selector(models,
                         generator,
                         evaluation_metrics,
                         min_instances,
                         max_instances,
                         instance_factor = 3) -> pd.DataFrame:
    # First run with min_instances
    total_instances = 0
    num_instances = min_instances
    min_resource_generator = itertools.islice(generator, min_instances)
    df = model_selector(models, min_resource_generator, evaluation_metrics)

    models = _remove_bad_model(models, df)

    total_instances += num_instances
    num_instances = num_instances * instance_factor

    while (len(models) > 1) and (total_instances <= max_instances):
        print("doing while loop")
        tiny_generator = itertools.islice(generator, num_instances)

        new_df = model_selector(models, tiny_generator, evaluation_metrics, max_instances=num_instances)
        df = _update_df(df, new_df, total_instances, num_instances)

        models = _remove_bad_model(models, new_df)

        total_instances += num_instances
        num_instances = num_instances * instance_factor


    return df

def _update_df(df1, df2, total_instances, num_instances):
    for i, _ in df2.iterrows():
        for j in df2.keys():
            denom = total_instances + num_instances
            df1.at[i, j] = total_instances / denom * df1.at[i, j] + num_instances / denom * df2.at[i, j]
    return df1

"""This implementation is especially horrible. But it works"""
def _remove_bad_model(models, df):
    name = None
    for i, row1 in df.iterrows():
        cumulative_and = 1
        for j, row2 in df.iterrows():
            cumulative_and *= (row1 <= row2).prod()
        if cumulative_and == 1:
            name = i
    if name:
        for i in range(len(models)):
            if models[i].model_name == name:
                models.pop(i)
                return models
        #return models
    else:
        return models

def visualize_model_selector(output: pd.DataFrame):
    # Preprocesses data.
    data = []
    data.append(list(output.keys()))
    rows = []
    row_names = []
    for i, row in output.iterrows():
        rows.append(list(row))
        row_names.append(i)
    data.append(rows)

    return make_radar_plot(data, row_names)
