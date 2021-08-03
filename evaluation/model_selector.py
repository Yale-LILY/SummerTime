import pandas as pd
import itertools
from .plotutils.radar import make_radar_plot
from typing import List, Tuple

# TODO: figure out how to horizontally import model and data
# for type annotations
def model_selector(models,
                   generator,
                   metrics,
                   max_instances : int = -1) -> pd.DataFrame:
    store_data = {} # dictionary to be converted to pd.Dataframe

    if max_instances == -1:
        tiny_generator = generator
    else:
        tiny_generator = itertools.islice(generator, max_instances)

    tiny_generators = list(itertools.tee(tiny_generator, len(models)*len(metrics)))

    for model in models:
        store_data[model.model_name] = {}

        for metric in metrics:
            # TODO: make default keys a class variable
            get_keys = metric.evaluate(['test'], ['test'])
            # used for averaging metric across examples
            sum_score_dict = {key: 0 for key in get_keys}
            num_instances = 0

            current_generator = tiny_generators.pop()
            for instance in current_generator:
                input = model.summarize([instance.source])
                score_dict = metric.evaluate(input, [instance.summary])
                sum_score_dict = {key: sum_score_dict[key] + score_dict[key] for key in sum_score_dict}
                num_instances += 1

            avg_score_dict = {key: sum_score_dict[key]/num_instances for key in sum_score_dict}

            for key in avg_score_dict:
                store_data[model.model_name][key] = avg_score_dict[key]

    df = pd.DataFrame.from_dict(store_data, orient='index')
    return df

def smart_model_selector(models,
                         generator,
                         metrics,
                         min_instances: int,
                         max_instances: int,
                         factor : int = 3) -> pd.DataFrame:
    total_instances = 0
    # first run with min_instances instances
    num_instances = min_instances
    tiny_generator = itertools.islice(generator, min_instances)
    df = model_selector(models, tiny_generator, metrics)

    models = _remove_bad_model(models, df)

    total_instances += num_instances

    num_instances = num_instances * factor

    while (len(models) > 1) and (total_instances <= max_instances):
        tiny_generator = itertools.islice(generator, num_instances)
        new_df = model_selector(models, tiny_generator, metrics)
        df = _update_df(df, new_df, total_instances, num_instances)

        models = _remove_bad_model(models, new_df)

        total_instances += num_instances
        num_instances = num_instances * factor

    return df


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

# Merges df1 and df2
def _update_df(df1: pd.DataFrame,
               df2: pd.DataFrame,
               total_instances : int,
               num_instances : int) -> pd.DataFrame:
    for i, _ in df2.iterrows():
        for j in df2.keys():
            denom = total_instances + num_instances
            df1.at[i, j] = total_instances / denom * df1.at[i, j] + num_instances / denom * df2.at[i, j]
    return df1

# Removes a model's row from the dataframe if it is worse than every other model
# on every metric

# TODO: figure out how to do horizontal import for type annotations
def _remove_bad_model(models, df : pd.DataFrame):
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
    else:
        return models
