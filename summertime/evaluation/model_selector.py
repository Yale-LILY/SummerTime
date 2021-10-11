import math
from functools import reduce
import operator
import itertools
from .plotutils.radar import make_radar_plot
from typing import List, Generator
from prettytable import PrettyTable
from summertime.model.base_model import SummModel
from summertime.dataset.st_dataset import SummInstance
from summertime.evaluation.base_metric import SummMetric


class EvaluationTable(dict):
    def __init__(self, *args, **kw):
        super(EvaluationTable, self).__init__(*args, **kw)

    def __str__(self):
        out = PrettyTable()
        metrics = list(self[list(self.keys())[0]].keys())
        out.field_names = ["Model"] + metrics
        for model_name in self:
            to_add = [model_name] + [self[model_name][metric] for metric in metrics]
            out.add_row(to_add)
        out.float_format = ".3"
        return out.__str__()

    def __repr__(self):
        return self.__str__()


class ModelSelector:
    def __init__(
        self,
        models: List[SummModel],
        generator: Generator[SummInstance, None, None],
        metrics: List[SummMetric],
        max_instances: int = -1,
    ):

        self.models = models

        if max_instances == -1:
            self.generator = generator
        else:
            self.generator = itertools.islice(generator, max_instances)

        self.metrics = metrics

    def run(self) -> EvaluationTable:
        """Evaluates every model on every metric, returning an EvaluationTable"""
        store_data = EvaluationTable()

        tiny_generators = list(
            itertools.tee(self.generator, len(self.models) * len(self.metrics))
        )

        for model in self.models:
            store_data[model.model_name] = {}

            for metric in self.metrics:
                # TODO: make default keys a class variable
                get_keys = metric.evaluate(["test"], ["test"])
                # used for averaging metric across examples
                sum_score_dict = {key: 0 for key in get_keys}
                num_instances = 0

                current_generator = tiny_generators.pop()
                for instance in current_generator:
                    input = model.summarize([instance.source])
                    score_dict = metric.evaluate(input, [instance.summary])
                    sum_score_dict = {
                        key: sum_score_dict[key] + score_dict[key]
                        for key in sum_score_dict
                    }

                    num_instances += 1

                avg_score_dict = {
                    key: sum_score_dict[key] / num_instances for key in sum_score_dict
                }

                for key in avg_score_dict:
                    store_data[model.model_name][key] = avg_score_dict[key]

        return store_data

    def run_halving(self, min_instances: int, factor: int = 3) -> EvaluationTable:
        models = self.models

        total_instances = 0
        # first run with min_instances instances
        num_instances = min_instances
        tiny_generator = itertools.islice(self.generator, min_instances)

        temp_selector = ModelSelector(self.models, tiny_generator, self.metrics)
        table = temp_selector.run()

        models = _remove_bad_model(self.models, table)

        total_instances += num_instances

        num_instances = num_instances * factor

        while len(models) > 1:
            tiny_generator = itertools.islice(self.generator, num_instances)
            temp_selector = ModelSelector(models, tiny_generator, self.metrics)
            new_table = temp_selector.run()
            table = _update_table(table, new_table, total_instances, num_instances)

            models = _remove_bad_model(models, new_table)

            total_instances += num_instances
            num_instances = num_instances * factor

        return table

    def visualize(self, output: EvaluationTable):
        # Preprocesses data.
        data = []
        metrics = list(output[list(output.keys())[0]].keys())
        data.append(metrics)
        rows = []
        row_names = []
        for model in output:
            rows.append([output[model][metric] for metric in metrics])
            row_names.append(model)
        data.append(rows)

        return make_radar_plot(data, row_names)


def _update_table(
    table: EvaluationTable,
    new_table: EvaluationTable,
    total_instances: int,
    num_instances: int,
) -> EvaluationTable:
    """Merges df1 and df2"""
    for model in new_table:
        for metric in new_table[model]:
            denom = total_instances + num_instances
            table[model][metric] = (
                total_instances / denom * table[model][metric]
                + num_instances / denom * new_table[model][metric]
            )
    return table


def _remove_bad_model(models: List[SummModel], table: EvaluationTable):
    """Removes a model's row from the dataframe if it is worse than every other model
    on every metric"""
    name = None
    for model in table:
        cumulative_and = 1
        for other in table:
            cumulative_and *= reduce(operator.mul, 
                [
                    1 if table[model][metric] <= table[other][metric] else 0
                    for metric in table[model]
                ], 1
            )
        if cumulative_and == 1:
            name = model
    if name:
        for i in range(len(models)):
            if models[i].model_name == name:
                models.pop(i)
                return models
    else:
        return models
