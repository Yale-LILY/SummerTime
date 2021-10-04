import unittest

from dataset import SUPPORTED_SUMM_DATASETS, list_all_datasets
from dataset.st_dataset import SummDataset, SummInstance, CustomDataset
from dataset.dataset_loaders import ArxivDataset

from helpers import print_with_color


NUM_DUMMY_DATA_INSTANCES = 10

DUMMY_DATA_WITH_QUERY = [
    # Source doc
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. \
     The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected \
     by the shutoffs which were expected to last through at least midday tomorrow.",
    # Summary
    "California's largest electricity provider has turned off power to hundreds of thousands of customers.",
    # Query
    "What is the main topic of this?",
]

DUMMY_DATA_WITHOUT_QUERY = [
    # Source doc
    "Alice : I am a girl. \
     Bob : I am a boy.",
    # Summary
    "Alice and Bob say who they are",
]


def get_dummy_data_with_query(n: int):
    return [DUMMY_DATA_WITH_QUERY] * n


def get_dummy_data_without_query(n: int):
    return [DUMMY_DATA_WITHOUT_QUERY] * n


class TestDatasets(unittest.TestCase):
    def _test_instance(
        self,
        ins: SummInstance,
        is_query: bool = False,
        is_multi_document: bool = False,
        is_dialogue: bool = False,
    ):
        if is_multi_document or is_dialogue:
            self.assertTrue(isinstance(ins.source, list))
        else:
            self.assertTrue(isinstance(ins.source, list) or isinstance(ins.source, str))
        if is_query:
            self.assertTrue(isinstance(ins.query, str))

    def test_all_datasets(self):

        # Test custom dataset
        print_with_color(f"{'#' * 10} Loading custom dataset... {'#' * 10}\n\n", "35")

        train_set = [
            {
                "source": instance[0],
                "summary": instance[1],
                "query": instance[2],
            }
            for instance in get_dummy_data_with_query(NUM_DUMMY_DATA_INSTANCES)
        ]
        validation_set = [
            {
                "source": instance[0],
                "summary": None,
                "query": instance[2],
            }
            for instance in get_dummy_data_with_query(NUM_DUMMY_DATA_INSTANCES)
        ]
        test_set = [
            {
                "source": instance[0],
                "summary": instance[1],
                "query": instance[2],
            }
            for instance in get_dummy_data_with_query(NUM_DUMMY_DATA_INSTANCES)
        ]

        custom_dataset = CustomDataset(
            train_set=train_set,
            validation_set=validation_set,
            test_set=test_set,
            query_based=True,
            multi_doc=False,
        )

        test_datasets = [custom_dataset] + SUPPORTED_SUMM_DATASETS

        # Test pre-loaded SummerTime datasets
        print_with_color(f"{'#' * 10} Testing all datasets... {'#' * 10}\n\n", "35")
        print(list_all_datasets())

        num_datasets = 0

        for ds_cls in test_datasets:

            # TODO: Temporarily skipping Arxiv (size/time), > 30min download time for Travis-CI
            if ds_cls in [ArxivDataset]:
                continue
            elif isinstance(ds_cls, CustomDataset):
                ds = ds_cls
            else:
                print_with_color(f"Testing {ds_cls} dataset...", "35")
                ds: SummDataset = ds_cls()

                ds.show_description()

            # must have at least one of train/dev/test set
            assert ds.train_set or ds.validation_set or ds.test_set

            if ds.train_set is not None:
                train_set = list(ds.train_set)
                print(f"{ds_cls} has a training set of {len(train_set)} examples")
                self._test_instance(
                    train_set[0],
                    is_multi_document=ds.is_multi_document,
                    is_dialogue=ds.is_dialogue_based,
                )

            if ds.validation_set is not None:
                val_set = list(ds.validation_set)
                print(f"{ds_cls} has a validation set of {len(val_set)} examples")
                self._test_instance(
                    val_set[0],
                    is_multi_document=ds.is_multi_document,
                    is_dialogue=ds.is_dialogue_based,
                )

            if ds.test_set is not None:
                test_set = list(ds.test_set)
                print(f"{ds_cls} has a test set of {len(test_set)} examples")
                self._test_instance(
                    test_set[0],
                    is_multi_document=ds.is_multi_document,
                    is_dialogue=ds.is_dialogue_based,
                )

            print_with_color(f"{ds.dataset_name} dataset test complete\n", "32")
            num_datasets += 1

        print_with_color(
            f"{'#' * 10} test_all_datasets {__name__} complete ({num_datasets} datasets) {'#' * 10}",
            "32",
        )


if __name__ == "__main__":
    unittest.main()
