import unittest

from dataset.dataset_loaders import ArxivDataset
from dataset import SUPPORTED_SUMM_DATASETS, list_all_datasets, list_all_datasets_detailed
from dataset.st_dataset import SummDataset, SummInstance

from helpers import print_with_color


class TestDatasets(unittest.TestCase):

    def _test_instance(self, ins: SummInstance, is_query: bool = False, is_multi_document: bool = False, is_dialogue: bool = False):
        if is_multi_document or is_dialogue:
            self.assertTrue(isinstance(ins.source, list))
        else:
            self.assertTrue(isinstance(ins.source, list) or isinstance(ins.source, str))
        if is_query:
            self.assertTrue(isinstance(ins.query, str))

    def test_all_datasets(self):
        print_with_color(f"{'#' * 10} Testing all datasets... {'#' * 10}\n\n", "35")

        print(list_all_datasets())
        print(list_all_datasets_detailed())

        num_datasets = 0

        for ds_cls in SUPPORTED_SUMM_DATASETS:
            # TODO: Temporarily skipping Arxiv (size/time)
            if ds_cls in [ArxivDataset]:
                continue

            print_with_color(f"Testing {ds_cls} dataset...", "35")
            ds: SummDataset = ds_cls()

            ds.show_description()

            # must have at least one of train/dev/test set
            assert ds.train_set or ds.validation_set or ds.test_set

            if ds.train_set is not None:
                train_set = list(ds.train_set)
                print(f"{ds_cls} has a training set of {len(train_set)} examples")
                self._test_instance(train_set[0], is_multi_document=ds.is_multi_document, is_dialogue=ds.is_dialogue_based)

            if ds.validation_set is not None:
                val_set = list(ds.validation_set)
                print(f"{ds_cls} has a validation set of {len(val_set)} examples")
                self._test_instance(val_set[0], is_multi_document=ds.is_multi_document, is_dialogue=ds.is_dialogue_based)

            if ds.test_set is not None:
                test_set = list(ds.test_set)
                print(f"{ds_cls} has a test set of {len(test_set)} examples")
                self._test_instance(test_set[0], is_multi_document=ds.is_multi_document, is_dialogue=ds.is_dialogue_based)
            
            print_with_color(f"{ds.dataset_name} dataset test complete\n", "32")
            num_datasets += 1

        print_with_color(f"{'#' * 10} test_all_datasets {__name__} complete ({num_datasets} datasets) {'#' * 10}", "32")


if __name__ == '__main__':
    unittest.main()
