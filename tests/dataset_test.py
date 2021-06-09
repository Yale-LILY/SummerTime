import unittest
from typing import Tuple, List, Optional

from dataset.huggingface_datasets import CnndmDataset
from dataset import SUPPORTED_SUMM_DATASETS, list_all_datasets
from dataset.st_dataset import SummDataset, SummInstance


class TestDatasets(unittest.TestCase):
    
    def _test_instance(self, ins: SummInstance, is_query: bool = False):
        self.assertTrue(isinstance(ins.source, list) or isinstance(ins.source, str))
        self.assertTrue(isinstance(ins.summary, str))
        if is_query:
            self.assertTrue(isinstance(ins.query, str))

    def test_all_datasets(self):
        print(f"{'#' * 10} test_all_datasets STARTS {'#' * 10}")
        
        for ds_cls in SUPPORTED_SUMM_DATASETS:
            print(f"Testing on the {ds_cls} dataset...")
            ds: SummDataset = ds_cls()
            
            # must have at least one of train/dev/test set
            assert ds.train_set or ds.dev_set or ds.test_set
            
            if ds.train_set is not None:
                train_set = list(ds.train_set)
                print(f"{ds_cls} has a training set of {len(train_set)} examples")
                self._test_instance(train_set[0])
                
            if ds.dev_set is not None:
                dev_set = list(ds.dev_set)
                print(f"{ds_cls} has a training set of {len(dev_set)} examples")
                self._test_instance(dev_set[0])
                
            if ds.test_set is not None:
                test_set = list(ds.test_set)
                print(f"{ds_cls} has a training set of {len(test_set)} examples")
                self._test_instance(test_set[0])
        print(f"{'#' * 10} test_all_models {__name__} ENDS {'#' * 10}")


if __name__ == '__main__':
    unittest.main()
