# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This file causes the doctests to be included as part of unit tests.

To make sure the doctests of a specific module are included,
please replicate the `addTests` call for the iterators module below.
"""

import doctest
import model.third_party.HMNet.DataLoader.infinibatch.infinibatch.iterators


def load_tests(loader, tests, ignore):
    tests.addTests(
        doctest.DocTestSuite(
            model.third_party.HMNet.DataLoader.infinibatch.infinibatch.iterators
        )
    )
    return tests
