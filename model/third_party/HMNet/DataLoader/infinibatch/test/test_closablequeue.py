# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from threading import Thread
import unittest

from model.third_party.HMNet.DataLoader.infinibatch.infinibatch.closablequeue import (
    ClosableQueue,
    ClosedException,
)


class TestClosableQueue(unittest.TestCase):
    def setUp(self):
        self.queue = ClosableQueue(maxsize=10)

    def put_items(self, items, close=False):
        for item in items:
            self.queue.put(item)
        if close:
            self.queue.close()

    def get_items(self, num_items):
        return [self.queue.get() for _ in range(num_items)]

    def test_basic(self):
        self.put_items(range(10))
        self.assertListEqual(self.get_items(10), list(range(10)))

    def test_closed_put(self):
        self.queue.close()
        self.assertRaises(ClosedException, self.queue.put, 42)

    def test_closed_get(self):
        self.put_items(range(10))
        self.queue.close()
        self.assertListEqual(self.get_items(10), list(range(10)))
        self.assertRaises(ClosedException, self.queue.get)

    def test_basic_two_threads(self):
        thread = Thread(target=self.put_items, args=(range(20),))
        thread.start()
        result = self.get_items(20)
        thread.join()
        self.assertListEqual(result, list(range(20)))
