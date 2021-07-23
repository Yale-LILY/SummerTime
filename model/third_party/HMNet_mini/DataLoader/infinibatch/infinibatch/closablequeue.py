# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import deque
from threading import Condition, Lock, Thread


class ClosedException(Exception):
    pass


class ClosableQueue:
    """
    A thread-safe queue that can be closed

    As long as the the queue is not closed, it behaves just like a thread-safe queue with a capacity limit:
        - put blocks until the item can be added
        - get blocks until there is an item to be returned

    Once the queue is closed, no more items can be added but existing items can be removed:
        - put always raises a ClosedException
        - get returns an item if the queue is not empty and otherwise raises a ClosedException
    """
    def __init__(self, maxsize: int=1000):
        self._maxsize = maxsize
        self._queue = deque()
        self._mutex = Lock()
        self._not_empty = Condition(self._mutex)
        self._not_full = Condition(self._mutex)
        self._closed = False

    def put(self, item):
        with self._not_full:
            if self._closed:
                raise ClosedException('This queue has been closed, no more items can be added.')
            while len(self._queue) >= self._maxsize:
                self._not_full.wait()
                if self._closed:
                    raise ClosedException('This queue has been closed, no more items can be added.')
            self._queue.append(item)
            self._not_empty.notify()
        
    def get(self):
        with self._not_empty:
            if self._closed and len(self._queue) == 0:
                raise ClosedException('This queue has been closed and is empty, no more items can be retrieved.')
            while len(self._queue) == 0:
                self._not_empty.wait()
                if self._closed and len(self._queue) == 0:
                    raise ClosedException('This queue has been closed and is empty, no more items can be retrieved.')
            item = self._queue.popleft()
            self._not_full.notify()
        return item
            
    def close(self):
        with self._mutex:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()