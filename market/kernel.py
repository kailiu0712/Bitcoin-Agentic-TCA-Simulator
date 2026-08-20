import heapq
import itertools


class Kernel:
    def __init__(self, exchange, seed=0):
        self.exchange = exchange
        self.time = 0.0
        self._queue = []
        self._counter = itertools.count()
        self.seed = seed

    def schedule(self, timestamp, callback):
        if timestamp < self.time:
            raise ValueError("event time cannot move backward")
        heapq.heappush(self._queue, (timestamp, next(self._counter), callback))

    def run(self, until):
        while self._queue and self._queue[0][0] <= until:
            timestamp, _, callback = heapq.heappop(self._queue)
            self.time = timestamp
            callback(self)
        self.time = until

