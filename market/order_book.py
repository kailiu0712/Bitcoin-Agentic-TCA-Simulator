import heapq
from collections import defaultdict, deque


class OrderBook:
    """Small price-time-priority LOB. Empty heap entries are removed lazily."""

    def __init__(self):
        self.orders = {}
        self.levels = {"BUY": defaultdict(deque), "SELL": defaultdict(deque)}
        self.heaps = {"BUY": [], "SELL": []}

    def add(self, order):
        if order.is_market:
            raise ValueError("market orders cannot rest")
        if order.order_id in self.orders:
            raise ValueError("duplicate order id")
        self.orders[order.order_id] = order
        level = self.levels[order.side][order.price]
        if not level:
            heapq.heappush(self.heaps[order.side], -order.price if order.side == "BUY" else order.price)
        level.append(order.order_id)

    def cancel(self, order_id):
        order = self.orders.pop(order_id, None)
        if order:
            order.remaining = 0.0
            return True
        return False

    def _clean(self, side):
        heap = self.heaps[side]
        while heap:
            price = -heap[0] if side == "BUY" else heap[0]
            q = self.levels[side][price]
            while q and q[0] not in self.orders:
                q.popleft()
            if q:
                return price
            heapq.heappop(heap)
            self.levels[side].pop(price, None)
        return None

    def best_price(self, side):
        return self._clean(side)

    def best_order(self, side):
        price = self._clean(side)
        return None if price is None else self.orders[self.levels[side][price][0]]

    def depth(self, side, price=None):
        price = self.best_price(side) if price is None else price
        if price is None:
            return 0.0
        return sum(self.orders[i].remaining for i in self.levels[side].get(price, ()) if i in self.orders)

    def crosses(self, order):
        opposite = "SELL" if order.side == "BUY" else "BUY"
        best = self.best_price(opposite)
        if best is None:
            return False
        return order.is_market or (order.side == "BUY" and order.price >= best) or (order.side == "SELL" and order.price <= best)

    def remove_filled(self, order):
        if order.remaining <= 1e-12:
            self.orders.pop(order.order_id, None)

    def state(self):
        bid, ask = self.best_price("BUY"), self.best_price("SELL")
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        return {"best_bid": bid, "best_ask": ask, "bid_size": self.depth("BUY", bid),
                "ask_size": self.depth("SELL", ask), "mid": mid,
                "spread": ask - bid if bid is not None and ask is not None else None}

