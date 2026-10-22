import datetime as dt
import itertools
import math

from exchange import BookView, DoneMsg, FillMsg, MarketOrderMsg, OrderRejectedMsg, Side
from kernel import AgentId, Kernel, Oracle


class SmartTaker:
    """
    Oracle-vs-book arbitrageur. Polls every `poll_interval` (it sees the oracle in
    real time).

        gap = (ref - mid) / tick
        gap >=  threshold_ticks -> buy  (up to max_inventory)
        gap <= -threshold_ticks -> sell (down to -max_inventory)
        otherwise unwind: short and gap >= 0 -> buy, long and gap <= 0 -> sell

    The order sweeps only levels strictly on the profitable side of ref, capped by
    max_order_size and the inventory room left.
    """

    def __init__(
        self,
        my_id: AgentId,
        exchange_id: AgentId,
        kernel: Kernel,
        book: BookView,
        oracle: Oracle,
        *,
        threshold_ticks: float,
        max_order_size: float,
        max_inventory: float,
        poll_interval: dt.timedelta = dt.timedelta(milliseconds=100),
        book_depth: int = 10,
    ):
        self.my_id = my_id
        self.exchange_id = exchange_id
        self.kernel = kernel
        self.book = book
        self.oracle = oracle
        self.threshold_ticks = threshold_ticks
        self.max_order_size = max_order_size
        self.max_inventory = max_inventory
        self.poll_interval = poll_interval
        self.book_depth = book_depth

        self.inventory = 0.0
        self.cash = 0.0
        self._coid = itertools.count(1)
        self._in_flight = False  # one market order at a time

    # protocol

    def kernel_start(self, start_time: dt.datetime) -> None:
        self.kernel.set_wakeup(self.my_id, start_time + self.poll_interval)

    def kernel_stop(self, stop_time: dt.datetime) -> None:
        ref = self.oracle.get_price(stop_time)
        print(
            f"[SmartTaker] inventory={self.inventory:.2f} cash={self.cash:.2f} "
            f"mtm_pnl={self.cash + self.inventory * ref:.2f}"
        )

    def on_wakeup(self, current_time: dt.datetime) -> None:
        self.kernel.set_wakeup(self.my_id, current_time + self.poll_interval)
        if self._in_flight:
            return
        mid = self.book.mid()  # None if a side is empty -> skip
        if mid is None:
            return

        ref = self.oracle.get_price(current_time)
        gap = (ref - mid) / self.book.tick_size
        th, inv = self.threshold_ticks, self.inventory
        if gap >= th:
            side, room = Side.BUY, self.max_inventory - inv
        elif gap <= -th:
            side, room = Side.SELL, self.max_inventory + inv
        elif gap >= 0 and inv < 0:
            side, room = Side.BUY, -inv
        elif gap <= 0 and inv > 0:
            side, room = Side.SELL, inv
        else:
            return

        lot = self.book.lot_size
        want = math.floor(min(room, self.max_order_size) / lot + 1e-9)
        levels = (
            self.book.asks(self.book_depth)
            if side is Side.BUY
            else self.book.bids(self.book_depth)
        )
        take = 0
        for lv in levels:
            if take >= want or (
                lv.price >= ref if side is Side.BUY else lv.price <= ref
            ):
                break
            take += min(lv.lots, want - take)
        if take == 0:
            return

        self._in_flight = True
        msg = MarketOrderMsg(self.my_id, next(self._coid), side, round(take * lot, 8))
        self.kernel.send_message(self.my_id, self.exchange_id, msg)

    def on_message(self, current_time: dt.datetime, msg) -> None:
        if isinstance(msg, FillMsg):
            signed = msg.size if msg.side is Side.BUY else -msg.size
            self.inventory += signed
            self.cash -= signed * msg.price
        elif isinstance(msg, (DoneMsg, OrderRejectedMsg)):
            self._in_flight = False
