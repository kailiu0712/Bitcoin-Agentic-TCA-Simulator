import datetime as dt
import itertools
import math

from exchange import (
    CancelOrderMsg,
    DoneMsg,
    FillMsg,
    LimitOrderMsg,
    OrderAcceptedMsg,
    OrderRejectedMsg,
    Side,
)
from kernel import AgentId, Kernel, Oracle


class SmartMM:
    """
    Classic ladder market maker with linear inventory skew.

        fair   = oracle price as seen `ref_lag` ago (the MM is a bit stale)
        center = fair - skew_ticks * clip(inventory / max_inventory, -1, 1)
        bid_k  = floor(center - half_spread - k * level_gap)   k = 0..n_levels-1
        ask_k  = ceil (center + half_spread + k * level_gap)

    - at |inventory| >= max_inventory the side that would add to it is pulled
    - quotes are post-only and diffed against live orders, so a level that
      does not change keeps its queue position
    - re-quotes every `refresh_interval`, and `reaction_delay` after any fill
    """

    def __init__(
        self,
        my_id: AgentId,
        exchange_id: AgentId,
        kernel: Kernel,
        oracle: Oracle,
        *,
        tick_size: float,
        quote_size: float,
        max_inventory: float,
        skew_ticks: float,
        n_levels: int = 5,
        half_spread_ticks: int = 1,
        level_gap_ticks: int = 2,
        ref_lag: dt.timedelta = dt.timedelta(seconds=1),
        refresh_interval: dt.timedelta = dt.timedelta(seconds=1),
        reaction_delay: dt.timedelta = dt.timedelta(0),
    ):
        self.my_id = my_id
        self.exchange_id = exchange_id
        self.kernel = kernel
        self.oracle = oracle
        self.tick_size = tick_size
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self.skew_ticks = skew_ticks
        self.n_levels = n_levels
        self.half_spread_ticks = half_spread_ticks
        self.level_gap_ticks = level_gap_ticks
        self.ref_lag = ref_lag
        self.refresh_interval = refresh_interval
        self.reaction_delay = reaction_delay

        self.inventory = 0.0
        self.cash = 0.0

        self._coid = itertools.count(1)
        self._pending: dict[int, tuple[Side, int]] = {}  # client id -> (side, ticks)
        self._live: dict[int, list] = {}  # order id -> [side, ticks, remaining]
        self._cancelling: set[int] = set()  # order ids with a cancel in flight
        self._drop: set[int] = set()  # client ids to cancel once accepted
        self._next_refresh = dt.datetime.min
        self._requote_queued = False

    # protocol

    def kernel_start(self, start_time: dt.datetime) -> None:
        self._next_refresh = start_time
        self.kernel.set_wakeup(self.my_id, start_time)

    def kernel_stop(self, stop_time: dt.datetime) -> None:
        ref = self.oracle.get_price(stop_time)
        print(
            f"[SmartMM] inventory={self.inventory:.2f} cash={self.cash:.2f} "
            f"mtm_pnl={self.cash + self.inventory * ref:.2f}"
        )

    def on_wakeup(self, current_time: dt.datetime) -> None:
        self._requote_queued = False
        if current_time >= self._next_refresh:
            self._next_refresh = current_time + self.refresh_interval
            self.kernel.set_wakeup(self.my_id, self._next_refresh)
        self._requote(current_time)

    def on_message(self, current_time: dt.datetime, msg) -> None:
        if isinstance(msg, OrderAcceptedMsg):
            side, ticks = self._pending.pop(msg.client_order_id)
            self._live[msg.order_id] = [side, ticks, self.quote_size]
            if msg.client_order_id in self._drop:
                self._drop.discard(msg.client_order_id)
                self._cancel(msg.order_id)
        elif isinstance(msg, FillMsg):
            signed = msg.size if msg.side is Side.BUY else -msg.size
            self.inventory += signed
            self.cash -= signed * msg.price
            order = self._live.get(msg.order_id)
            if order is not None:
                order[2] = round(order[2] - msg.size, 8)
            if not self._requote_queued:  # batch all fills of one sweep
                self._requote_queued = True
                self.kernel.set_wakeup(self.my_id, current_time + self.reaction_delay)
        elif isinstance(msg, DoneMsg):
            self._live.pop(msg.order_id, None)
            self._cancelling.discard(msg.order_id)
        elif isinstance(msg, OrderRejectedMsg):
            if msg.reason == "order not found":  # a cancel; the order is already gone
                self._live.pop(msg.client_order_id, None)
                self._cancelling.discard(msg.client_order_id)
            else:  # a post-only / increment reject; retried at the next re-quote
                self._pending.pop(msg.client_order_id, None)
                self._drop.discard(msg.client_order_id)

    # quoting

    def _targets(self, now: dt.datetime) -> dict[tuple[Side, int], float]:
        fair = self.oracle.get_price(now - self.ref_lag)
        skew = self.skew_ticks * max(
            -1.0, min(1.0, self.inventory / self.max_inventory)
        )
        center = round(
            fair / self.tick_size - skew, 6
        )  # round: kill float noise before floor/ceil

        want: dict[tuple[Side, int], float] = {}
        for k in range(self.n_levels):
            off = self.half_spread_ticks + k * self.level_gap_ticks
            if self.inventory < self.max_inventory and center - off >= 1:
                want[(Side.BUY, math.floor(center - off))] = self.quote_size
            if self.inventory > -self.max_inventory:
                want[(Side.SELL, math.ceil(center + off))] = self.quote_size
        return want

    def _requote(self, now: dt.datetime) -> None:
        want = self._targets(now)
        kept: set[tuple[Side, int]] = set()

        for oid, (side, ticks, remaining) in list(self._live.items()):
            if oid in self._cancelling:
                continue
            key = (side, ticks)
            if key in want and key not in kept and abs(remaining - want[key]) < 1e-9:
                kept.add(key)
            else:  # off-target or partially filled -> replace at full size
                self._cancel(oid)

        for coid, key in self._pending.items():
            if coid in self._drop:
                continue
            if key in want and key not in kept:
                kept.add(key)
            else:
                self._drop.add(coid)

        for key, size in want.items():
            if key not in kept:
                self._place(key[0], key[1], size)

    def _place(self, side: Side, ticks: int, size: float) -> None:
        coid = next(self._coid)
        self._pending[coid] = (side, ticks)
        price = round(ticks * self.tick_size, 8)
        msg = LimitOrderMsg(self.my_id, coid, side, price, size, post_only=True)
        self.kernel.send_message(self.my_id, self.exchange_id, msg)

    def _cancel(self, order_id: int) -> None:
        self._cancelling.add(order_id)
        self.kernel.send_message(
            self.my_id, self.exchange_id, CancelOrderMsg(self.my_id, order_id)
        )
