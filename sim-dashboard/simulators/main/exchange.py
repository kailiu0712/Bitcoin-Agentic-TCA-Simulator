from __future__ import annotations

import datetime as dt
import gc
import heapq
import itertools
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

from kernel import AgentId, Kernel


class Side(Enum):
    BUY = auto()
    SELL = auto()

_EPOCH = dt.datetime(1970, 1, 1)


# agent -> exchange


@dataclass(slots=True)
class LimitOrderMsg:
    agent_id: AgentId
    client_order_id: int
    side: Side
    price: float
    size: float
    post_only: bool = False


@dataclass(slots=True)
class MarketOrderMsg:
    agent_id: AgentId
    client_order_id: int
    side: Side
    size: float


@dataclass(slots=True)
class CancelOrderMsg:
    agent_id: AgentId
    order_id: int


# exchange -> agent


@dataclass(slots=True)
class OrderAcceptedMsg:
    client_order_id: int
    order_id: int


@dataclass(slots=True)
class OrderRejectedMsg:
    client_order_id: int
    reason: str


@dataclass(slots=True)
class FillMsg:
    order_id: int
    trade_id: int
    price: float
    size: float
    side: Side
    liquidity: str  # "maker" | "taker"


@dataclass(slots=True)
class DoneMsg:
    order_id: int
    reason: str  # "filled" | "canceled" | "no_liquidity" | "self-trade"


# internal order representation


@dataclass(slots=True)
class MarketOrder:
    order_id: int
    agent_id: AgentId
    side: Side
    size: int
    time: dt.datetime


@dataclass(slots=True)
class LimitOrder:
    order_id: int
    agent_id: AgentId
    side: Side
    price: int
    size: int
    time: dt.datetime


class PriceLevel:
    __slots__ = ("total_size", "orders")

    def __init__(self) -> None:
        self.total_size: int = 0
        self.orders: dict[int, LimitOrder] = {}


# read-only book view


@dataclass(frozen=True, slots=True)
class Level:
    """one aggregated price level (L2): no order ids, no agent ids"""

    price: float  # USD
    size: float  # base units
    price_ticks: int
    lots: int


class BookView:
    """
    Read-only, zero-latency L2 view of an Exchange's order book.
    """

    __slots__ = ("_ex",)

    def __init__(self, exchange: Exchange) -> None:
        object.__setattr__(self, "_ex", exchange)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("BookView is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BookView is read-only")

    # unit info

    @property
    def tick_size(self) -> float:
        return self._ex.tick_size

    @property
    def lot_size(self) -> float:
        return self._ex.lot_size

    # top of book

    def best_bid_ticks(self) -> Optional[int]:
        return self._ex._best_price(Side.BUY)

    def best_ask_ticks(self) -> Optional[int]:
        return self._ex._best_price(Side.SELL)

    def best_bid(self) -> Optional[float]:
        t = self.best_bid_ticks()
        return None if t is None else self._ex._ticks_to_price(t)

    def best_ask(self) -> Optional[float]:
        t = self.best_ask_ticks()
        return None if t is None else self._ex._ticks_to_price(t)

    def mid(self) -> Optional[float]:
        """None unless both sides are non-empty"""
        bid, ask = self.best_bid_ticks(), self.best_ask_ticks()
        if bid is None or ask is None:
            return None
        return round((bid + ask) / 2 * self._ex.tick_size, 8)

    def spread(self) -> Optional[float]:
        """None unless both sides are non-empty"""
        bid, ask = self.best_bid_ticks(), self.best_ask_ticks()
        if bid is None or ask is None:
            return None
        return self._ex._ticks_to_price(ask - bid)

    # depth

    def bids(self, depth: int = 10) -> tuple[Level, ...]:
        """best (highest) first"""
        return self._levels(Side.BUY, depth)

    def asks(self, depth: int = 10) -> tuple[Level, ...]:
        """best (lowest) first"""
        return self._levels(Side.SELL, depth)

    def size_at(self, side: Side, price: float) -> float:
        """resting size at an exact price; 0.0 if no level or price off-tick"""
        ex = self._ex
        ticks = ex._price_to_ticks(price)
        if ticks is None:
            return 0.0
        level = (ex.bids if side is Side.BUY else ex.asks).get(ticks)
        return 0.0 if level is None else ex._lots_to_size(level.total_size)

    def size_at_ticks(self, side: Side, ticks: int) -> float:
        """fast path: resting size at an exact integer tick price"""
        ex = self._ex
        level = (ex.bids if side is Side.BUY else ex.asks).get(ticks)
        return 0.0 if level is None else level.total_size * ex.lot_size

    def is_empty(self, side: Side) -> bool:
        return self._ex._best_price(side) is None

    def _levels(self, side: Side, depth: int) -> tuple[Level, ...]:
        if depth <= 0:
            return ()
        ex = self._ex
        return tuple(
            Level(ex._ticks_to_price(p), ex._lots_to_size(q), p, q)
            for p, q in ex._levels(side, depth)
        )


# LOBSTER tape


class LobsterEventType(Enum):
    SUBMISSION = 1
    PARTIAL_CANCEL = 2
    DELETION = 3
    EXECUTION_VISIBLE = 4


# LOBSTER sentinels for empty levels
_EMPTY_ASK_PRICE = 9_999_999_999
_EMPTY_BID_PRICE = -9_999_999_999


class RowRecorder:
    """
    spill_dir=None  -> in memory
    spill_dir=path  -> written to npy
    """

    def __init__(
        self,
        n_cols: int,
        chunk_rows: int = 1 << 16,
        spill_dir: Optional[Union[str, Path]] = None,
        prefix: str = "orderbook",
    ) -> None:
        self.n_cols = n_cols
        self.chunk_rows = chunk_rows
        self.spill_dir = Path(spill_dir) if spill_dir is not None else None
        self.prefix = prefix
        if self.spill_dir is not None:
            self.spill_dir.mkdir(parents=True, exist_ok=True)

        self._buf = np.empty((chunk_rows, n_cols), dtype=np.int64)
        self._n = 0
        self._chunks: list[np.ndarray] = []
        self._files: list[Path] = []
        self.n_rows = 0

    def __len__(self) -> int:
        return self.n_rows

    def new_row(self) -> np.ndarray:
        if self._n == self.chunk_rows:
            self._rollover()
        row = self._buf[self._n]
        self._n += 1
        self.n_rows += 1
        return row

    def _spill(self, rows: np.ndarray) -> None:
        if self.spill_dir is None:
            return
        path = self.spill_dir / f"{self.prefix}_{len(self._files):06d}.npy"
        np.save(path, rows)
        self._files.append(path)

    def _rollover(self) -> None:
        if self.spill_dir is None:
            self._chunks.append(self._buf)
            self._buf = np.empty((self.chunk_rows, self.n_cols), dtype=np.int64)
        else:
            self._spill(self._buf)
        self._n = 0

    def flush(self) -> None:
        if self.spill_dir is not None and self._n > 0:
            self._spill(self._buf[: self._n])
            self._n = 0

    def iter_chunks(self) -> Iterator[np.ndarray]:
        for f in self._files:
            yield np.load(f, mmap_mode="r")
        yield from self._chunks
        if self._n:
            yield self._buf[: self._n]

    def to_array(self) -> np.ndarray:
        chunks = list(self.iter_chunks())
        if not chunks:
            return np.empty((0, self.n_cols), dtype=np.int64)
        return np.concatenate(chunks)

    def to_csv(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            for chunk in self.iter_chunks():
                np.savetxt(f, chunk, fmt="%d", delimiter=",")

    def cleanup_chunks(self) -> None:
        gc.collect()
        for f in self._files:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
        self._files.clear()


class Exchange:
    """a single-pair FIFO exchange"""

    def __init__(
        self,
        my_id: AgentId,
        kernel: Kernel,
        tick_size: float = 0.01,
        lot_size: float = 1e-8,
        book_levels: Optional[int] = 10,  # None / 0 disables book recording
        spill_dir: Optional[Union[str, Path]] = None,
        book_chunk_rows: int = 1 << 16,
        msg_filename: str = "message.csv",
        orderbook_filename: str = "orderbook.csv",
    ):
        self.my_id = my_id
        self.kernel = kernel
        self.tick_size = tick_size
        self.lot_size = lot_size

        price_scale = tick_size * 10_000
        if abs(price_scale - round(price_scale)) > 1e-6:
            raise ValueError(
                "tick_size must be a multiple of 1e-4 to be representable in LOBSTER's "
                "price*10000 fixed-point convention"
            )
        self._price_scale = round(price_scale)

        self.bids: dict[int, PriceLevel] = {}
        self.asks: dict[int, PriceLevel] = {}
        self._bid_heap: list[int] = []
        self._ask_heap: list[int] = []
        self.orders_by_id: dict[int, LimitOrder] = {}
        self._book_view = BookView(self)
        self._trade_seq = itertools.count(1)
        self._order_seq = itertools.count(1)

        self.spill_dir = Path(spill_dir) if spill_dir is not None else None
        self.msg_filename = msg_filename
        self.orderbook_filename = orderbook_filename

        self.msg_rec = RowRecorder(
            n_cols=6,
            chunk_rows=book_chunk_rows,
            spill_dir=spill_dir,
            prefix="message",
        )

        self.book_levels = book_levels or 0
        self.book_rec: Optional[RowRecorder] = None
        if self.book_levels > 0:
            self.book_rec = RowRecorder(
                n_cols=4 * self.book_levels,
                chunk_rows=book_chunk_rows,
                spill_dir=spill_dir,
                prefix="orderbook",
            )
            # row template: every level empty
            self._book_template = np.zeros(4 * self.book_levels, dtype=np.int64)
            self._book_template[0::4] = _EMPTY_ASK_PRICE
            self._book_template[2::4] = _EMPTY_BID_PRICE

    # protocol

    def kernel_start(self, start_time: dt.datetime) -> None:
        pass

    def kernel_stop(self, stop_time: dt.datetime) -> None:
        if self.msg_rec is not None:
            self.msg_rec.flush()
            if self.spill_dir is not None:
                msg_csv = self.spill_dir / self.msg_filename
                self.msg_rec.to_csv(msg_csv)
                self.msg_rec.cleanup_chunks()
                print(
                    f"[Exchange] Message tape saved to '{msg_csv}' ({len(self.msg_rec)} rows). Temporary chunks cleaned."
                )

        if self.book_rec is not None:
            self.book_rec.flush()
            if self.spill_dir is not None:
                book_csv = self.spill_dir / self.orderbook_filename
                self.book_rec.to_csv(book_csv)
                self.book_rec.cleanup_chunks()
                print(
                    f"[Exchange] Orderbook snapshot tape saved to '{book_csv}' ({len(self.book_rec)} rows). Temporary chunks cleaned."
                )

    def on_wakeup(self, current_time: dt.datetime) -> None:
        pass

    def on_message(self, current_time: dt.datetime, msg) -> None:
        if isinstance(msg, LimitOrderMsg):
            self._handle_limit(current_time, msg)
        elif isinstance(msg, MarketOrderMsg):
            self._handle_market(current_time, msg)
        elif isinstance(msg, CancelOrderMsg):
            self._handle_cancel(current_time, msg)

    # order handling

    def _handle_limit(self, now: dt.datetime, msg: LimitOrderMsg) -> None:
        price = self._price_to_ticks(msg.price)
        size = self._size_to_lots(msg.size)
        if price is None or size is None:
            self._route(
                msg.agent_id,
                OrderRejectedMsg(msg.client_order_id, "invalid price/size increment"),
            )
            return
        if msg.post_only and self._would_cross(msg.side, price):
            self._route(
                msg.agent_id,
                OrderRejectedMsg(
                    msg.client_order_id, "post_only order would cross the book"
                ),
            )
            return

        order_id = next(self._order_seq)
        order = LimitOrder(order_id, msg.agent_id, msg.side, price, size, now)
        self._route(
            order.agent_id, OrderAcceptedMsg(msg.client_order_id, order.order_id)
        )
        self._match(now, order)

        if order.size > 0:
            self._add_resting_order(order)
            self._log(
                now,
                LobsterEventType.SUBMISSION,
                order.order_id,
                order.size,
                order.price,
                self._direction(order.side),
            )
        else:
            self._route(order.agent_id, DoneMsg(order.order_id, "filled"))

    def _handle_market(self, now: dt.datetime, msg: MarketOrderMsg) -> None:
        size = self._size_to_lots(msg.size)
        if size is None:
            self._route(
                msg.agent_id,
                OrderRejectedMsg(msg.client_order_id, "invalid size increment"),
            )
            return

        order_id = next(self._order_seq)
        order = MarketOrder(order_id, msg.agent_id, msg.side, size, now)
        self._route(
            order.agent_id, OrderAcceptedMsg(msg.client_order_id, order.order_id)
        )
        self._match(now, order)

        reason = "filled" if order.size <= 0 else "no_liquidity"
        self._route(order.agent_id, DoneMsg(order.order_id, reason))

    def _handle_cancel(self, now: dt.datetime, msg: CancelOrderMsg) -> None:
        order = self.orders_by_id.get(msg.order_id)
        if order is None or order.agent_id != msg.agent_id:
            self._route(msg.agent_id, OrderRejectedMsg(msg.order_id, "order not found"))
            return

        book = self.bids if order.side is Side.BUY else self.asks
        level = book[order.price]
        del level.orders[order.order_id]
        level.total_size -= order.size
        if not level.orders:
            del book[order.price]

        del self.orders_by_id[order.order_id]
        self._log(
            now,
            LobsterEventType.DELETION,
            order.order_id,
            order.size,
            order.price,
            self._direction(order.side),
        )
        self._route(msg.agent_id, DoneMsg(order.order_id, "canceled"))

    def book_view(self) -> BookView:
        """read-only, zero-latency L2 view of this exchange's book"""
        return self._book_view

    # matching mechanism

    def _match(
        self, now: dt.datetime, incoming: Union[LimitOrder, MarketOrder]
    ) -> None:
        opposite_side = Side.SELL if incoming.side is Side.BUY else Side.BUY
        book = self.asks if opposite_side is Side.SELL else self.bids

        while incoming.size > 0:
            best_price = self._best_price(opposite_side)
            if best_price is None:
                break
            if isinstance(incoming, LimitOrder):
                crosses = (
                    incoming.side is Side.BUY and best_price <= incoming.price
                ) or (incoming.side is Side.SELL and best_price >= incoming.price)
                if not crosses:
                    break

            level = book[best_price]
            orders = level.orders
            while orders and incoming.size > 0:
                resting = next(iter(orders.values()))

                if resting.agent_id == incoming.agent_id:
                    # STP (decrement-and-cancel)
                    dec = min(resting.size, incoming.size)
                    resting.size -= dec
                    incoming.size -= dec
                    level.total_size -= dec
                    direction = self._direction(resting.side)
                    if resting.size <= 0:
                        del orders[resting.order_id]
                        del self.orders_by_id[resting.order_id]
                        self._log(
                            now,
                            LobsterEventType.DELETION,
                            resting.order_id,
                            dec,
                            resting.price,
                            direction,
                        )
                        self._route(
                            resting.agent_id, DoneMsg(resting.order_id, "self-trade")
                        )
                    else:
                        self._log(
                            now,
                            LobsterEventType.PARTIAL_CANCEL,
                            resting.order_id,
                            dec,
                            resting.price,
                            direction,
                        )
                    continue

                trade_size = min(resting.size, incoming.size)
                trade_id = next(self._trade_seq)
                resting.size -= trade_size
                incoming.size -= trade_size
                level.total_size -= trade_size
                self._log(
                    now,
                    LobsterEventType.EXECUTION_VISIBLE,
                    resting.order_id,
                    trade_size,
                    best_price,
                    self._direction(resting.side),
                )
                fill_price = self._ticks_to_price(best_price)
                fill_size = self._lots_to_size(trade_size)
                self._route(
                    resting.agent_id,
                    FillMsg(
                        resting.order_id,
                        trade_id,
                        fill_price,
                        fill_size,
                        resting.side,
                        "maker",
                    ),
                )
                self._route(
                    incoming.agent_id,
                    FillMsg(
                        incoming.order_id,
                        trade_id,
                        fill_price,
                        fill_size,
                        incoming.side,
                        "taker",
                    ),
                )
                if resting.size <= 0:
                    del orders[resting.order_id]
                    del self.orders_by_id[resting.order_id]
                    self._route(resting.agent_id, DoneMsg(resting.order_id, "filled"))

            if not orders:
                del book[best_price]

    # order maintainence

    def _add_resting_order(self, order: LimitOrder) -> None:
        book = self.bids if order.side is Side.BUY else self.asks
        level = book.get(order.price)
        if level is None:
            level = PriceLevel()
            book[order.price] = level
            self._push_price(order.side, order.price)
        level.orders[order.order_id] = order
        level.total_size += order.size
        self.orders_by_id[order.order_id] = order

    def _push_price(self, side: Side, price: int) -> None:
        heap = self._bid_heap if side is Side.BUY else self._ask_heap
        heapq.heappush(heap, -price if side is Side.BUY else price)

    def _best_price(self, side: Side) -> Optional[int]:
        heap = self._bid_heap if side is Side.BUY else self._ask_heap
        book = self.bids if side is Side.BUY else self.asks
        while heap:
            price = -heap[0] if side is Side.BUY else heap[0]
            if price in book:
                return price
            heapq.heappop(heap)
        return None

    def _would_cross(self, side: Side, price: int) -> bool:
        opposite_best = self._best_price(Side.SELL if side is Side.BUY else Side.BUY)
        if opposite_best is None:
            return False
        return price >= opposite_best if side is Side.BUY else price <= opposite_best

    def _levels(self, side: Side, depth: int) -> list[tuple[int, int]]:
        book = self.bids if side is Side.BUY else self.asks
        if not book:
            return []
        prices = (
            heapq.nlargest(depth, book.keys())
            if side is Side.BUY
            else heapq.nsmallest(depth, book.keys())
        )
        return [(p, book[p].total_size) for p in prices]

    # tick/lot <-> float

    def _price_to_ticks(self, price: Optional[float]) -> Optional[int]:
        if price is None or price <= 0:
            return None
        ticks = round(price / self.tick_size)
        if abs(ticks * self.tick_size - price) > self.tick_size * 1e-6:
            return None
        return ticks

    def _ticks_to_price(self, ticks: int) -> float:
        return round(ticks * self.tick_size, 8)

    def _size_to_lots(self, size: Optional[float]) -> Optional[int]:
        if size is None or size <= 0:
            return None
        lots = round(size / self.lot_size)
        if abs(lots * self.lot_size - size) > self.lot_size * 1e-6:
            return None
        return lots

    def _lots_to_size(self, lots: int) -> float:
        return lots * self.lot_size

    def _direction(self, side: Side) -> int:
        return 1 if side is Side.BUY else -1

    # internal

    def _log(
        self,
        time: dt.datetime,
        event_type: LobsterEventType,
        order_id: int,
        size: int,
        price_ticks: int,
        direction: int,
    ) -> None:
        scaled_price = price_ticks * self._price_scale

        if self.msg_rec is not None:
            row = self.msg_rec.new_row()
            row[0] = int(
                (time.replace(tzinfo=None) - _EPOCH).total_seconds() * 1_000_000
            )
            row[1] = event_type.value
            row[2] = order_id
            row[3] = size  # lot
            row[4] = scaled_price  # price * 10000
            row[5] = direction  # 1 = buy, -1 = sell (maker side)

        self._record_book()

    def _record_book(self) -> None:
        """one LOBSTER orderbook row = book state right after the latest message"""
        if self.book_rec is None:
            return
        row = self.book_rec.new_row()
        row[:] = self._book_template
        n = self.book_levels
        scale = self._price_scale
        asks = self.asks
        for i, p in enumerate(heapq.nsmallest(n, asks.keys())):
            row[4 * i] = p * scale
            row[4 * i + 1] = asks[p].total_size
        bids = self.bids
        for i, p in enumerate(heapq.nlargest(n, bids.keys())):
            row[4 * i + 2] = p * scale
            row[4 * i + 3] = bids[p].total_size

    def _route(self, agent_id: AgentId, msg) -> None:
        self.kernel.send_message(self.my_id, agent_id, msg)
