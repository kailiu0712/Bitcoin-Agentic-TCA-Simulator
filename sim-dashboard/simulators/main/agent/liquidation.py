import csv
import datetime as dt
import itertools
from pathlib import Path
from typing import Any, Optional, Union

from exchange import (
    BookView,
    DoneMsg,
    FillMsg,
    MarketOrderMsg,
    OrderAcceptedMsg,
    OrderRejectedMsg,
    Side,
)
from kernel import AgentId, Kernel


class LiquidationBot:
    """TWAP Liquidation Agent."""

    def __init__(
        self,
        my_id: AgentId,
        exchange_id: AgentId,
        kernel: Kernel,
        side: Side,
        total_quantity: float,
        start_time: dt.datetime,
        end_time: dt.datetime,
        num_slices: int = 10,
        book: Optional[BookView] = None,
        spill_dir: Optional[Union[str, Path]] = None,
        report_filename: str = "execution_report.csv",
    ) -> None:
        self.my_id = my_id
        self.exchange_id = exchange_id
        self.kernel = kernel
        self.side = side
        self.total_quantity = float(total_quantity)
        self.start_time = start_time
        self.end_time = end_time
        self.num_slices = max(1, num_slices)
        self.book = book
        self.spill_dir = Path(spill_dir) if spill_dir is not None else None
        self.report_filename = report_filename

        total_duration = (end_time - start_time).total_seconds()
        if total_duration <= 0:
            raise ValueError("end_time must be later than start_time")

        self.slice_interval = dt.timedelta(seconds=total_duration / self.num_slices)
        self.qty_per_slice = self.total_quantity / self.num_slices
        self.remaining_quantity = self.total_quantity

        self.twap_samples = []

        self._slice_idx = 0
        self._order_seq = itertools.count(1)
        self.order_id_to_cid: dict[int, int] = {}

        self.fills: list[dict[str, Any]] = []
        self.total_executed_qty: float = 0.0
        self.total_executed_notional: float = 0.0
        self.arrival_price: Optional[float] = None

    def kernel_start(self, start_time: dt.datetime) -> None:
        wake_time = max(start_time, self.start_time)
        self.kernel.set_wakeup(self.my_id, wake_time)

    def kernel_stop(self, stop_time: dt.datetime) -> None:
        self._generate_report()

    def _current_mid(self) -> Optional[float]:
        if self.book is None:
            return None
        bid_t = self.book.best_bid_ticks()
        ask_t = self.book.best_ask_ticks()
        if bid_t is None or ask_t is None:
            return None
        return 0.5 * (bid_t + ask_t) * self.book._ex.tick_size

    def on_wakeup(self, current_time: dt.datetime) -> None:
        if current_time < self.start_time:
            self.kernel.set_wakeup(self.my_id, self.start_time)
            return

        mid = self._current_mid()
        if mid is not None:
            if self.arrival_price is None:
                self.arrival_price = mid
            self.twap_samples.append((current_time, mid))

        if self._slice_idx >= self.num_slices or self.remaining_quantity <= 1e-9:
            return

        if self._slice_idx == self.num_slices - 1:
            slice_size = self.remaining_quantity
        else:
            slice_size = min(self.qty_per_slice, self.remaining_quantity)

        if slice_size > 0:
            cid = next(self._order_seq)
            self.kernel.send_message(
                self.my_id,
                self.exchange_id,
                MarketOrderMsg(
                    agent_id=self.my_id,
                    client_order_id=cid,
                    side=self.side,
                    size=slice_size,
                ),
            )
            self.remaining_quantity -= slice_size

        self._slice_idx += 1
        if self._slice_idx < self.num_slices:
            next_wake = self.start_time + self._slice_idx * self.slice_interval
            if next_wake <= self.end_time:
                self.kernel.set_wakeup(self.my_id, next_wake)

    def on_message(self, current_time: dt.datetime, msg: Any) -> None:
        if isinstance(msg, OrderAcceptedMsg):
            self.order_id_to_cid[msg.order_id] = msg.client_order_id
        elif isinstance(msg, FillMsg):
            self.total_executed_qty += msg.size
            self.total_executed_notional += msg.size * msg.price
            self.fills.append(
                {
                    "time": current_time.isoformat(),
                    "order_id": msg.order_id,
                    "trade_id": msg.trade_id,
                    "side": msg.side.name,
                    "price": msg.price,
                    "size": msg.size,
                    "notional": msg.price * msg.size,
                    "liquidity": msg.liquidity,
                }
            )
        elif isinstance(msg, DoneMsg):
            pass
        elif isinstance(msg, OrderRejectedMsg):
            pass

    def _generate_report(self) -> None:
        if self.spill_dir is None:
            return
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.spill_dir / self.report_filename

        vwap = (
            self.total_executed_notional / self.total_executed_qty
            if self.total_executed_qty > 0
            else 0.0
        )
        completion_rate = (
            (self.total_executed_qty / self.total_quantity * 100.0)
            if self.total_quantity > 0
            else 0.0
        )

        sign = 1.0 if self.side is Side.BUY else -1.0
        twap_benchmark = (
            sum(p for _, p in self.twap_samples) / len(self.twap_samples)
            if self.twap_samples
            else 0.0
        )

        # Implementation Shortfall (bps & USD)
        slippage_bps = (
            sign * (vwap - self.arrival_price) / self.arrival_price * 10_000
            if self.arrival_price and vwap > 0
            else 0.0
        )
        slippage_usd = (
            sign * self.total_executed_qty * (vwap - self.arrival_price)
            if self.arrival_price and vwap > 0
            else 0.0
        )

        # Shortfall w.r.t. TWAP (bps)
        twap_shortfall_bps = (
            sign * (vwap - twap_benchmark) / twap_benchmark * 10_000
            if twap_benchmark > 0 and vwap > 0
            else 0.0
        )

        summary_rows = [
            ["Side", self.side.name],
            ["Target Quantity", f"{self.total_quantity:.6f}"],
            ["Executed Quantity", f"{self.total_executed_qty:.6f}"],
            ["Completion Rate", f"{completion_rate:.2f}%"],
            ["VWAP", f"{vwap:.6f}"],
            [
                "Arrival Price",
                f"{self.arrival_price if self.arrival_price else 0.0:.6f}",
            ],
            ["Slippage (bps)", f"{slippage_bps:.2f}"],
            ["Slippage (USD)", f"{slippage_usd:.2f}"],
            ["Shortfall w.r.t. TWAP (bps)", f"{twap_shortfall_bps:.2f}"],
            ["Total Notional", f"{self.total_executed_notional:.2f}"],
            ["Start Time", self.start_time.isoformat()],
            ["End Time", self.end_time.isoformat()],
            ["Number of Slices", str(self.num_slices)],
            ["Number of Fills", str(len(self.fills))],
        ]

        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["# Execution Summary"])
            writer.writerow(["metric", "value"])
            writer.writerows(summary_rows)
            writer.writerow([])
            writer.writerow(["# Fills Detail"])
            if self.fills:
                header = list(self.fills[0].keys())
                writer.writerow(header)
                for fill in self.fills:
                    writer.writerow([fill[k] for k in header])
            else:
                writer.writerow(
                    [
                        "time",
                        "order_id",
                        "trade_id",
                        "side",
                        "price",
                        "size",
                        "notional",
                        "liquidity",
                    ]
                )

        print(
            f"[LiquidationBot] Report generated at '{report_path}'. (Filled: {completion_rate:.1f}%, VWAP: {vwap:.4f}, Slippage: {slippage_bps:.2f} bps)"
        )
