from collections import defaultdict
from .order_book import OrderBook


class Exchange:
    def __init__(self):
        self.book = OrderBook()
        self.trades, self.events = [], []
        self.trade_id = 0
        self.inventory, self.cash = defaultdict(float), defaultdict(float)

    def submit(self, order):
        while order.remaining > 1e-12 and self.book.crosses(order):
            pre_trade_state = self.book.state()
            resting = self.book.best_order("SELL" if order.side == "BUY" else "BUY")
            qty = min(order.remaining, resting.remaining)
            order.remaining -= qty
            resting.remaining -= qty
            self.trade_id += 1
            buy = order if order.side == "BUY" else resting
            sell = order if order.side == "SELL" else resting
            trade = {"timestamp": order.timestamp, "event_type": "TRADE", "agent_id": order.agent_id,
                     "side": order.side, "price": resting.price, "size": qty,
                     "order_id": order.order_id, "trade_id": self.trade_id,
                     "buyer_agent": buy.agent_id, "seller_agent": sell.agent_id,
                     "aggressor_side": order.side}
            trade.update(pre_trade_state)
            self.trades.append(trade)
            self.inventory[buy.agent_id] += qty; self.inventory[sell.agent_id] -= qty
            self.cash[buy.agent_id] -= qty * resting.price; self.cash[sell.agent_id] += qty * resting.price
            self.book.remove_filled(resting)
        if order.remaining > 1e-12 and not order.is_market:
            self.book.add(order)
            kind = "LIMIT_ADD"
        else:
            kind = "MARKET_EXECUTION" if order.is_market else "LIMIT_EXECUTION"
        self._record(order.timestamp, kind, order.agent_id, order.side, order.price, order.remaining, order.order_id)
        return self.trades

    def cancel(self, timestamp, agent_id, order_id):
        ok = self.book.cancel(order_id)
        if ok:
            self._record(timestamp, "LIMIT_CANCEL", agent_id, None, None, 0.0, order_id)
        return ok

    def _record(self, timestamp, event_type, agent_id, side, price, size, order_id):
        row = {"timestamp": timestamp, "event_type": event_type, "agent_id": agent_id, "side": side,
               "price": price, "size": size, "order_id": order_id, "trade_id": None}
        row.update(self.book.state())
        self.events.append(row)

    def record_quote(self,timestamp,agent_id="exchange"):
        self._record(timestamp,"QUOTE_UPDATE",agent_id,None,None,None,None)

    def output(self):
        out = list(self.events)
        for t in self.trades:
            row = dict(t)
            out.append(row)
        return sorted(out, key=lambda x: (x["timestamp"], 0 if x["event_type"] == "TRADE" else 1))
