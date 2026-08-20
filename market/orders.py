from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    order_id: int
    agent_id: str
    side: str
    quantity: float
    timestamp: float
    price: Optional[float] = None
    remaining: Optional[float] = None

    def __post_init__(self):
        self.side = self.side.upper()
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        self.remaining = self.quantity if self.remaining is None else self.remaining

    @property
    def is_market(self):
        return self.price is None

