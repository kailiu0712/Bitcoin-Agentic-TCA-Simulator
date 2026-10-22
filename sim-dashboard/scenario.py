"""
Run ONE simulation for the dashboard and save its tape: a model's market (models.py) plus a TWAP
liquidation, or the counterfactual (the same run with nothing liquidated), set up exactly as the
impact-law tests set up their runs (protocol.py). One pipeline for every model.

    python scenario.py --model extended_santafe_none --seed-index 0 --minutes 15 \
        --side sell --quantity 3712.8 --start-min 5 --duration-min 5 --slices 60 --out <dir>
    (add --counterfactual for the same run with nothing liquidated)

Writes into --out:
    message.npy    the exchange's LOBSTER message tape, one row per book change: time (us since 1970),
                   event type, order id, size (lots), price x 10^4, direction. replay.py rebuilds the
                   book from it (the exchange's own orderbook tape is off: it slows a run by ~30%).
    fills.csv      the liquidation bot's fills
    meta.json      the run's configuration
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import protocol as P
from models import EXCHANGE_ID, LIQUIDATION_BOT_ID, MODELS

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--model", choices=list(MODELS), default="extended_santafe_none")
ap.add_argument("--seed-index", type=int, default=0, help=f"0..{P.N_SEEDS - 1}, as in the tests")
ap.add_argument("--minutes", type=float, default=15.0, help="simulated time")
ap.add_argument("--side", choices=["sell", "buy"], default="sell")
ap.add_argument("--quantity", type=float, default=3712.8, help="liquidation size, SOL (default: Q/V_D = 0.2%%)")
ap.add_argument("--start-min", type=float, default=5.0, help="liquidation start, minute (the tests: 10, their burn-in)")
ap.add_argument("--duration-min", type=float, default=5.0)
ap.add_argument("--slices", type=int, default=60, help="TWAP child orders (the tests: 60)")
ap.add_argument("--counterfactual", action="store_true", help="same run, nothing liquidated")
ap.add_argument("--out", required=True)
args = ap.parse_args()

model = MODELS[args.model]
sys.path.insert(0, str(Path(__file__).resolve().parent / "simulators" / model.code))
from agent.liquidation import LiquidationBot  # noqa: E402
from exchange import Exchange, Side  # noqa: E402
from kernel import Kernel  # noqa: E402

run = P.Run(args.seed_index)
# child orders on the exchange's lot grid; Q = slices x child (protocol.Run)
child_size = round(args.quantity / args.slices / P.LOT_SIZE) * P.LOT_SIZE
quantity = 0.0 if args.counterfactual else round(child_size * args.slices, 8)
liq_start = run.start + dt.timedelta(minutes=args.start_min)
liq_end = liq_start + dt.timedelta(minutes=args.duration_min)
stop_time = run.start + dt.timedelta(minutes=args.minutes)

kernel = Kernel(n_agents=model.n_agents)
exchange = Exchange(my_id=EXCHANGE_ID, kernel=kernel, tick_size=P.TICK_SIZE, lot_size=P.LOT_SIZE,
                    book_levels=0, spill_dir=None)   # the tape stays in memory and is saved below
book = exchange.book_view()
market_maker, noise_trader, others = model.market(kernel, book, run)
# the counterfactual keeps the bot with nothing to sell, so both runs have the same agents
liquidation_bot = LiquidationBot(
    my_id=LIQUIDATION_BOT_ID, exchange_id=EXCHANGE_ID, kernel=kernel,
    side=Side.SELL if args.side == "sell" else Side.BUY, total_quantity=quantity,
    start_time=liq_start, end_time=liq_end, num_slices=args.slices, book=book)

kernel.set_agents([exchange, market_maker, noise_trader, liquidation_bot, *others])
print(f"{args.model} seed {run.seed}: {args.minutes:g} min, "
      f"{'counterfactual' if args.counterfactual else f'{args.side} {quantity:g} SOL'}", flush=True)
kernel.run(run.start, stop_time, progress_bar=True)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.save(out / "message.npy", exchange.msg_rec.to_array())
pd.DataFrame(liquidation_bot.fills, columns=["time", "order_id", "trade_id", "side", "price", "size",
                                             "notional", "liquidity"]).to_csv(out / "fills.csv", index=False)
meta = {
    "model": args.model, "seed_index": args.seed_index, "seed": run.seed, "counterfactual": args.counterfactual,
    "venue": "Coinbase", "asset": "SOL",
    "start_time": run.start.isoformat(), "stop_time": stop_time.isoformat(),
    "liq_start": liq_start.isoformat(), "liq_end": liq_end.isoformat(),
    "side": args.side, "quantity": quantity, "slices": args.slices,
    "tick_size": P.TICK_SIZE, "lot_size": P.LOT_SIZE, "order_size": P.ORDER_SIZE,
    "daily_volume": P.V_D, "init_price": run.init_price,
}
(out / "meta.json").write_text(json.dumps(meta, indent=2))   # written last: the run is complete
print(f"saved {len(exchange.msg_rec)} tape rows to {out}", flush=True)
