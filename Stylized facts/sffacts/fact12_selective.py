"""
Stylized fact 12 (P0-C / P1) - Selective liquidity taking: the timing of large trades.

Empirical claim
    Large trades are **not** dropped into randomly chosen book states.
    Schnaubelt, Rende & Krauss find that liquidity improves roughly **2-3
    minutes before** large BTC trades, consistent with traders timing large
    orders when liquidity is relatively favourable.

Why it matters for the simulator
    Drawing organic taker size independently of the book state exaggerates
    ordinary impact: real large takers wait for depth.  Organic taker arrival
    and size should depend on spread, depth and volatility.

Method
    Two complementary tests.

    *Cross-section*: compare the book state an order sees to the unconditional
    (time-weighted) book state, by order-size decile.  If large orders arrived
    blind, every decile would see the average book.

    *Event study*: for large orders, track spread and same-side visible depth
    over a lookback window on the exchange clock, normalised by the
    unconditional mean, and see whether depth builds ahead of the trade.  A
    matched control of median-size orders is shown alongside, so that a
    trivial "liquidity is always mean-reverting" effect cannot be mistaken for
    deliberate timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact12"
FACT_ID = 12
PRIORITY = "P0-C / P1"
TITLE = "Selective liquidity taking / timing of large trades"

N_SIZE_DECILES = 10


def _unconditional(rd, cfg) -> tuple[float, float]:
    """Time-weighted mean spread and mean one-sided depth over all book states."""
    acc = rd.accumulators
    total_w = float(acc["total_weight_ns"][0])
    band_idx = cfg.QUOTE_TAPE_DEPTH_BAND_INDEX
    depth = acc["depth_sum"]
    mean_depth = float((depth[band_idx, 0] + depth[band_idx, 1]) / (2 * total_w))
    hour = acc["hour"]
    mean_spread = float(hour[1].sum() / hour[0].sum()) if hour[0].sum() > 0 else np.nan
    return mean_spread, mean_depth


def _cross_section(rd, cfg) -> pd.DataFrame:
    trades = rd.trades
    qty = trades["qty"].to_numpy(np.float64)
    band = cfg.DEPTH_BANDS_BPS[cfg.QUOTE_TAPE_DEPTH_BAND_INDEX]
    bid_d = trades[f"depth_bid_{band:g}bps"].to_numpy(np.float64)
    ask_d = trades[f"depth_ask_{band:g}bps"].to_numpy(np.float64)
    sign = trades["sign"].to_numpy(np.float64)
    consumed = np.where(sign > 0, ask_d, bid_d)
    spread = trades["spread_bps"].to_numpy(np.float64)

    bins = st.quantile_bins(qty, N_SIZE_DECILES)
    med_qty, counts = st.group_mean(qty, bins, N_SIZE_DECILES)
    mean_depth, _ = st.group_mean(consumed, bins, N_SIZE_DECILES)
    mean_spread, _ = st.group_mean(spread, bins, N_SIZE_DECILES)
    return pd.DataFrame({
        "size_decile": np.arange(1, N_SIZE_DECILES + 1),
        "mean_qty_btc": med_qty,
        "mean_consumed_depth_btc": mean_depth,
        "mean_spread_bps": mean_spread,
        "n": counts,
    })


def _event_study(rd, cfg, mask: np.ndarray, uncond_spread: float,
                 uncond_depth: float) -> pd.DataFrame:
    """Spread / same-side depth on a lookback grid before the aggressive order."""
    trades = rd.trades
    ts = trades["exch_ns"].to_numpy(np.int64)[mask]
    sign = trades["sign"].to_numpy(np.float64)[mask]
    if ts.size < 30:
        return pd.DataFrame()

    q_exch = rd.q_exch
    q_spread = rd.q_spread_bps
    q_bid_d = rd.quotes["bid_depth"].to_numpy(np.float64)
    q_ask_d = rd.quotes["ask_depth"].to_numpy(np.float64)

    rows = []
    for offset in tuple(cfg.PRE_TRADE_LOOKBACK_S) + (0.0,):
        pos = st.asof_left(q_exch, ts + int(offset * 1e9))
        ok = pos >= 0
        pos, sgn = pos[ok], sign[ok]
        if pos.size < 30:
            continue
        consumed = np.where(sgn > 0, q_ask_d[pos], q_bid_d[pos])
        rows.append({
            "offset_s": offset,
            "mean_spread_bps": float(np.nanmean(q_spread[pos])),
            "mean_consumed_depth_btc": float(np.nanmean(consumed)),
            "spread_vs_unconditional": float(np.nanmean(q_spread[pos])) / uncond_spread,
            "depth_vs_unconditional": float(np.nanmean(consumed)) / uncond_depth,
            "n": int(pos.size),
        })
    return pd.DataFrame(rows)


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    cross, events = {}, {}

    for name, rd in ctx.each():
        uncond_spread, uncond_depth = _unconditional(rd, cfg)
        qty = rd.trades["qty"].to_numpy(np.float64)
        large = qty >= np.nanquantile(qty, cfg.LARGE_TRADE_QUANTILE)
        control = (qty >= np.nanquantile(qty, 0.45)) & (qty <= np.nanquantile(qty, 0.55))

        xs = _cross_section(rd, cfg)
        ev_large = _event_study(rd, cfg, large, uncond_spread, uncond_depth)
        ev_ctrl = _event_study(rd, cfg, control, uncond_spread, uncond_depth)
        cross[name] = xs
        events[name] = (ev_large, ev_ctrl)

        # Event-weighted control: the book the average aggressive order sees.
        # Trades cluster in busy moments, so every decile can sit below the
        # time-weighted mean while large orders still time the book relatively
        # well - or, as here, relatively badly.
        order_mean_depth = float(np.average(xs.mean_consumed_depth_btc, weights=xs.n))

        ctx.out.save_table(xs, FACT, "book_state_by_size_decile", name,
                           caption=f"Book state seen by each order-size decile - {rd.label}")
        if len(ev_large):
            ctx.out.save_table(ev_large, FACT, "pre_trade_liquidity_large", name,
                               caption=f"Liquidity before large orders - {rd.label}")

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        ax.plot(xs.mean_qty_btc, xs.mean_consumed_depth_btc, "o-", ms=5, lw=1.7)
        ax.axhline(uncond_depth, **pl.REFERENCE_KW)
        ax.axhline(order_mean_depth, color="#B45309", lw=1.2, ls="-.", alpha=0.9)
        ax.set_xscale("log")
        pl.finish(ax, "Depth seen, by order size", "mean order size in decile (BTC)",
                  "same-side visible depth (BTC)",
                  note="Dashed grey: time-weighted mean depth over all book states. "
                       "Dash-dot orange: the depth the average aggressive order sees "
                       "(the event-weighted control).")

        ax = axes[1]
        if len(ev_large):
            ax.plot(ev_large.offset_s, ev_large.depth_vs_unconditional, "o-", ms=5,
                    lw=1.8, label="large orders (top 1%)")
        if len(ev_ctrl):
            ax.plot(ev_ctrl.offset_s, ev_ctrl.depth_vs_unconditional, "s--", ms=4,
                    lw=1.3, alpha=0.8, label="median-size control")
        ax.axhline(1.0, **pl.REFERENCE_KW)
        ax.axvline(-150, **pl.LITERATURE_KW)
        pl.finish(ax, "Depth build-up before the order", "seconds before the order",
                  "depth / unconditional mean", legend=True,
                  note="Dotted vertical: the 2-3 minute window reported by Schnaubelt et al.")

        ax = axes[2]
        if len(ev_large):
            ax.plot(ev_large.offset_s, ev_large.spread_vs_unconditional, "o-", ms=5,
                    lw=1.8, label="large orders")
        if len(ev_ctrl):
            ax.plot(ev_ctrl.offset_s, ev_ctrl.spread_vs_unconditional, "s--", ms=4,
                    lw=1.3, alpha=0.8, label="median-size control")
        ax.axhline(1.0, **pl.REFERENCE_KW)
        pl.finish(ax, "Spread before the order", "seconds before the order",
                  "spread / unconditional mean", legend=True)
        fig.suptitle(f"Fact 12 - selective liquidity taking | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "selective_liquidity", name)

        top_ratio = float(xs.mean_consumed_depth_btc.iloc[-1] / uncond_depth)
        bot_ratio = float(xs.mean_consumed_depth_btc.iloc[0] / uncond_depth)
        top_vs_orders = float(xs.mean_consumed_depth_btc.iloc[-1] / order_mean_depth)
        at_150 = np.nan
        if len(ev_large):
            near = ev_large.iloc[(ev_large.offset_s + 150).abs().argsort()[:1]]
            at_150 = float(near.depth_vs_unconditional.iloc[0])
        metrics += [
            {"metric": "depth seen by top size decile / unconditional", "regime": name,
             "value": top_ratio, "detail": "> 1 means large orders time the book"},
            {"metric": "depth seen by bottom size decile / unconditional",
             "regime": name, "value": bot_ratio, "detail": ""},
            {"metric": "depth seen by top size decile / depth seen by the average order",
             "regime": name, "value": top_vs_orders,
             "detail": "event-weighted control: > 1 means large orders pick better "
                       "books than other orders do"},
            {"metric": "depth 150 s before a large order / unconditional",
             "regime": name, "value": at_150,
             "detail": "Schnaubelt et al. report liquidity improving 2-3 min ahead"},
            {"metric": "depth at the moment of a large order / 150 s before",
             "regime": name,
             "value": (float(ev_large.loc[ev_large.offset_s == 0.0,
                                          "depth_vs_unconditional"].iloc[0]) / at_150
                       if len(ev_large) and np.isfinite(at_150) and at_150 else np.nan),
             "detail": "> 1 means depth built up into the trade"},
        ]

    fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
    for name, rd in ctx.each():
        xs = cross[name]
        axes[0].plot(xs.size_decile, xs.mean_consumed_depth_btc, "o-", ms=4, lw=1.7,
                     color=rd.color, label=rd.label)
        ev = events[name][0]
        if len(ev):
            axes[1].plot(ev.offset_s, ev.depth_vs_unconditional, "o-", ms=4, lw=1.8,
                         color=rd.color, label=rd.label)
    pl.finish(axes[0], "Depth seen by order-size decile", "order-size decile",
              "same-side visible depth (BTC)", legend=True)
    axes[1].axhline(1.0, **pl.REFERENCE_KW)
    axes[1].axvline(-150, **pl.LITERATURE_KW)
    pl.finish(axes[1], "Depth before large orders", "seconds before the order",
              "depth / unconditional mean", legend=True)
    fig.suptitle("Fact 12 - selective liquidity taking by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "selective_liquidity", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Schnaubelt, Rende & Krauss: liquidity improves ~2-3 minutes "
                      "before large BTC trades.",
    }
