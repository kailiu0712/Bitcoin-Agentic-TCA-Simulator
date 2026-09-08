"""
Stylized fact 8c (P0-C) - real-time (clock) OFI, and TFI under several
definitions.

Why this exists, next to fact08 and fact08b
    ``fact08`` aggregates OFI over blocks of 50 *quote events* - an event-time
    clock.  That is the natural microstructure choice, but it is not what
    Cont, Kukanov & Stoikov regress and not what ``Stylied facts ZJ.pdf`` p8
    tabulates: both use fixed **wall-clock** intervals.  A 50-quote block lasts
    a few seconds when the market is busy and a minute when it is quiet, so its
    coefficient is not comparable with a per-second one.

    This module therefore reports the same regression on a real-time clock, over
    a horizon grid extended past ZJ's to 300 s and 600 s, and prints the three
    numbers ZJ tabulate: **beta, its t-statistic, and R-squared**.

    ``fact08b`` already covers 1-60 s and adds the depth conditioning.  Nothing
    there is changed by this module; instead, the overlapping horizons are
    **cross-checked against it at runtime** (see ``_panel``), so the two cannot
    silently drift apart.

The TFI definition problem
    ZJ never state how they build trade-flow imbalance, and the choice matters
    far more than the choice of clock.  Signed *volume* is the obvious reading
    and is what ``fact08b`` reports, but BTC trade sizes span three orders of
    magnitude (p99/median ~ 700x, and the largest 1% of orders carry ~40% of
    all volume), so a volume-weighted regressor is dominated by a handful of
    prints.  Signed trade *count* weights every decision equally.

    Four definitions are therefore estimated on one identical window partition:

        signed_volume_btc     sum of eps_j * q_j          (fact08b's choice)
        signed_notional_usd   sum of eps_j * q_j * p_j
        signed_trade_count    sum of eps_j
        signed_sqrt_volume    sum of eps_j * sqrt(q_j)

    R-squared is invariant to rescaling a regressor, so volume and notional
    differ only through the within-window variation of price - they should and
    do come out nearly identical.  Count and sqrt-volume are genuinely
    different regressors, not rescalings.

    Reporting only one of these as "the" TFI overstates how settled the
    comparison with OFI is; all four are reported, with the one closest to ZJ's
    published BTC value flagged in the output table.

Interpretation
    Every R-squared here is *contemporaneous* explanatory power on the same
    window, not prediction and not a causal decomposition.  OFI and TFI are not
    independent either: a trade that consumes the queue changes the best size,
    so the same event enters both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import fits
from sfcore import plotting as pl
from sffacts.fact08_ofi import _ofi_series
from sffacts.fact08b_ofi_tfi import _window_panel as _fact08b_panel

FACT = "fact08c"
FACT_ID = 8
PRIORITY = "P0-C"
TITLE = "Real-time OFI across horizons, and TFI under several definitions"

#: Wall-clock horizons.  ZJ p8 stops at 60 s; 300 s and 600 s are added to show
#: where the contemporaneous relation stops tightening.
WINDOWS_S: tuple[float, ...] = (1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0)

#: Quote-event block sizes for the event-time reference rows.  Included so the
#: "real time instead of event time" contrast is visible in one table; these are
#: computed with the same endpoint convention as the clock windows and are NOT
#: fact08's numbers, which use a different alignment.
EVENT_BLOCKS: tuple[int, ...] = (10, 50, 200, 1000)

#: ZJ p9's published BTC TFI R-squared at 10 s, read off the bar chart. Used
#: only to label which definition lands closest; never to fit anything.
ZJ_BTC_TFI_R2_AT_10S = 0.166

MIN_WINDOWS = 200


def _tfi_series(trades) -> dict[str, np.ndarray]:
    """The four trade-flow-imbalance regressors, per aggressive order."""
    sign = trades["sign"].to_numpy(np.float64)
    qty = trades["qty"].to_numpy(np.float64)
    return {
        "signed_volume_btc": sign * qty,
        "signed_notional_usd": sign * trades["notional"].to_numpy(np.float64),
        "signed_trade_count": sign.copy(),
        "signed_sqrt_volume": sign * np.sqrt(qty),
    }


def _panel(rd, seconds: float, verify: bool = True) -> pd.DataFrame:
    """One row per non-overlapping clock window; OFI, dmid and every TFI.

    ``e_n`` describes the transition from quote row n-1 to row n, so a window
    holding rows [i0..i1] accounts for the price change from row i0-1 to i1.
    The opening price is therefore read at ``i0 - 1``, which is why a window
    starting at the very first row is dropped.

    When ``verify`` and the horizon is one ``fact08b`` also computes, the OFI
    and dmid columns are asserted equal to that module's, so the two cannot
    drift apart as either is edited.
    """
    quotes = rd.quotes
    qts = quotes["recv_ns"].to_numpy(np.int64)
    mid = rd.q_mid
    ofi = _ofi_series(rd)

    trades = rd.trades
    tts = trades["recv_ns"].to_numpy(np.int64)
    tfi = _tfi_series(trades)

    step = int(seconds * 1e9)
    first = (qts[0] // step) * step
    edges = np.arange(first, qts[-1] + step, step, dtype=np.int64)
    qi = np.searchsorted(qts, edges, side="left")
    ti = np.searchsorted(tts, edges, side="left")

    cum_ofi = np.concatenate([[0.0], np.cumsum(ofi)])
    start, stop = qi[:-1], qi[1:]
    ok = (stop > start) & (start >= 1)
    start, stop = start[ok], stop[ok]
    lo, hi = ti[:-1][ok], ti[1:][ok]

    panel = pd.DataFrame({
        "window_start_ns": edges[:-1][ok],
        "ofi_btc": cum_ofi[stop] - cum_ofi[start],
        "dmid_usd": mid[stop - 1] - mid[start - 1],
        "dmid_bps": (mid[stop - 1] - mid[start - 1]) / mid[start - 1] * 1e4,
        "n_quote_events": (stop - start).astype(np.float64),
        "n_trades": (hi - lo).astype(np.float64),
    })
    for name, values in tfi.items():
        cum = np.concatenate([[0.0], np.cumsum(values)])
        panel[name] = cum[hi] - cum[lo]

    if verify and seconds in (1.0, 5.0, 10.0, 30.0, 60.0):
        reference = _fact08b_panel(rd, seconds)
        if len(reference) != len(panel):
            raise AssertionError(f"fact08c/fact08b window count differs at {seconds}s")
        for column in ("ofi_btc", "dmid_usd"):
            if not np.allclose(reference[column].to_numpy(),
                               panel[column].to_numpy(), rtol=0, atol=1e-9):
                raise AssertionError(f"fact08c/fact08b disagree on {column} at {seconds}s")
    return panel


def _event_panel(rd, block: int) -> pd.DataFrame:
    """Event-time reference: OFI over fixed blocks of quote events."""
    mid = rd.q_mid
    ofi = _ofi_series(rd)
    n = (len(ofi) // block) * block
    if n < block * MIN_WINDOWS:
        return pd.DataFrame()
    # Blocks start at row 1 so every block has a predecessor for its opening
    # price, matching the clock-window convention exactly.
    starts = np.arange(1, n - block + 1, block)
    stops = starts + block
    cum = np.concatenate([[0.0], np.cumsum(ofi)])
    return pd.DataFrame({
        "ofi_btc": cum[stops] - cum[starts],
        "dmid_usd": mid[stops - 1] - mid[starts - 1],
    })


def _fit_row(y, x, label) -> dict:
    fit = fits.ols_hc3(np.asarray(y, dtype=np.float64),
                       np.asarray(x, dtype=np.float64)[:, None], names=[label])
    return {"beta": float(fit["params"][1]), "std_error": float(fit["se"][1]),
            "t_stat": float(fit["t"][1]), "r2": fit["r2"], "n_windows": fit["n"]}


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    ofi_tables, tfi_tables = {}, {}

    for name, rd in ctx.each():
        ofi_rows, tfi_rows, panels = [], [], {}
        for seconds in WINDOWS_S:
            panel = _panel(rd, seconds)
            if len(panel) < MIN_WINDOWS:
                continue
            panels[seconds] = panel
            y_usd = panel.dmid_usd.to_numpy()
            for response, y in (("dmid_usd", y_usd),
                                ("dmid_bps", panel.dmid_bps.to_numpy())):
                ofi_rows.append({
                    "clock": "real time", "window_s": seconds, "response": response,
                    "share_windows_with_a_trade": float((panel.n_trades > 0).mean()),
                    "mean_quote_events_per_window": float(panel.n_quote_events.mean()),
                    **_fit_row(y, panel.ofi_btc, "ofi")})
            # Every TFI definition, plus what it adds on top of OFI.
            for definition in _tfi_series(rd.trades):
                row = {"window_s": seconds, "definition": definition,
                       **_fit_row(y_usd, panel[definition], "tfi")}
                joint = fits.ols_hc3(y_usd,
                                     np.column_stack([panel.ofi_btc, panel[definition]]),
                                     names=["ofi", "tfi"])
                row["r2_ofi_alone"] = _fit_row(y_usd, panel.ofi_btc, "ofi")["r2"]
                row["r2_joint_with_ofi"] = joint["r2"]
                row["incremental_r2_over_ofi"] = joint["r2"] - row["r2_ofi_alone"]
                tfi_rows.append(row)

        for block in EVENT_BLOCKS:
            events = _event_panel(rd, block)
            if events.empty:
                continue
            ofi_rows.append({
                "clock": "quote-event time", "window_s": np.nan,
                "block_quote_events": block, "response": "dmid_usd",
                **_fit_row(events.dmid_usd, events.ofi_btc, "ofi")})

        ofi_table = pd.DataFrame(ofi_rows)
        tfi_table = pd.DataFrame(tfi_rows)
        if ofi_table.empty:
            continue
        # Flag the definition landing closest to ZJ's published BTC value.
        if not tfi_table.empty:
            at10 = tfi_table.loc[tfi_table.window_s == 10.0]
            if len(at10):
                closest = at10.loc[(at10.r2 - ZJ_BTC_TFI_R2_AT_10S).abs().idxmin(),
                                   "definition"]
                tfi_table["closest_to_zj_at_10s"] = tfi_table.definition == closest
        ofi_tables[name], tfi_tables[name] = ofi_table, tfi_table

        ctx.out.save_table(ofi_table, FACT, "realtime_ofi", name,
                           caption=f"Real-time OFI: beta, t-stat and R2 by horizon "
                                   f"(event-time rows for reference) - {rd.label}")
        ctx.out.save_table(tfi_table, FACT, "tfi_definitions", name,
                           caption=f"TFI under four definitions, same windows - {rd.label}")

        usd = ofi_table.loc[(ofi_table.clock == "real time")
                            & (ofi_table.response == "dmid_usd")].set_index("window_s")
        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)

        ax = axes[0]
        ax.errorbar(usd.index, usd.beta, yerr=usd.std_error, fmt="o-", ms=5.5, lw=1.7,
                    capsize=3, color=rd.color)
        ax.axhline(0.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        ax.set_xticks(list(WINDOWS_S), [f"{w:g}" for w in WINDOWS_S])
        pl.finish(ax, "Real-time OFI coefficient", "window length (s)",
                  "USD per BTC of OFI",
                  note="Error bars are HC3 standard errors. ZJ p8 report a "
                       "negative BTC coefficient at 10-60 s; Cont-Kukanov-Stoikov "
                       "predict a positive one.")

        ax = axes[1]
        ax.plot(usd.index, usd.r2, "o-", ms=5.5, lw=1.7, color=rd.color,
                label="real time")
        events = ofi_table.loc[ofi_table.clock == "quote-event time"]
        if len(events):
            for _, row in events.iterrows():
                ax.axhline(row.r2, ls=":", lw=1.0, color="#9CA3AF")
            ax.axhline(events.r2.iloc[0], ls=":", lw=1.0, color="#9CA3AF",
                       label=f"quote-event blocks ({', '.join(str(b) for b in EVENT_BLOCKS)})")
        ax.set_xscale("log")
        ax.set_xticks(list(WINDOWS_S), [f"{w:g}" for w in WINDOWS_S])
        ax.set_ylim(0, None)
        pl.finish(ax, "Explanatory power by horizon", "window length (s)",
                  "R-squared", legend=True,
                  note="A 50-quote block is a few seconds when busy and a minute "
                       "when quiet, which is why an event-time R2 does not sit on "
                       "the clock-time curve.")

        ax = axes[2]
        if not tfi_table.empty:
            for definition, sub in tfi_table.groupby("definition", sort=True):
                sub = sub.set_index("window_s").sort_index()
                marker = "s-" if definition == "signed_volume_btc" else "o-"
                ax.plot(sub.index, sub.r2, marker, ms=4.5, lw=1.5,
                        label=definition.replace("signed_", "").replace("_", " "))
            ax.plot(usd.index, usd.r2, "^--", ms=4.5, lw=1.5, color="#4B5563",
                    label="OFI (reference)")
            ax.axhline(ZJ_BTC_TFI_R2_AT_10S, **pl.LITERATURE_KW)
        ax.set_xscale("log")
        ax.set_xticks(list(WINDOWS_S), [f"{w:g}" for w in WINDOWS_S])
        pl.finish(ax, "TFI by definition", "window length (s)", "R-squared",
                  legend=True,
                  note=f"Dotted line is ZJ p9's published BTC TFI R2 at 10 s "
                       f"({ZJ_BTC_TFI_R2_AT_10S:.3f}). Volume and notional differ "
                       "only by within-window price variation, so they coincide.")
        fig.suptitle(f"Fact 8c - real-time OFI and TFI definitions | {rd.label}",
                     x=0.02, ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "realtime_ofi_tfi", name)

        def _at(seconds, column="r2"):
            row = usd.loc[usd.index == seconds]
            return float(row[column].iloc[0]) if len(row) else np.nan

        metrics += [
            {"metric": "real-time OFI R2 at 10 s", "regime": name,
             "value": _at(10.0), "detail": "ZJ p8 BTC: 0.32%"},
            {"metric": "real-time OFI R2 at 600 s", "regime": name,
             "value": _at(600.0), "detail": "beyond ZJ's grid"},
            {"metric": "real-time OFI beta at 10 s (USD per BTC)", "regime": name,
             "value": _at(10.0, "beta"),
             "detail": f"t = {_at(10.0, 't_stat'):.1f}; ZJ p8 BTC: -0.960 (t=-5.3)"},
        ]
        if not tfi_table.empty:
            at10 = tfi_table.loc[tfi_table.window_s == 10.0].set_index("definition")
            for definition in at10.index:
                metrics.append({
                    "metric": f"TFI R2 at 10 s ({definition})", "regime": name,
                    "value": float(at10.loc[definition, "r2"]),
                    "detail": ("closest to ZJ's 0.166" if at10.loc[definition,
                               "closest_to_zj_at_10s"] else "")})

    if ofi_tables:
        fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
        for name, rd in ctx.each():
            table = ofi_tables.get(name)
            if table is None:
                continue
            usd = table.loc[(table.clock == "real time")
                            & (table.response == "dmid_usd")].set_index("window_s")
            axes[0].plot(usd.index, usd.r2, "o-", ms=5, lw=1.7, color=rd.color,
                         label=rd.label)
            axes[1].errorbar(usd.index, usd.beta, yerr=usd.std_error, fmt="o-",
                             ms=5, lw=1.7, capsize=3, color=rd.color, label=rd.label)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xticks(list(WINDOWS_S), [f"{w:g}" for w in WINDOWS_S])
        axes[1].axhline(0.0, **pl.REFERENCE_KW)
        pl.finish(axes[0], "Real-time OFI R-squared by regime", "window length (s)",
                  "R-squared", legend=True)
        pl.finish(axes[1], "Real-time OFI coefficient by regime", "window length (s)",
                  "USD per BTC of OFI", legend=True)
        fig.suptitle("Fact 8c - real-time OFI by regime", x=0.02, ha="left",
                     fontsize=12.5, fontweight="semibold")
        pl.layout(fig, rect=(0, 0, 1, 0.94))
        ctx.out.save_figure(fig, FACT, "realtime_ofi_tfi", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "ZJ p8 tabulate beta, t and R2 for clock windows of "
                      "1/5/10/30/60 s (BTC R2 0.00-0.59%, negative beta); ZJ p9 put "
                      "BTC TFI near 0.166 at 10 s. Cont, Kukanov & Stoikov (2014) "
                      "predict a positive OFI coefficient falling with depth.",
    }
