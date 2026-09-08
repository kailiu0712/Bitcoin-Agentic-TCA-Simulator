"""
Stylized fact 8b (P0-C) - OFI vs TFI over clock windows, and beta ~ 1/depth.

Source
    ``Stylied facts ZJ.pdf`` p8, p9 and p10.  ZJ regress the mid-price change
    on order-flow imbalance over **clock** windows of 1, 5, 10, 30 and 60
    seconds and tabulate R-squared, the slope and its t-statistic; then compare
    that against trade-flow imbalance and against a joint model; then plot the
    impact coefficient against the reciprocal of mean depth.

    ``fact08_ofi`` already builds the exact event-by-event Cont-Kukanov-Stoikov
    OFI, but aggregates it over blocks of 50 quote events and never constructs
    TFI at all.  A block of 50 quote events is not a clock window, so its
    numbers are not comparable with ZJ's table; and without TFI the central
    claim on p9 - that these markets are trade-driven rather than quote-driven -
    cannot be checked either way.

Definitions (all three models share one sample and one window partition)
    Over window w spanning quote rows [i0 .. i1]:

        OFI_w   = sum of e_n for n in [i0 .. i1]
        TFI_w   = sum of eps_j * q_j over aggressive orders j in w
        dmid_w  = m[i1] - m[i0 - 1]

    The ``i0 - 1`` matters.  ``e_n`` is defined on the transition from quote row
    n-1 to row n, so a window holding rows i0..i1 accounts for the price change
    from row i0-1 to row i1.  Anchoring dmid at row i0 instead - which is the
    natural-looking choice - drops exactly one price change per window and
    biases the slope toward zero.  ``fact08`` has that off-by-one; it is fixed
    here, which is one reason the two modules' coefficients differ.

Reading the comparison
    R-squared is *contemporaneous* explanatory power, not prediction and not a
    causal decomposition.  TFI and OFI are also not independent: a trade that
    consumes the queue changes the best size, so the same event contributes to
    both. A higher TFI R-squared therefore says the price moves when trades
    print, not that trades cause the move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import fits
from sfcore import plotting as pl
from sfcore import stats as st
from sffacts.fact08_ofi import _ofi_series

FACT = "fact08b"
FACT_ID = 8
PRIORITY = "P0-C"
TITLE = "OFI vs TFI over clock windows; impact coefficient vs depth"

#: ZJ p8/p9 window grid, seconds.
WINDOWS_S: tuple[float, ...] = (1.0, 5.0, 10.0, 30.0, 60.0)

#: Window used for the depth-conditioned slope on ZJ p10.
DEPTH_WINDOW_S = 10.0
N_DEPTH_GROUPS = 5


def _window_panel(rd, seconds: float) -> pd.DataFrame:
    """One row per non-overlapping clock window, with all three regressors."""
    quotes = rd.quotes
    qts = quotes["recv_ns"].to_numpy(np.int64)
    mid = rd.q_mid
    ofi = _ofi_series(rd)
    l1_depth = (quotes["bid_qty"].to_numpy(np.float64)
                + quotes["ask_qty"].to_numpy(np.float64))
    band_depth = (quotes["bid_depth"].to_numpy(np.float64)
                  + quotes["ask_depth"].to_numpy(np.float64))

    trades = rd.trades
    tts = trades["recv_ns"].to_numpy(np.int64)
    sign = trades["sign"].to_numpy(np.float64)
    signed_qty = sign * trades["qty"].to_numpy(np.float64)
    # Signed trade COUNT as well as signed volume.  BTC order sizes are heavy
    # enough (p99/median ~ 700x, top 1% of orders ~ 40% of volume) that a
    # volume-weighted regressor is dominated by a handful of prints, while a
    # count weights every taker decision equally.  fact08c shows count is the
    # strongest of four candidate definitions and the closest to ZJ's published
    # figure, so it is the one the R-squared panel plots.
    signed_count = sign

    step = int(seconds * 1e9)
    first = (qts[0] // step) * step
    edges = np.arange(first, qts[-1] + step, step, dtype=np.int64)
    qi = np.searchsorted(qts, edges, side="left")
    ti = np.searchsorted(tts, edges, side="left")

    cum_ofi = np.concatenate([[0.0], np.cumsum(ofi)])
    cum_tfi = np.concatenate([[0.0], np.cumsum(signed_qty)])
    cum_cnt = np.concatenate([[0.0], np.cumsum(signed_count)])
    cum_l1 = np.concatenate([[0.0], np.cumsum(l1_depth)])
    cum_band = np.concatenate([[0.0], np.cumsum(band_depth)])

    start, stop = qi[:-1], qi[1:]
    # A window needs at least one quote event, and a predecessor row for the
    # opening price, so windows that start at the very first row are dropped.
    ok = (stop > start) & (start >= 1)
    start, stop = start[ok], stop[ok]
    n_quotes = (stop - start).astype(np.float64)
    return pd.DataFrame({
        "window_start_ns": edges[:-1][ok],
        "ofi_btc": cum_ofi[stop] - cum_ofi[start],
        "tfi_btc": cum_tfi[ti[1:][ok]] - cum_tfi[ti[:-1][ok]],
        "tfi_count": cum_cnt[ti[1:][ok]] - cum_cnt[ti[:-1][ok]],
        "dmid_usd": mid[stop - 1] - mid[start - 1],
        "dmid_bps": (mid[stop - 1] - mid[start - 1]) / mid[start - 1] * 1e4,
        "mean_l1_depth_btc": (cum_l1[stop] - cum_l1[start]) / n_quotes,
        "mean_band_depth_btc": (cum_band[stop] - cum_band[start]) / n_quotes,
        "n_quote_events": n_quotes,
        "n_trades": (ti[1:][ok] - ti[:-1][ok]).astype(np.float64),
    })


#: Which TFI column each model label uses.  "TFI"/"OFI+TFI" keep the original
#: signed-volume definition so the existing rows and the slope panel are
#: unchanged; the "_count" variants are additional.
TFI_COLUMN = {"TFI": "tfi_btc", "OFI+TFI": "tfi_btc",
              "TFI_count": "tfi_count", "OFI+TFI_count": "tfi_count"}


def _three_models(panel: pd.DataFrame, response: str) -> list[dict]:
    """OFI-only, TFI-only and joint regressions on one identical sample.

    Run for both TFI definitions.  R-squared is unit-free, so volume and count
    are directly comparable; the slopes are not (USD per BTC versus USD per net
    trade), which is why the coefficient panel keeps a single definition.
    """
    y = panel[response].to_numpy(np.float64)
    ofi = panel["ofi_btc"].to_numpy(np.float64)
    tfi = panel["tfi_btc"].to_numpy(np.float64)
    cnt = panel["tfi_count"].to_numpy(np.float64)
    rows = []
    for label, design, names in (
        ("OFI", ofi[:, None], ["ofi"]),
        ("TFI", tfi[:, None], ["tfi"]),
        ("OFI+TFI", np.column_stack([ofi, tfi]), ["ofi", "tfi"]),
        ("TFI_count", cnt[:, None], ["tfi"]),
        ("OFI+TFI_count", np.column_stack([ofi, cnt]), ["ofi", "tfi"]),
    ):
        fit = fits.ols_hc3(y, design, names=names)
        row = {"model": label, "r2": fit["r2"], "n": fit["n"]}
        for j, term in enumerate(names, start=1):
            row[f"beta_{term}"] = float(fit["params"][j])
            row[f"se_{term}"] = float(fit["se"][j])
            row[f"t_{term}"] = float(fit["t"][j])
        rows.append(row)
    return rows


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    model_tables, depth_tables = {}, {}

    for name, rd in ctx.each():
        rows, panels = [], {}
        for seconds in WINDOWS_S:
            panel = _window_panel(rd, seconds)
            if len(panel) < 100:
                continue
            panels[seconds] = panel
            for response in ("dmid_usd", "dmid_bps"):
                for entry in _three_models(panel, response):
                    rows.append({"window_s": seconds, "response": response,
                                 "windows": len(panel),
                                 "share_windows_with_a_trade":
                                     float((panel["n_trades"] > 0).mean()),
                                 **entry})
        models = pd.DataFrame(rows)
        model_tables[name] = models
        if models.empty:
            continue
        ctx.out.save_table(models, FACT, "ofi_tfi_models", name,
                           caption=f"Clock-window OFI / TFI / joint regressions - {rd.label}")

        # ZJ p10: slope against 1/depth, within depth quintiles at 10 s.
        panel = panels.get(DEPTH_WINDOW_S)
        depth_rows = []
        if panel is not None:
            groups = st.quantile_bins(panel["mean_l1_depth_btc"].to_numpy(), N_DEPTH_GROUPS)
            for g in range(N_DEPTH_GROUPS):
                sub = panel.loc[groups == g]
                if len(sub) < 200:
                    continue
                fit = fits.ols_hc3(sub["dmid_usd"].to_numpy(),
                                   sub["ofi_btc"].to_numpy()[:, None], names=["ofi"])
                mean_depth = float(sub["mean_l1_depth_btc"].mean())
                depth_rows.append({
                    "quintile": f"Q{g + 1}", "mean_l1_depth_btc": mean_depth,
                    "inverse_depth": 1.0 / mean_depth if mean_depth > 0 else np.nan,
                    "beta_usd_per_btc": float(fit["params"][1]),
                    "se": float(fit["se"][1]), "t": float(fit["t"][1]),
                    "r2": fit["r2"], "n": fit["n"]})
        depth = pd.DataFrame(depth_rows)
        depth_tables[name] = depth
        if len(depth):
            ctx.out.save_table(depth, FACT, "beta_vs_inverse_depth", name,
                               caption=f"OFI slope by L1-depth quintile at "
                                       f"{DEPTH_WINDOW_S:g}s - {rd.label}")

        usd = models.loc[models.response == "dmid_usd"]
        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)

        ax = axes[0]
        positions = np.arange(len(WINDOWS_S))
        for offset, (label, colour, legend_label) in zip(
                (-0.27, 0.0, 0.27),
                (("OFI", "#2E7D32", "OFI"),
                 ("TFI_count", "#E67E22", "TFI (signed trade count)"),
                 ("OFI+TFI_count", "#2C6FBB", "OFI + TFI (count)"))):
            sub = usd.loc[usd.model == label].set_index("window_s").reindex(WINDOWS_S)
            ax.bar(positions + offset, sub["r2"].to_numpy(), width=0.26,
                   color=colour, label=legend_label)
        # The signed-volume TFI, outlined only, so switching definition is
        # visible rather than silent.
        volume = usd.loc[usd.model == "TFI"].set_index("window_s").reindex(WINDOWS_S)
        ax.bar(positions, volume["r2"].to_numpy(), width=0.26, facecolor="none",
               edgecolor="#7F5539", lw=1.1, linestyle="--",
               label="TFI (signed volume)")
        ax.set_xticks(positions, [f"{w:g}s" for w in WINDOWS_S])
        # Headroom so the four-entry legend clears the bars.
        tallest = usd.loc[usd.model == "OFI+TFI_count", "r2"].max()
        if np.isfinite(tallest):
            ax.set_ylim(0, tallest * 1.5)
        pl.finish(ax, "Explanatory power: OFI vs TFI", "window length",
                  "R-squared", legend=True,
                  note="TFI bars use signed trade COUNT, the strongest of the four "
                       "definitions in fact08c; the dashed outline is the "
                       "signed-volume version. Contemporaneous fit on the same "
                       "windows, not prediction. ZJ p9 find TFI far ahead of OFI.")

        ax = axes[1]
        for label, colour in (("OFI", "#2E7D32"), ("TFI", "#E67E22")):
            sub = usd.loc[usd.model == label].set_index("window_s").reindex(WINDOWS_S)
            term = "ofi" if label == "OFI" else "tfi"
            ax.errorbar(WINDOWS_S, sub[f"beta_{term}"], yerr=sub[f"se_{term}"],
                        fmt="o-", ms=5, lw=1.6, capsize=3, color=colour, label=label)
        ax.axhline(0.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        pl.finish(ax, "Slope by window length", "window length (s)",
                  "USD per BTC of imbalance", legend=True,
                  note="ZJ p8 report a negative, significant OFI slope for BTC at "
                       "10-60 s, which is the opposite of the Cont-Kukanov-Stoikov "
                       "sign.")

        ax = axes[2]
        if len(depth):
            ax.errorbar(depth["inverse_depth"], depth["beta_usd_per_btc"],
                        yerr=depth["se"], fmt="o", ms=6, capsize=3,
                        color="#E67E22", ecolor="#9CA3AF")
            for _, row in depth.iterrows():
                ax.annotate(row["quintile"], (row["inverse_depth"], row["beta_usd_per_btc"]),
                            textcoords="offset points", xytext=(0, 8), fontsize=8,
                            ha="center", color="#4B5563")
            if len(depth) >= 3:
                line = st.ols(depth["beta_usd_per_btc"].to_numpy(),
                              depth["inverse_depth"].to_numpy()[:, None])
                grid = np.linspace(depth["inverse_depth"].min(),
                                   depth["inverse_depth"].max(), 20)
                ax.plot(grid, line["beta"][0] + line["beta"][1] * grid,
                        ls="--", lw=1.6, color="#2C6FBB",
                        label=f"slope {line['beta'][1]:.3g}")
        ax.axhline(0.0, **pl.REFERENCE_KW)
        pl.finish(ax, f"Impact coefficient vs 1/depth ({DEPTH_WINDOW_S:g}s)",
                  "1 / mean L1 depth (1/BTC)", "OFI slope (USD per BTC)",
                  legend=len(depth) >= 3,
                  note="Q1 is the thinnest quintile. Cont-Kukanov-Stoikov predict "
                       "a rising line; ZJ found it only for SOL.")
        fig.suptitle(f"Fact 8b - OFI vs TFI | {rd.label}", x=0.02, ha="left",
                     fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "ofi_tfi", name)

        def _r2(label, seconds):
            row = usd.loc[(usd.model == label) & (usd.window_s == seconds)]
            return float(row["r2"].iloc[0]) if len(row) else np.nan

        metrics += [
            {"metric": "R2 of OFI alone at 10 s", "regime": name,
             "value": _r2("OFI", 10.0), "detail": "ZJ p8 BTC: 0.32%"},
            {"metric": "R2 of TFI alone at 10 s (signed volume)", "regime": name,
             "value": _r2("TFI", 10.0), "detail": "ZJ p9 BTC: ~0.166"},
            {"metric": "R2 of TFI alone at 10 s (signed trade count)", "regime": name,
             "value": _r2("TFI_count", 10.0),
             "detail": "strongest definition; plotted in the R2 panel"},
            {"metric": "R2 of the joint model at 10 s (count TFI)", "regime": name,
             "value": _r2("OFI+TFI_count", 10.0),
             "detail": "incremental power of the count TFI given OFI"},
            {"metric": "OFI slope at 10 s (USD per BTC)", "regime": name,
             "value": float(usd.loc[(usd.model == "OFI") & (usd.window_s == 10.0),
                                    "beta_ofi"].iloc[0])
                      if len(usd.loc[(usd.model == "OFI") & (usd.window_s == 10.0)])
                      else np.nan,
             "detail": "ZJ p8 BTC: -9.60e-01 (t=-5.3)"},
        ]

    fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
    for name, rd in ctx.each():
        models = model_tables.get(name)
        if models is None or models.empty:
            continue
        usd = models.loc[models.response == "dmid_usd"]
        for label, style, tag in (("OFI", "o--", "OFI"),
                                  ("TFI_count", "s-", "TFI (count)")):
            sub = usd.loc[usd.model == label].set_index("window_s").reindex(WINDOWS_S)
            axes[0].plot(WINDOWS_S, sub["r2"], style, ms=5, lw=1.7, color=rd.color,
                         alpha=1.0 if label == "TFI_count" else 0.55,
                         label=f"{rd.label} - {tag}")
        depth = depth_tables.get(name)
        if depth is not None and len(depth):
            axes[1].plot(depth["inverse_depth"], depth["beta_usd_per_btc"], "o-",
                         ms=5, lw=1.7, color=rd.color, label=rd.label)
    axes[0].set_xscale("log")
    pl.finish(axes[0], "OFI vs TFI explanatory power by regime",
              "window length (s)", "R-squared", legend=True)
    axes[1].axhline(0.0, **pl.REFERENCE_KW)
    pl.finish(axes[1], f"Impact coefficient vs 1/depth ({DEPTH_WINDOW_S:g}s)",
              "1 / mean L1 depth (1/BTC)", "OFI slope (USD per BTC)", legend=True)
    fig.suptitle("Fact 8b - OFI vs TFI by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "ofi_tfi", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "ZJ p8-p10 (2026-08-25): OFI R2 of 0-2% at 1-60 s, TFI far "
                      "higher, and beta ~ 1/depth linear only for SOL. Cont, "
                      "Kukanov & Stoikov (2014) predict a positive OFI slope "
                      "falling with depth.",
    }
