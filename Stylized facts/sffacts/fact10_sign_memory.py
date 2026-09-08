"""
Stylized fact 10 (P0-C) - Long memory of order signs.

Empirical claim
    Buy/sell signs are strongly persistent.  Lillo & Farmer (2004) find the sign
    autocorrelation decays approximately as a power law, C(l) ~ l^-gamma, with
    gamma around 0.6 on the LSE - i.e. long memory (gamma < 1), usually
    attributed to parent-order splitting.

Why it matters for the simulator
    An IID buy/sell Poisson taker is too simple: liquidations interact with
    persistent organic flow.  The source document is explicit that the venue's
    own exponent should be estimated rather than 0.6 copied across.

Method
    The sign series is taken at two granularities, because they answer different
    questions and the literature uses both:

    * **aggressive orders** - one sign per marketable order (a burst of fills
      sharing a packet and a side).  This is the "market order" series and the
      closest analogue of the Lillo-Farmer construction.
    * **individual fills** - one sign per printed trade.  This inflates
      persistence mechanically, because one sweep prints many same-signed fills,
      so it is reported alongside rather than instead.

    gamma is estimated by OLS on log C(l) vs log l over a fitting window that
    excludes the first few lags (where the tick-level mechanics dominate) and
    the far tail (where the estimate is noise-dominated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact10"
FACT_ID = 10
PRIORITY = "P0-C"
TITLE = "Long memory of order signs"

LITERATURE_GAMMA = 0.6
#: Lags used for the power-law fit.  Short lags are contaminated by mechanical
#: splitting within a single sweep; very long lags are dominated by estimation
#: noise once C(l) approaches the ~1/sqrt(N) noise floor.
FIT_LAG_MIN = 10
FIT_LAG_MAX = 1000


def _sign_acf(sign: np.ndarray, max_lag: int) -> pd.DataFrame:
    lags = np.arange(1, max_lag + 1)
    values = st.acf(sign, max_lag)
    return pd.DataFrame({"lag": lags[: values.size], "acf": values})


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    order_acfs, fill_acfs, fits = {}, {}, {}

    for name, rd in ctx.each():
        order_sign = rd.trades["sign"].to_numpy(np.float64)
        fill_sign = rd.fills["sign"].to_numpy(np.float64)

        max_lag = min(cfg.ACF_MAX_LAG_EVENT, max(order_sign.size // 20, 10))
        a_order = _sign_acf(order_sign, max_lag)
        a_fill = _sign_acf(fill_sign, min(cfg.ACF_MAX_LAG_EVENT, max(fill_sign.size // 20, 10)))
        order_acfs[name], fill_acfs[name] = a_order, a_fill

        window = a_order[(a_order.lag >= FIT_LAG_MIN) & (a_order.lag <= FIT_LAG_MAX)]
        fit = st.fit_loglog(window.lag.to_numpy(), window.acf.to_numpy())
        fit_fill_window = a_fill[(a_fill.lag >= FIT_LAG_MIN) & (a_fill.lag <= FIT_LAG_MAX)]
        fit_fill = st.fit_loglog(fit_fill_window.lag.to_numpy(), fit_fill_window.acf.to_numpy())
        fits[name] = fit

        table = a_order.copy()
        table["series"] = "aggressive orders"
        tf = a_fill.copy()
        tf["series"] = "individual fills"
        ctx.out.save_table(pd.concat([table, tf], ignore_index=True), FACT,
                           "sign_autocorrelation", name,
                           caption=f"Order-sign autocorrelation - {rd.label}")

        noise = 1.0 / np.sqrt(order_sign.size)
        fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
        ax = axes[0]
        ok = a_order.acf > 0
        ax.plot(a_order.lag[ok], a_order.acf[ok], ".", ms=3, alpha=0.55,
                label="aggressive orders")
        ok_f = a_fill.acf > 0
        ax.plot(a_fill.lag[ok_f], a_fill.acf[ok_f], ".", ms=3, alpha=0.35,
                color="#9CA3AF", label="individual fills")
        if np.isfinite(fit["exponent"]):
            xs = np.geomspace(FIT_LAG_MIN, FIT_LAG_MAX, 50)
            ax.plot(xs, fit["prefactor"] * xs ** fit["exponent"], lw=2.0,
                    color="#B45309",
                    label=f"fit: gamma={-fit['exponent']:.3f} (R2={fit['r2']:.2f})")
        xs = np.geomspace(1, max_lag, 50)
        ax.plot(xs, 0.5 * xs ** (-LITERATURE_GAMMA), **pl.LITERATURE_KW)
        ax.axhline(noise, color="#9CA3AF", lw=0.9, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Sign autocorrelation (log-log)", "lag (events)", "C(l)",
                  legend=True,
                  note=f"Dotted green: slope of the Lillo-Farmer gamma={LITERATURE_GAMMA} "
                       f"reference. Grey dotted: 1/sqrt(N) noise floor.")

        ax = axes[1]
        head = a_order[a_order.lag <= 60]
        ax.bar(head.lag, head.acf, width=0.8, color="#6B8FBF")
        ax.axhline(noise, color="#9CA3AF", lw=0.9, ls=":")
        ax.axhline(0.0, color="#4B5563", lw=0.8)
        pl.finish(ax, "First 60 lags", "lag (aggressive orders)", "C(l)")
        fig.suptitle(f"Fact 10 - long memory of order signs | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig, rect=(0, 0, 1, 0.94))
        ctx.out.save_figure(fig, FACT, "sign_memory", name)

        metrics += [
            {"metric": "sign-ACF decay exponent gamma (aggressive orders)",
             "regime": name, "value": -fit["exponent"],
             "detail": f"R2={fit['r2']:.3f}, lags {FIT_LAG_MIN}-{FIT_LAG_MAX}; "
                       f"literature ~{LITERATURE_GAMMA}"},
            {"metric": "sign-ACF decay exponent gamma (individual fills)",
             "regime": name, "value": -fit_fill["exponent"],
             "detail": f"R2={fit_fill['r2']:.3f}"},
            {"metric": "C(1) sign autocorrelation", "regime": name,
             "value": float(a_order.acf.iloc[0]), "detail": ""},
            {"metric": "C(100) sign autocorrelation", "regime": name,
             "value": float(a_order.acf.iloc[99]) if len(a_order) > 99 else np.nan,
             "detail": "long memory if still well above the noise floor"},
            {"metric": "long memory (gamma < 1)", "regime": name,
             "value": 1.0 if -fit["exponent"] < 1.0 else 0.0,
             "detail": "1 = yes"},
        ]

    fig, ax = pl.new_figure(1, 1, width=7.4, height=4.8)
    for name, rd in ctx.each():
        a = order_acfs[name]
        ok = a.acf > 0
        ax.plot(a.lag[ok], a.acf[ok], ".", ms=3, alpha=0.5, color=rd.color)
        fit = fits[name]
        if np.isfinite(fit["exponent"]):
            xs = np.geomspace(FIT_LAG_MIN, FIT_LAG_MAX, 50)
            ax.plot(xs, fit["prefactor"] * xs ** fit["exponent"], lw=2.2,
                    color=rd.color, label=f"{rd.label}: gamma={-fit['exponent']:.3f}")
    xs = np.geomspace(1, FIT_LAG_MAX, 50)
    ax.plot(xs, 0.5 * xs ** (-LITERATURE_GAMMA), **pl.LITERATURE_KW,
            label=f"Lillo-Farmer slope (gamma={LITERATURE_GAMMA})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    pl.finish(ax, "Fact 10 - order-sign long memory by regime", "lag (aggressive orders)",
              "C(l)", legend=True)
    pl.layout(fig, rect=(0, 0, 1, 1.0))
    ctx.out.save_figure(fig, FACT, "sign_memory", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": f"Lillo & Farmer (2004): C(l) ~ l^-gamma with gamma ~ "
                      f"{LITERATURE_GAMMA} on the LSE; long memory if gamma < 1.",
    }
