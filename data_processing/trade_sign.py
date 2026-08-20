import numpy as np


def infer_trade_sign(prices, prevailing_mid):
    """Quote test followed by tick rule; returns signs and provenance."""
    prices, prevailing_mid = np.asarray(prices), np.asarray(prevailing_mid)
    signs = np.sign(prices - prevailing_mid).astype(int)
    source = np.where(signs != 0, "QUOTE_TEST", "TICK_RULE")
    tick = np.sign(np.diff(prices, prepend=prices[0]))
    last = 1
    for i in range(len(signs)):
        if signs[i] == 0:
            if tick[i] != 0: last = int(tick[i])
            signs[i] = last
    return signs, source

