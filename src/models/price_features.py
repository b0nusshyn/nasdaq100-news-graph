"""Optional, close-time features computed on the shared trading-session calendar."""

import numpy as np
import pandas as pd

EXTENDED_PRICE = ("momentum_5d", "momentum_20d", "volatility_5d", "relative_momentum_5d", "relative_momentum_20d")


def compute_extended_price(prices: pd.DataFrame) -> pd.DataFrame:
    """Use observed adjacent closes and the contemporaneous equal-weight member return.

    Momentum is cumulative log return. A missing daily return invalidates its trailing
    window. Market-relative momentum subtracts the sum of daily market log returns.
    """
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    members = prices.pivot(index="date", columns="ticker", values="in_index").reindex_like(close)
    returns = np.log(close).diff()
    market = returns.where(members.eq(True)).mean(axis=1)
    momentum_5 = returns.rolling(5).sum()
    momentum_20 = returns.rolling(20).sum()
    features = (
        momentum_5, momentum_20, returns.rolling(5).std(),
        momentum_5.sub(market.rolling(5).sum(), axis=0),
        momentum_20.sub(market.rolling(20).sum(), axis=0),
    )
    keys = pd.MultiIndex.from_frame(prices[["date", "ticker"]])
    d, t = close.index.get_indexer(keys.get_level_values(0)), close.columns.get_indexer(keys.get_level_values(1))
    return pd.DataFrame({name: values.to_numpy()[d, t] for name, values in zip(EXTENDED_PRICE, features, strict=True)},
                        index=prices.index)
