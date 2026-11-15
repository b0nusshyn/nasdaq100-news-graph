"""Forward returns and up/down labels on NASDAQ sessions (issue #15)."""

from __future__ import annotations

import argparse

import pandas as pd

from src.core.io import read_table, write_table
from src.core.schema import HORIZONS, conform, validate
from src.data.build_index_history import trading_days
from src.data.validation import checked_prices


def make_labels(prices: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Use the exact h-th future NASDAQ session, never jump across missing prices.

    Missing target closes and non-session source rows produce null returns/labels.
    Unrequested frozen-contract horizons remain null; row order matches input.
    """
    if not horizons or len(set(horizons)) != len(horizons) or set(horizons) - set(HORIZONS):
        raise ValueError(f"horizons must be a nonempty unique subset of {HORIZONS}")
    frame = checked_prices(prices)
    out = frame[["date", "ticker"]].copy()
    for horizon in HORIZONS:
        out[f"ret_{horizon}d"] = float("nan")
        out[f"y_{horizon}d"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    if not frame.empty:
        sessions = trading_days(frame.date.min(), frame.date.max() + pd.Timedelta(days=30))
        for _, group in frame.groupby("ticker"):
            positions = sessions.get_indexer(group.date)
            closes = group.set_index("date").close
            for horizon in horizons:
                valid = positions >= 0
                targets = sessions[positions[valid] + horizon]
                source = group.loc[valid]
                future = closes.reindex(targets).to_numpy()
                returns = pd.Series(future / source.close.to_numpy() - 1, index=source.index)
                out.loc[source.index, f"ret_{horizon}d"] = returns
                y = returns.gt(0).astype("Int8").mask(returns.isna())
                out.loc[source.index, f"y_{horizon}d"] = y
    out = conform(out, "labels")
    validate(out, "labels")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--horizons", type=int, choices=HORIZONS, nargs="+", default=list(HORIZONS))
    args = p.parse_args()
    labels = make_labels(read_table("prices"), tuple(args.horizons))
    path = write_table(labels, "labels")
    print(f"{path}: {len(labels)} rows; " + ", ".join(
        f"{h}d: {labels[f'y_{h}d'].notna().sum()} labelled" for h in args.horizons
    ))


if __name__ == "__main__":
    main()
