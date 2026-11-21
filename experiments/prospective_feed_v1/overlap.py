"""Compare complete historical panels without computing labels or model scores."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.market_direction_v1.study import features
from experiments.prospective_cohort_v1.live import calendar, digest, encode, load_freeze, write_once

COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def aligned(frame, dates, cohort):
    frame = frame[["date", "ticker", *COLUMNS]].copy()
    frame.date = pd.to_datetime(frame.date)
    frame = frame.set_index(["date", "ticker"])
    expected = pd.MultiIndex.from_product([dates, cohort], names=frame.index.names)
    if not frame.index.is_unique or len(frame) != len(expected) or not frame.index.isin(expected).all():
        raise ValueError("panel must contain exactly one row per expected session and security")
    values = frame[COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("bars must be finite and positive")
    if ((frame.high < frame[["open", "close"]].max(axis=1)) |
        (frame.low > frame[["open", "close"]].min(axis=1))).any():
        raise ValueError("invalid OHLC bounds")
    return frame.reindex(expected).astype(float)


def errors(a, b):
    d = abs(a - b).to_numpy()
    valid = np.isfinite(d)
    if not valid.any():
        raise ValueError("no comparable values")
    return {"n": int(valid.sum()), "missingness_disagreements": int((a.isna() != b.isna()).sum()),
            "median_abs": float(np.median(d[valid])), "max_abs": float(d[valid].max())}


def compare(reference, candidate, dates, cohort):
    a, b = [aligned(p, dates, cohort) for p in (reference, candidate)]
    adjusted = []
    for p in (a, b):
        q = p.copy()
        q.loc[:, COLUMNS[:4]] = p[COLUMNS[:4]].mul(p.adj_close / p.close, axis=0)
        adjusted.append(q)
    x, y = adjusted
    start = dates[0]
    xf, yf = [features(p.reset_index().assign(in_index=True))[0].set_index(["date", "ticker"])
              for p in adjusted]
    return {
        "rows": len(a),
        "adjusted_daily_close_return": errors(x.close.groupby("ticker").pct_change(),
                                                y.close.groupby("ticker").pct_change()),
        "normalized_adjusted_ohlc": {k: errors(x[k] / x.loc[start, "close"].reindex(x.index, level="ticker"),
                                              y[k] / y.loc[start, "close"].reindex(y.index, level="ticker"))
                                    for k in COLUMNS[:4]},
        "volume_exact_rows": int(a.volume.eq(b.volume).sum()),
        "volume_relative_error": errors(a.volume / b.volume, b.volume / b.volume),
        "frozen_features": {k: errors(xf[k].astype(float), yf[k].astype(float)) for k in xf},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--end", required=True)
    args = p.parse_args()
    cfg, frozen = load_freeze()
    sessions, end = calendar(args.end)
    dates = sessions.index[end+1-cfg["warmup_sessions"]:end+1]
    panels = [pd.read_parquet(path) for path in (args.reference, args.candidate)]
    report = compare(*panels, dates, cfg["cohort"])
    report.update(freeze_sha256=frozen, end=args.end,
                  first=dates[0].date().isoformat(),
                  reference_sha256=digest(args.reference.read_bytes()),
                  candidate_sha256=digest(args.candidate.read_bytes()), approved_source=False)
    write_once(args.output, encode(report))


if __name__ == "__main__":
    main()
