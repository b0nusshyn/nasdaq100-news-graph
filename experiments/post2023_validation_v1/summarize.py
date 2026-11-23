"""Render matched developmental results and paired date-block uncertainty."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def interval(values):
    """Circular 10-session block bootstrap to preserve overlapping outcomes."""
    n = len(values)
    starts = np.random.default_rng(42).integers(n, size=(2500, (n + 9) // 10, 1))
    indices = ((starts + np.arange(10)) % n).reshape(2500, -1)[:, :n]
    draws = np.nanmean(values[indices], axis=1)
    return [float(x) for x in np.nanquantile(draws, [.025, .975])]


def fmt(x):
    return "Undefined" if x is None else f"{x:.4f}"


def main():
    header, *rows = [json.loads(s) for s in (ROOT / "development_runs.jsonl").read_text().splitlines()]
    key = {(r["year"], r["horizon"], r["arm"]): r for r in rows}
    print("| Window | Horizon | Model | Accuracy | Majority | MCC | "
          "Within-day AUC | Rank IC | AUC difference vs logistic [95% interval] |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        ref = key[r["year"], r["horizon"], "stock_logistic"]
        diff = np.array(r["daily_auc"], dtype=float) - np.array(ref["daily_auc"], dtype=float)
        low, high = interval(diff)
        m = r["metrics"]
        paired = "Reference" if r is ref else f"{np.nanmean(diff):+.4f} [{low:+.4f}, {high:+.4f}]"
        print(f'| {r["year"]} | {r["horizon"]}d | {r["arm"]} | {m["accuracy"]:.4f} | '
              f'{r["majority"]:.4f} | {fmt(m["mcc"])} | {r["within_day_auc"]:.4f} | {fmt(r["rank_ic"])} | {paired} |')
    print("\n### Selective accuracy")
    print("\n| Window | Horizon | Model | Accuracy [95% interval] | Coverage | Rows | Dates | "
          "Up share | Subset majority |")
    print("|---|---:|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        s = r["selective_20"]
        low, high = s["block_interval"]
        print(f'| {r["year"]} | {r["horizon"]}d | {r["arm"]} | {s["accuracy"]:.2%} [{low:.2%}, {high:.2%}] | '
              f'{s["coverage"]:.2%} | {s["rows"]} | {s["dates"]} | {s["up_share"]:.2%} | {s["majority"]:.2%} |')
    print(f'\nThe fixed-30 cohort is {", ".join(header["cohort"])}.')


if __name__ == "__main__":
    main()
