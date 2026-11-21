"""Render matched assessment results and temporal block intervals."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent / "runs"


def block_interval(left, right):
    assert left.keys() == right.keys()
    difference = np.array(list(left.values()), dtype=float) - np.array([right[k] for k in left], dtype=float)
    n = len(difference)
    starts = np.random.default_rng(42).integers(n, size=(2500, (n + 9) // 10, 1))
    blocks = ((starts + np.arange(10)) % n).reshape(2500, -1)[:, :n]
    low, high = np.quantile(np.nanmean(difference[blocks], axis=1), [.025, .975])
    return f"{np.nanmean(difference):+.4f} [{low:+.4f}, {high:+.4f}]"


def display(value):
    return "—" if value is None else f"{value:.4f}"


def main():
    runs = [json.loads(line) for line in (ROOT / "ledger.jsonl").read_text().splitlines()]
    records = {(r["fold"], r["horizon"], r["seed"], r["arm"]): r for r in runs}
    print("| Run | Rows | Fit AUC | Val AUC | Assess AUC | Fit–assess gap | Acc. | Majority | MCC | F1 | Up % | Step |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        m = r["metrics"]
        chosen = next(h for h in r["history"] if h["step"] == r["best_step"])
        probe = chosen["train_probe_within_day_auc"]
        values = [probe, r["best_val_within_day_auc"], m["within_day_auc"], probe - m["within_day_auc"],
                  m["accuracy"], m["majority"], m["mcc"], m["f1"]]
        scores = " | ".join(map(display, values))
        print(f'| {r["id"]} | {r["rows"]} | {scores} | {m["share_predicted_up"]:.1%} | '
              f'{r["best_step"]}/{r["steps"]} |')
    print("\n| Run | Comparator | Paired AUC difference [95% block interval] |\n|---|---|---|")
    for (fold, h, seed, arm), r in records.items():
        comparator = {"original": "price", "signed": "original", "centered": "signed"}.get(arm)
        if prior := records.get((fold, h, seed, comparator)):
            diff = block_interval(r["metrics"]["daily_auc"], prior["metrics"]["daily_auc"])
            print(f'| {r["id"]} | {comparator} | {diff} |')
    print("\n| Window | Fit | Validation | Assessment |\n|---|---|---|---|")
    for fold in sorted({r["fold"] for r in runs}):
        r = next(r for r in runs if r["fold"] == fold)
        ranges = [" to ".join(r["dates"][stage][:2]) for stage in ("fit", "validation", "assessment")]
        print(f'| {fold} | {" | ".join(ranges)} |')


if __name__ == "__main__":
    main()
