"""Verify matched saved predictions and render the full development tables."""
import json

import pandas as pd

from experiments.cohort_outperformance_v1.study import LEVELS, ROOT
from experiments.gat_edge_weight_v1.summarize import block_interval, display
from src.models.ladder_report import within_day_auc


def main():
    runs = ROOT / "runs"
    records = [json.loads(line) for line in (runs / "ledger.jsonl").read_text().splitlines()]
    assert len(records) == len({r["id"] for r in records}) == 16
    matches = {}
    calibrated = {}
    for r in records:
        p = pd.read_parquet(runs / f'{r["id"]}.parquet')
        keys = p[["date", "ticker", "y_true", "direction"]]
        assert p.date.dt.year.max() <= 2025 and len(p) == r["rows"]
        assert p.p_up.between(0, 1).all()
        pd.testing.assert_frame_equal(keys, matches.setdefault((r["year"], r["horizon"]), keys))
        calibrated[r["id"]] = within_day_auc(p)
    print("## All predictor fits\n")
    print("| Run | Fit AUC | Checkpoint-period AUC | Calibration AUC | Assessment AUC | Calibrated AUC | "
          "Accuracy | Majority | MCC | Rank IC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        m = r["metrics"]
        cells = [r["fit_auc"], r["validation_auc"], r["calibration_auc"], m["within_day_auc"],
                 calibrated[r["id"]], m["accuracy"], m["majority"], m["mcc"], r["rank_ic"]]
        print(f'| {r["id"]} | {" | ".join(map(display, cells))} |')
    print("\n## Matched ranking differences\n")
    print("| Run | Comparator | Difference [95% date-block interval] |\n|---|---|---|")
    lookup = {(r["year"], r["horizon"], r["arm"]): r for r in records}
    for r in records:
        if r["arm"] == "logistic":
            continue
        base = "price_lstm" if r["arm"] == "gat_corr" else "logistic"
        ref = lookup[(r["year"], r["horizon"], base)]
        a, b = (dict(enumerate(x["metrics"]["daily_auc"])) for x in (r, ref))
        print(f'| {r["id"]} | {base} | {block_interval(a, b)} |')
    print("\n## Selective accuracy by frozen operating point\n")
    print("| Run | Nominal | Accuracy [95% date-block interval] | Actual | Rows | Dates | "
          "Positive / negative calls | Subset majority | Difference (pp) |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in records:
        for k in LEVELS:
            s = r["selective"][str(k)]
            if s["rows"]:
                low, high = s["block_interval"]
                accuracy = f'{s["accuracy"]:.2%} [{low:.2%}, {high:.2%}]'
                majority = f'{s["majority"]:.2%}'
                diff = f'{(s["accuracy"]-s["majority"])*100:+.2f}'
            else:
                accuracy = majority = diff = "—"
            print(f'| {r["id"]} | {k:.0%} | {accuracy} | {s["coverage"]:.2%} | {s["rows"]} | {s["dates"]} | '
                  f'{s["up_rows"]} / {s["down_rows"]} | {majority} | {diff} |')
    print("\n## Secondary diagnostic on absolute direction\n")
    print("| Run | Accuracy | Majority | MCC | Within-day AUC |\n|---|---:|---:|---:|---:|")
    for r in records:
        m = r["secondary_direction"]
        cells = [m[k] for k in ("accuracy", "majority", "mcc", "within_day_auc")]
        print(f'| {r["id"]} | {" | ".join(map(display, cells))} |')


if __name__ == "__main__":
    main()
