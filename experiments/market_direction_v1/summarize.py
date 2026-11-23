"""Produce exact selective subgroup and paired ranking diagnostics without fitting models."""
import json

import numpy as np
import pandas as pd

from experiments.market_direction_v1.study import ROOT, selective
from src.core.io import read_predictions
from src.eval.run_baselines import load_inputs
from src.models.ladder_report import auc


def main():
    use = {}
    all_rows = []
    sector = pd.read_csv("src/graph/gics.csv").set_index("ticker").gics_sector
    subgroup = []
    points = []
    for follow, name in [(False, "runs.jsonl"), (True, "followup_runs.jsonl")]:
        rows = [json.loads(s) for s in (ROOT / name).read_text().splitlines()][1:]
        all_rows.extend(rows)
        for x in rows:
            w, h, model = (x[k] for k in ("window", "horizon", "model"))
            path = ROOT / ("followup_predictions" if follow else "predictions") / f"w{w}_h{h}_{model}.parquet"
            scores = pd.read_parquet(path)
            scores["sector"] = scores.ticker.map(sector).fillna("Unknown")
            scores["regime"] = np.where(scores.market_20.ge(0), "rising", "falling")
            use[w, h, model] = scores
            if model != "boosted_relative":
                continue
            for level, row in x["selective"].items():
                flag = scores.confidence.ge(row["threshold"])
                selected = scores.assign(selected=flag)
                if level == "0.2":
                    for group_name in ("ticker", "sector", "regime"):
                        for group, data in selected.groupby(group_name):
                            result = selective(data, row["threshold"]) if group_name != "ticker" else {}
                            if not result:
                                yes = data[data.selected]
                                result = {"rows": len(yes), "dates": yes.date.nunique(),
                                          "coverage": len(yes)/len(data), "accuracy": yes.correct.mean()}
                            subgroup.append({"window": w, "horizon": h, "group_type": group_name,
                                             "group": group, **result})
                points.append({"window": w, "horizon": h, "nominal": level, **row})
    contrasts = []
    for (w, h, m), a in use.items():
        if m != "boosted_relative":
            continue
        b = use[w, h, "stock_logistic"]
        assert a[["date", "ticker", "y_true"]].equals(b[["date", "ticker", "y_true"]])
        z = a.assign(reference=b.p_up)
        daily = np.array([auc(y.y_true, y.p_up)-auc(y.y_true, y.reference) for _, y in z.groupby("date")
                          if y.y_true.nunique() == 2])
        n = len(daily)
        rng = np.random.default_rng(42)
        i = (rng.integers(n, size=(2000, (n+9)//10, 1))+np.arange(10)) % n
        ci = np.quantile(daily[i.reshape(2000,-1)[:,:n]].mean(axis=1), [.025, .975])
        contrasts.append({"window": w, "horizon": h, "auc_delta": float(daily.mean()), "interval": ci.tolist()})
    p = read_predictions("gat_lstm")
    _, labels, _ = load_inputs(include_news=False, legacy_labels=True)
    audit = []
    for h, q in p.groupby("horizon"):
        z = q.merge(labels, on=["date", "ticker"], validate="one_to_one")
        v = z.groupby("date").agg(up=("y_true", "mean"), r=(f"fwd_ret_{h}d", "mean"), score=("p_up", "mean"))
        pos = z.date.map(v.r).gt(0)
        audit.append({"horizon": int(h), "pooled_always_up": float(z.y_true.mean()),
                      "ex_post_oracle_mean_day_accuracy": float(np.maximum(v.up, 1-v.up).mean()),
                      "class_rate_market_up_ex_post": float(z.y_true[pos].mean()),
                      "class_rate_market_down_ex_post": float(z.y_true[~pos].mean()),
                      "score_day_mean_variance_share": float(np.var(z.date.map(v.score))/np.var(z.p_up)),
                      "score_day_mean_corr_with_up_rate": float(v.score.corr(v.up))})
    pd.DataFrame(subgroup).to_csv(ROOT / "selective_20pct_subgroups.csv", index=False, mode="x")
    results = {"fits": len(all_rows), "selected_model_nominal_levels": points,
               "paired_ranking_vs_stock_logistic": contrasts, "official_audit": audit}
    with (ROOT / "diagnostics.json").open("x") as out:
        json.dump(results, out, indent=2)
    print("fits", len(all_rows))
    for q in points:
        print(q["window"], q["horizon"], q["nominal"], q["rows"], q["dates"],
              q["coverage"], q["accuracy"], q["majority"], q["up_share"])
    print(contrasts)


if __name__ == "__main__":
    main()
