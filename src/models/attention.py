"""Attention weights and the graph's contribution, per prediction (task 3.10, #42). Owner: Modelling.

An analysis output for NLP's interpretability work (2.9, 2.10), not a pipeline artifact, so it is
not in SCHEMA.md; the columns are documented in `results/attention/README.md`.

Attention is a softmax over a node's incoming edges of one type: it sums to one whether or not
the graph changed the prediction, so it says *which* neighbour, not *how much*. Every prediction
therefore also carries the graph's contribution to its logit (`GATLSTM.explain`).

Retrains the official proposed model (`configs/gat_lstm.yaml`, seed 42) on every fold and
horizon, refuses to write unless its test predictions equal `predictions/gat_lstm.parquet`
exactly, and writes:

- `<data>/attention/gat_lstm_edges.parquet`: every test (horizon, date, src, dst, edge_type)
- `<data>/attention/gat_lstm_contribution.parquet`: every test prediction
- `<data>/attention/gat_lstm_node_attention.parquet`: attention concentration per (prediction, edge type)
- `<data>/attention/checkpoints/gat_lstm_h<h>_fold<f>.pt` and `checkpoints.json`
- `results/attention/summary.md` and `results/attention/top_contributions.csv` (committed)

    python -m src.models.attention              # retrain, write everything
    python -m src.models.attention --summary    # rebuild the committed summaries from the Parquet files

`explain(date, ticker, horizon)` recomputes one prediction from the checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "gat_lstm"
TOP_N = 50  # candidate examples per horizon in top_contributions.csv
SHOWN = 10  # of which shown in summary.md


def attention_dir() -> Path:
    from src.core.paths import data_dir

    return data_dir() / "attention"


def results_dir() -> Path:
    from src.core.paths import RESULTS_DIR

    return RESULTS_DIR / "attention"


def explain_days(fitted, panel, day_idx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per prediction, one row per incoming edge of a prediction) for `day_idx`."""
    import torch

    from src.models.train import _chunks, make_batch

    model, types, heads = fitted.model, fitted.model.edge_types, fitted.model.heads
    window = fitted.features.window
    news_days = np.cumsum(np.nan_to_num(panel.news[:, :, 0]) > 0, axis=0)  # [D, K] has_news days so far
    preds, edges = [], []
    model.eval()
    with torch.no_grad():
        for chunk in _chunks(day_idx, fitted.batch_days):
            b = make_batch(chunk, panel, fitted.features, fitted.graph, fitted.horizon)
            out = model.explain(b["x"], b["edge_index"], b["edge_type"], b["edge_weight"])
            target = b["target"].numpy()
            at = b["at"]  # (day, ticker) of every node
            logit, base = out["logits"].numpy(), out["logits_no_graph"].numpy()
            src, dst = b["edge_index"].numpy()
            etype = b["edge_type"].numpy()
            keep = target[dst]
            deg = np.zeros((len(at), len(types)), np.int32)
            np.add.at(deg, (dst, etype), 1)
            p = pd.DataFrame({
                "horizon": fitted.horizon, "date": panel.days[at[:, 0]],
                "ticker": panel.tickers[at[:, 1]].astype("string"),
                "y_true": b["y"].numpy().astype("int8"), "has_news": panel.news[at[:, 0], at[:, 1], 0] > 0,
                # torch.sigmoid, as train_fold: p_up must equal the official predictions bit for bit
                "p_up": torch.sigmoid(out["logits"]).numpy(),
                "p_up_no_graph": torch.sigmoid(out["logits_no_graph"]).numpy(),
                "logit": logit, "logit_no_graph": base, "graph_logit": logit - base,
            })
            for t, name in enumerate(types):
                p[f"graph_logit_{name}"] = logit - out[f"drop_{name}"].numpy()
                p[f"in_degree_{name}"] = deg[:, t]
            p = p[target]
            # A batch holds whole days, so the day mean is over all of that day's predictions. The
            # day-level part shifts every ticker alike (market timing); within-day AUC ignores it.
            p.insert(p.columns.get_loc("graph_logit") + 1, "graph_logit_within_day",
                     p["graph_logit"] - p.groupby("date")["graph_logit"].transform("mean"))
            preds.append(p)
            alpha = out["attention"].numpy()[keep]
            s_at, d_at = at[src[keep]], at[dst[keep]]
            lo = s_at[:, 0] - window  # the window is days lo + 1 .. d
            before = np.where(lo >= 0, news_days[np.maximum(lo, 0), s_at[:, 1]], 0)
            src_days = news_days[s_at[:, 0], s_at[:, 1]] - before
            e = pd.DataFrame({
                "horizon": fitted.horizon, "date": panel.days[d_at[:, 0]],
                "src": panel.tickers[s_at[:, 1]].astype("string"), "dst": panel.tickers[d_at[:, 1]].astype("string"),
                "edge_type": pd.Series(np.asarray(types, dtype=object)[etype[keep]], dtype="string"),
                "weight": b["edge_weight"].numpy()[keep],
                **{f"alpha_{k}": alpha[:, k] for k in range(heads)},
                "alpha_mean": alpha.mean(axis=1),
                "src_has_news": panel.news[s_at[:, 0], s_at[:, 1], 0] > 0,
                "src_news_days": src_days.astype(np.int16),
            })
            edges.append(e)
    return pd.concat(preds, ignore_index=True), pd.concat(edges, ignore_index=True)


def node_attention(edges: pd.DataFrame, heads: int) -> pd.DataFrame:
    """Concentration of each prediction's incoming attention, per edge type.

    `entropy`: Shannon entropy of the attention over the k incoming edges, divided by log k (1 =
    uniform, 0 = all weight on one edge), averaged over heads. `top_share`: the largest
    head-averaged weight; `uniform_share` = 1 / k is what it would be with no preference.
    Only predictions with k >= 2 incoming edges of the type, since one edge always gets 1.
    """
    keys = ["horizon", "date", "dst", "edge_type"]
    a = edges[[f"alpha_{k}" for k in range(heads)]].to_numpy(np.float64)
    plogp = pd.DataFrame(-a * np.log(np.clip(a, 1e-12, None)), columns=range(heads))
    g = pd.concat([edges[keys].reset_index(drop=True), plogp,
                   edges["alpha_mean"].reset_index(drop=True)], axis=1).groupby(keys, observed=True)
    out = g[list(range(heads))].sum().mean(axis=1).rename("h").to_frame()
    out["k"] = g.size()
    out["top_share"] = g["alpha_mean"].max()
    out = out[out["k"] >= 2]
    out["entropy"] = out["h"] / np.log(out["k"])
    out["uniform_share"] = 1 / out["k"]
    return out.drop(columns="h").reset_index().rename(columns={"dst": "ticker"})


def within_day_auc_of(df: pd.DataFrame, col: str) -> float:
    from src.models.ladder_report import within_day_auc

    return within_day_auc(df[["date", "y_true"]].assign(p_up=df[col]))


def summarize(contrib: pd.DataFrame, nodes: pd.DataFrame, top_edges: pd.DataFrame, types) -> str:
    from src.graph.comention_report import md

    lines = [
        "# Attention and graph contribution, proposed model (task 3.10, #42)",
        "",
        "Generated by `python -m src.models.attention` from the official seed-42 `gat_lstm` (tuned config T1), "
        "retrained on every fold and horizon; its test predictions equal `predictions/gat_lstm.parquet` exactly. "
        "Test windows only. Column definitions and caveats: `README.md` in this folder; the full tables are in "
        "`<data>/attention/` on the Drive.",
        "",
        "Attention weights are a softmax over a node's incoming edges of one type: they sum to one whether or "
        "not the graph changed the prediction. The contribution measure below is how much the graph moved the "
        "logit; read attention only together with it.",
        "",
        "## Attention concentration, per edge type",
        "",
        "Per prediction with at least 2 incoming edges of the type. `entropy` 1 = uniform over the incoming "
        "edges, 0 = all weight on one edge (averaged over the 4 heads). `top share` is the largest head-averaged "
        "weight, `uniform` = 1/k is what it would be with no preference.",
        "",
    ]
    conc = nodes.groupby(["horizon", "edge_type"], observed=True).agg(
        predictions=("k", "size"), median_k=("k", "median"), mean_entropy=("entropy", "mean"),
        median_entropy=("entropy", "median"), median_top_share=("top_share", "median"),
        median_uniform=("uniform_share", "median"))
    conc["top / uniform (median)"] = (nodes.assign(r=nodes["top_share"] / nodes["uniform_share"])
                                     .groupby(["horizon", "edge_type"], observed=True)["r"].median())
    lines += [md(conc.reset_index().astype({"predictions": int}), index=False, digits=3), ""]

    lines += ["## Graph contribution to each prediction", "",
              "`graph logit` = logit minus the logit with the graph signal set to zero (the dense layer sees "
              "`[h ; 0]`). It splits into a **day-level** part, that day's mean over all tickers, which shifts "
              "every prediction alike (market timing, invisible to within-day AUC), and a **within-day** part "
              "(`graph_logit_within_day`), which is what can change the day's ranking. Scale references: each "
              "day's cross-sectional spread (std across tickers) of the model's logit, with and without the "
              "graph signal, as medians over days.", ""]
    cols = {}
    for h, c in contrib.groupby("horizon"):
        day = c.groupby("date")["graph_logit"].mean()
        wd = c["graph_logit_within_day"]
        centred = c["graph_logit"] - c["graph_logit"].mean()
        a, dp = c["graph_logit"].abs(), (c["p_up"] - c["p_up_no_graph"]).abs()
        cols[f"{int(h)}d"] = {
            "predictions": f"{len(c):,}",
            "abs(graph logit): median": f"{a.median():.3f}",
            "abs(graph logit): p90 / p99": f"{a.quantile(0.9):.3f} / {a.quantile(0.99):.3f}",
            "abs(dp): median / p90": f"{dp.median():.3f} / {dp.quantile(0.9):.3f}",
            "day-level part: median abs(day mean)": f"{day.abs().median():.3f}",
            "day-level share of graph-logit variance": f"{1 - (wd ** 2).sum() / (centred ** 2).sum():.2f}",
            "within-day part: median abs": f"{wd.abs().median():.4f}",
            "within-day part: p90 / p99 abs": f"{wd.abs().quantile(0.9):.4f} / {wd.abs().quantile(0.99):.4f}",
            "daily logit spread, graph on (median)": f"{c.groupby('date')['logit'].std().median():.4f}",
            "daily logit spread, graph off (median)": f"{c.groupby('date')['logit_no_graph'].std().median():.4f}",
            "within-day AUC, graph on": f"{within_day_auc_of(c, 'p_up'):.4f}",
            "within-day AUC, graph off": f"{within_day_auc_of(c, 'p_up_no_graph'):.4f}",
            "within-day AUC of graph logit alone": f"{within_day_auc_of(c, 'graph_logit'):.4f}",
        }
    t = pd.DataFrame(cols)
    lines += ["| | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
    lines += [f"| {k} | " + " | ".join(t.loc[k]) + " |" for k in t.index]
    by_fold = contrib.assign(a=contrib["graph_logit"].abs(), w=contrib["graph_logit_within_day"].abs()).groupby(
        ["horizon", "fold"])[["a", "w"]].median().unstack("fold")
    lines += ["", "Median by fold (folds differ a lot; read the pooled numbers with that in mind):", "",
              "| horizon | part | " + " | ".join(f"fold {f}" for f in by_fold["a"].columns) + " |",
              "|---|---|" + "---|" * by_fold["a"].shape[1]]
    for h in by_fold.index:
        lines.append(f"| {int(h)}d | abs(graph logit) | " + " | ".join(f"{v:.3f}" for v in by_fold.loc[h, "a"]) + " |")
        lines.append(f"| {int(h)}d | abs(within-day part) | "
                     + " | ".join(f"{v:.3f}" for v in by_fold.loc[h, "w"]) + " |")
    lines.append("")
    per_type = pd.DataFrame({
        (int(h), name): {"median": c[f"graph_logit_{name}"].abs().median(),
                         "p90": c[f"graph_logit_{name}"].abs().quantile(0.9),
                         "p99": c[f"graph_logit_{name}"].abs().quantile(0.99)}
        for h, c in contrib.groupby("horizon") for name in types}).T
    per_type.index = [f"{h}d {name}" for h, name in per_type.index]
    lines += ["Per edge type, abs(logit change when that type's message is removed), day-level part included "
              "(types are not additive):", "",
              md(per_type.reset_index(names="horizon, edge type"), index=False, digits=4), ""]

    lines += ["## Candidate examples: largest graph contribution", "",
              f"Top {SHOWN} per horizon by abs(within-day graph contribution) (all {TOP_N} per horizon in "
              "`top_contributions.csv`): ranked on the part that changes the day's ranking, because the raw "
              "contribution's largest values are day-level shifts that move every ticker alike. `via` is the edge "
              "type whose removal changes the logit most, and `top_src` its most-attended "
              "incoming edge. `src news days` counts the source's days with news in the input window; the "
              "source's articles for those days are in `article_tickers`.", ""]
    for h, g in top_edges.groupby("horizon"):
        g = g.head(SHOWN)
        shown = g[["date", "ticker", "y_true", "p_up", "p_up_no_graph", "graph_logit", "graph_logit_within_day",
                   "via", "top_src",
                   "top_alpha", "top_weight", "src_has_news", "src_news_days"]].copy()
        shown["date"] = shown["date"].dt.date
        lines += [f"### {int(h)} day{'s' if h > 1 else ''}", "", md(shown, index=False, digits=3), ""]
    return "\n".join(lines)


def top_contributions(contrib: pd.DataFrame, edges_for, types, n: int = TOP_N) -> pd.DataFrame:
    """The n predictions per horizon with the largest within-day graph contribution, with the edge
    that carried it."""
    rows = []
    for _h, c in contrib.groupby("horizon"):
        top = c.reindex(c["graph_logit_within_day"].abs().sort_values(ascending=False).index).head(n).copy()
        cols = [f"graph_logit_{t}" for t in types]
        top["via"] = top[cols].abs().idxmax(axis=1).str.removeprefix("graph_logit_")
        rows.append(top)
    top = pd.concat(rows, ignore_index=True)
    e = edges_for(top)
    e = e.sort_values("alpha_mean", ascending=False).drop_duplicates(["horizon", "date", "dst", "edge_type"])
    m = top.merge(e.rename(columns={"dst": "ticker", "edge_type": "via", "src": "top_src", "alpha_mean": "top_alpha",
                                    "weight": "top_weight"})[["horizon", "date", "ticker", "via", "top_src",
                                                              "top_alpha", "top_weight", "src_has_news",
                                                              "src_news_days"]],
                  on=["horizon", "date", "ticker", "via"], how="left")
    return m


def _edges_reader(path: Path):
    def edges_for(top: pd.DataFrame) -> pd.DataFrame:
        import pyarrow.parquet as pq

        dates = sorted(set(top["date"]))
        t = pq.read_table(path, filters=[("dst", "in", sorted(set(top["ticker"]))), ("date", "in", dates)])
        e = t.to_pandas()
        e["date"] = pd.to_datetime(e["date"])
        keys = set(zip(top["horizon"], top["date"], top["ticker"], strict=True))
        return e[[k in keys for k in zip(e["horizon"], e["date"], e["dst"], strict=True)]]
    return edges_for


def write_summary(out: Path, results: Path, types) -> None:
    contrib = pd.read_parquet(out / f"{MODEL}_contribution.parquet")
    nodes = pd.read_parquet(out / f"{MODEL}_node_attention.parquet")
    for df in (contrib, nodes):
        df["date"] = pd.to_datetime(df["date"])
    top = top_contributions(contrib, _edges_reader(out / f"{MODEL}_edges.parquet"), types)
    results.mkdir(parents=True, exist_ok=True)
    top.assign(date=top["date"].dt.date).to_csv(results / "top_contributions.csv", index=False)
    (results / "summary.md").write_text(summarize(contrib, nodes, top, types) + "\n", encoding="utf-8")


def run(out: Path) -> tuple[str, ...]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from src.core.config import load_config
    from src.core.io import read_predictions, read_table
    from src.core.paths import CONFIGS_DIR
    from src.models.panel import load_panel
    from src.models.train import folds_for, train_fold

    cfg = load_config(CONFIGS_DIR / f"{MODEL}.yaml")
    panel, edges = load_panel(), read_table("edges")
    official = read_predictions(MODEL)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    parts = {"contribution": [], "node_attention": []}
    records, types = [], ()
    try:
        for h in cfg["horizons"]:
            for fold in folds_for(panel):
                preds, info, fitted = train_fold(panel, edges, fold, h, cfg, log=lambda *_: None, return_model=True)
                types = fitted.model.edge_types
                contrib, edge_rows = explain_days(fitted, panel, fitted.test_idx)
                contrib.insert(1, "fold", np.int16(fold.index))
                edge_rows.insert(1, "fold", np.int16(fold.index))
                o = official[(official["horizon"] == h) & (official["fold"] == fold.index)]
                check = o.merge(contrib, on=["date", "ticker"], suffixes=("", "_x"))
                same = (check["p_up"] == check["p_up_x"].astype("float32")).all()
                if len(check) != len(o) or len(o) != len(contrib) or not same:
                    raise RuntimeError(f"h={h} fold {fold.index}: retrained predictions differ from "
                                       f"predictions/{MODEL}.parquet; attention would not explain the official run")
                parts["contribution"].append(contrib)
                parts["node_attention"].append(node_attention(edge_rows, fitted.model.heads).assign(fold=fold.index))
                tbl = pa.Table.from_pandas(edge_rows, preserve_index=False)
                if "edges" not in writers:
                    writers["edges"] = pq.ParquetWriter(out / f"{MODEL}_edges.parquet", tbl.schema)
                writers["edges"].write_table(tbl)
                ckpt = out / "checkpoints" / f"{MODEL}_h{h}_fold{fold.index}.pt"
                torch.save(fitted.model.state_dict(), ckpt)
                records.append({"horizon": h, "fold": fold.index, "checkpoint": ckpt.name,
                                "best_step": info["best_step"], "threshold": info["threshold"],
                                "features": fitted.features.names, "test_rows": len(contrib),
                                "edge_rows": len(edge_rows)})
                print(f"h={h} fold {fold.index}: {len(contrib)} predictions, {len(edge_rows)} edges, "
                      f"median |graph logit| {contrib['graph_logit'].abs().median():.4f}", flush=True)
    finally:
        for w in writers.values():
            w.close()
    for name, frames in parts.items():
        pd.concat(frames, ignore_index=True).to_parquet(out / f"{MODEL}_{name}.parquet", index=False)
    (out / "checkpoints" / "checkpoints.json").write_text(
        json.dumps({"model": MODEL, "seed": cfg["seed"], "config": cfg, "edge_types": list(types),
                    "folds": records}, indent=2, default=str), encoding="utf-8")
    return types


def load_fitted(horizon: int, fold_index: int, *, panel=None, edges=None):
    """A `Fitted` rebuilt from the saved checkpoint: same features, graph and weights."""
    import torch

    from src.core.config import load_config
    from src.core.io import read_table
    from src.core.paths import CONFIGS_DIR
    from src.models.panel import load_panel
    from src.models.train import Fitted, build_model, fold_inputs, folds_for

    meta = json.loads((attention_dir() / "checkpoints" / "checkpoints.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    if cfg != load_config(CONFIGS_DIR / f"{MODEL}.yaml"):
        raise ValueError(f"checkpoints were saved with a different configs/{MODEL}.yaml")
    panel = panel if panel is not None else load_panel()
    edges = edges if edges is not None else read_table("edges")
    fold = folds_for(panel)[fold_index]
    edge_types, features, graph = fold_inputs(panel, edges, fold, cfg)
    model = build_model(len(features.names), edge_types, cfg)
    model.load_state_dict(torch.load(attention_dir() / "checkpoints" / f"{MODEL}_h{horizon}_fold{fold_index}.pt"))
    model.eval()
    test_idx = np.flatnonzero(panel.row_mask(fold.test, horizon).any(axis=1))
    return Fitted(model, features, graph, test_idx, horizon, cfg["train"]["batch_days"])


def explain(date, ticker: str, horizon: int, *, panel=None, edges=None) -> tuple[pd.Series, pd.DataFrame]:
    """One test prediction recomputed from the checkpoints: (its contribution row, its incoming edges)."""
    from src.models.panel import load_panel
    from src.models.train import folds_for

    panel = panel if panel is not None else load_panel()
    date = pd.Timestamp(date)
    fold = next((f for f in folds_for(panel) if date in f.test), None)
    if fold is None:
        raise ValueError(f"{date.date()} is in no fold's test window")
    fitted = load_fitted(horizon, fold.index, panel=panel, edges=edges)
    contrib, e = explain_days(fitted, panel, panel.day_index([date]))
    row = contrib[contrib["ticker"] == ticker]
    if row.empty:
        raise ValueError(f"no {horizon}d prediction for {ticker} on {date.date()}")
    return row.iloc[0], e[e["dst"] == ticker].sort_values(["edge_type", "alpha_mean"], ascending=[True, False])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summary", action="store_true", help="rebuild the committed summaries; train nothing")
    args = p.parse_args()
    out = attention_dir()
    if args.summary:
        types = tuple(json.loads((out / "checkpoints" / "checkpoints.json").read_text(encoding="utf-8"))["edge_types"])
    else:
        out.mkdir(parents=True, exist_ok=True)
        types = run(out)
    write_summary(out, results_dir(), types)
    print((results_dir() / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
