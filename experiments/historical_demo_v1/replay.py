"""Replay verified historical checkpoints or recalculate metrics from saved predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.market_direction_v1.study import finite
from src.core.io import read_table
from src.eval.run_baselines import load_inputs
from src.graph.build_edges import comention_edges, corr_edges, load_gics, sector_edges
from src.models.panel import build_panel
from src.models.train import (
    _chunks,
    build_model,
    fold_edges,
    fold_inputs,
    fold_report,
    folds_for,
    make_batch,
)
from src.nlp.rebuild_timed_news import file_hash

DEV = Path("experiments/gat_edge_weight_v1/runs")
OFFICIAL = Path("data/predictions")
CHECKPOINTS = Path("data/attention/checkpoints")
RUN = "w0_h5_s42_original"


def infer(model, panel, features, graph, indices, horizon, batch_days, *, decompose=False):
    rows = []
    model.eval()
    with torch.inference_mode():
        for chunk in _chunks(indices, batch_days):
            b = make_batch(chunk, panel, features, graph, horizon)
            args = b["x"], b["edge_index"], b["edge_type"], b["edge_weight"]
            if decompose:
                explanation = model.explain(*args, node_day=b["node_day"])
                logits = explanation["logits"]
            else:
                logits, _ = model(*args, node_day=b["node_day"])
            use = b["target"].numpy()
            at = b["at"][use]
            frame = pd.DataFrame({"date": panel.days[at[:, 0]], "ticker": panel.tickers[at[:, 1]].astype("string"),
                                  "p_up": logits.sigmoid().numpy()[use],
                                  "y_true": b["y"].numpy()[use].astype("int8")})
            if decompose:
                frame["without_graph"] = explanation["logits_no_graph"].sigmoid().numpy()[use]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def evaluate(f, record):
    m = fold_report(f, record)
    return m | {"majority": max(m["base_rate"], 1-m["base_rate"]), "rows": len(f), "dates": f.date.nunique()}


def compare(fresh, saved):
    keys = ["date", "ticker", "y_true"]
    p = fresh.merge(saved[keys + ["p_up"]], on=keys, suffixes=("", "_archived"), validate="one_to_one")
    if len(p) != len(fresh) or len(p) != len(saved):
        raise ValueError("assessment keys or labels differ")
    return float((p.p_up - p.p_up_archived).abs().max())


def development():
    record = next(r for r in map(json.loads, (DEV / "ledger.jsonl").read_text().splitlines()) if r["id"] == RUN)
    prices, labels, empty = load_inputs(include_news=False, legacy_labels=True)
    panel = build_panel(prices, labels, empty, embeddings=False)
    cfg = record["config"]
    start, end, expected = record["dates"]["fit"]
    dates = panel.days[(panel.days >= start) & (panel.days <= end)]
    assert len(dates) == expected
    start, end, expected = record["dates"]["assessment"]
    assess = panel.days[(panel.days >= start) & (panel.days <= end)]
    assert len(assess) == expected
    from src.eval.splitter import Fold
    from src.models.panel import fit_features

    graph_path = DEV / "graph_0.parquet"
    saved_graph = pd.read_parquet(graph_path)
    rebuild = corr_edges(prices, [Fold(0, dates, assess)])
    order = ["fold", "edge_type", "src", "dst"]
    pd.testing.assert_frame_equal(rebuild.sort_values(order).reset_index(drop=True),
                                  saved_graph.sort_values(order).reset_index(drop=True))
    feat = fit_features(panel, dates, window=cfg["window_days"], price=tuple(cfg["features"]["price"]),
                        news=(), embeddings=False)
    graph = fold_edges(saved_graph, panel, 0, ("corr",))
    model = build_model(len(feat.names), ("corr",), cfg)
    checkpoint = DEV / f"{RUN}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    preds = infer(model, panel, feat, graph, panel.day_index(assess), 5, cfg["train"]["batch_days"])
    saved = DEV / f"{RUN}.parquet"
    delta = compare(preds, pd.read_parquet(saved))
    assert delta <= 1e-7
    return dict(mode="checkpoint_inference", run=RUN, seed=cfg["seed"], split=record["dates"],
                features=feat.names, graph_rows=len(rebuild), graph_rebuild_exact=True,
                checkpoint_step=record["best_step"],
                threshold=record["threshold"], metrics=evaluate(preds, record), max_probability_delta=delta,
                sha256={str(p): file_hash(p) for p in (checkpoint, saved, graph_path, Path("data/prices.parquet"),
                                                      Path("data/labels.parquet"), Path("data/membership.parquet"))})


def official(rebuild=False):
    rec = json.loads((OFFICIAL / "gat_lstm_run.json").read_text())
    cfg, saved = rec["config"], pd.read_parquet(OFFICIAL / "gat_lstm.parquet")
    graph = None
    graph_audit = []
    if rebuild:
        prices, labels, news = load_inputs(include_news=True, legacy_labels=True)
        panel = build_panel(prices, labels, news, embeddings=True)
        folds = folds_for(panel)
        graph = pd.concat([sector_edges(load_gics()), corr_edges(prices, folds),
                           comention_edges(read_table("article_tickers"), folds)], ignore_index=True)
        archive = Path("data/attention")
        node_cols = ["horizon", "fold", "date", "ticker", "y_true", "p_up_no_graph"]
        contribution = pd.read_parquet(archive / "gat_lstm_contribution.parquet", columns=node_cols)
        edge_keys = ["fold", "src", "dst", "edge_type"]
        observed = pd.read_parquet(archive / "gat_lstm_edges.parquet", columns=[*edge_keys, "weight"]).drop_duplicates()
        for fold in folds:
            actual = graph[graph.fold.isin([-1, fold.index])].assign(fold=fold.index)
            check = observed[observed.fold.eq(fold.index)].merge(
                actual, on=edge_keys, how="left", validate="many_to_one", suffixes=("_observed", "_rebuilt"))
            graph_audit.append({"fold": fold.index, "observed_edges": len(check),
                                "mismatches_at_1e_6": int((~check.weight_observed.sub(
                                    check.weight_rebuilt).abs().le(1e-6)).sum())})
    output = []
    for r in rec["folds"]:
        h, i = r["horizon"], r["fold"]
        p = saved[saved.horizon.eq(h) & saved.fold.eq(i)]
        m = evaluate(p, r) | {"horizon": h, "fold": i,
                             "start": str(p.date.min().date()), "end": str(p.date.max().date())}
        if rebuild:
            types, f, g = fold_inputs(panel, graph, folds[i], cfg)
            model = build_model(len(f.names), types, cfg)
            cp = CHECKPOINTS / f"gat_lstm_h{h}_fold{i}.pt"
            model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
            idx = np.flatnonzero(panel.row_mask(folds[i].test, h).any(axis=1))
            fresh = infer(model, panel, f, g, idx, h, cfg["train"]["batch_days"], decompose=True)
            m["reconstructed_graph_edges"] = len(g["src"])
            m["checkpoint_probability_max_delta"] = compare(fresh, p)
            m["checkpoint_replay_matches"] = m["checkpoint_probability_max_delta"] <= 1e-6
            m["fresh_inference_metrics"] = evaluate(fresh, r)
            no_graph = contribution[contribution.fold.eq(i) & contribution.horizon.eq(h)]
            m["checkpoint_no_graph_max_delta"] = compare(fresh.assign(p_up=fresh.without_graph),
                                                         no_graph.rename(columns={"p_up_no_graph": "p_up"}))
        output.append(m)
    fields = ["accuracy", "majority", "mcc", "within_day_auc"]
    averages = {str(h): {k: float(np.mean([r[k] for r in output if r["horizon"] == h])) for k in fields}
                for h in (1, 5)}
    return dict(mode="official_rebuild_audit" if rebuild else "saved_prediction_metric_reproduction",
                graph_rows=len(graph) if rebuild else None, observed_graph_audit=graph_audit,
                folds=output, official_fold_means=averages)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["development", "official-saved", "official-rebuild"])
    mode = p.parse_args().mode
    torch.set_num_threads(2)
    result = development() if mode == "development" else official(rebuild=mode == "official-rebuild")
    print(json.dumps(finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
