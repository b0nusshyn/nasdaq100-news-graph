"""Train the proposed model across walk-forward folds and write predictions. Owner: Modelling. Issues #29, #36.

Output: `<data>/predictions/<model>.parquet` (SCHEMA.md `predictions`), where `<model>` is the
config's `model` (`gat_lstm`; `lstm_news` and `lstm_price` for Baselines 6 and 5, which are
this same loop with the GAT, and then news, removed); an ablation run writes
`<model>_no_<what>.parquet`. One model per horizon and fold. Inputs come from
`src.models.panel`, the module every model shares: the same row set, and features normalised
with statistics from each training fold only.

Within a fold, the last `validation_fraction` of the training days is held out for early
stopping, separated from the days the model fits on by the walk-forward embargo, so a 5-day
label on the last fitted day cannot see into the validation days.

    python -m src.models.train --config configs/gat_lstm.yaml
    python -m src.models.train --config configs/gat_lstm.yaml --folds 0   # smoke run, nothing written
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "gat_lstm"
NEWS = "news"  # --ablate value that drops every news feature, embeddings included
# --ablate value that drops the `corr` edges of three stock-for-stock acquisitions pending in every
# fold's training window (3.4, #22): their correlation is a deal term, not economic spillover.
# Their sector and comention edges stay.
DEAL_ARB = "deal_arb"
DEAL_ARB_PAIRS = (("ADI", "MXIM"), ("AMD", "XLNX"), ("AZN", "ALXN"))


def model_name(ablate=(), base: str = MODEL) -> str:
    """`base` (the config's `model`, default `gat_lstm`), or `<base>_no_<a>_no_<b>` for an
    ablation. The file name and the `model` column of the predictions."""
    return base + "".join(f"_no_{a}" for a in sorted(ablate))


def fold_edges(edges: pd.DataFrame, panel, fold: int, edge_types: tuple[str, ...],
               drop_corr_pairs=()) -> dict[str, np.ndarray]:
    """This fold's edges (its own rows plus the shared `sector` rows) as ticker-index arrays,
    without the `corr` edges between any pair in `drop_corr_pairs` (either direction)."""
    from src.core.schema import ALL_FOLDS

    if not edge_types:  # Baseline 6 and the all-edges ablation: no graph, edges not needed
        empty = np.array([], dtype=np.int64)
        return {"src": empty, "dst": empty, "type": empty, "weight": np.array([], dtype=np.float32)}
    e = edges[edges["fold"].isin([fold, ALL_FOLDS]) & edges["edge_type"].isin(edge_types)]
    if drop_corr_pairs:
        pairs = {frozenset(p) for p in drop_corr_pairs}
        hit = [frozenset(sd) in pairs for sd in zip(e["src"], e["dst"], strict=True)]
        e = e[~((e["edge_type"] == "corr").to_numpy() & np.array(hit, dtype=bool))]
    missing = sorted(set(edge_types) - set(e["edge_type"]))
    if missing:
        raise ValueError(f"edges.parquet has no {missing} edges for fold {fold}: rebuild it (src.graph.build_edges)")
    src, dst = panel.tickers.get_indexer(e["src"]), panel.tickers.get_indexer(e["dst"])
    keep = (src >= 0) & (dst >= 0)  # edges to tickers without prices can never be used
    return {
        "src": src[keep], "dst": dst[keep],
        "type": e["edge_type"].map(edge_types.index).to_numpy()[keep],
        "weight": e["weight"].to_numpy(np.float32)[keep],
    }


def make_batch(days: np.ndarray, panel, features, graph: dict, horizon: int):
    """Several trading days as one disjoint graph. Nodes are the tickers in the index that day;
    edges join two of them. `target` marks the nodes in the row set (labelled for `horizon`)."""
    import torch

    xs, src, dst, etype, weight, ys, at = [], [], [], [], [], [], []
    offset = 0
    pos = np.full(len(panel.tickers), -1)
    for d in days:
        nodes = np.flatnonzero(panel.in_index[d])
        pos[:] = -1
        pos[nodes] = np.arange(len(nodes))
        s, t = pos[graph["src"]], pos[graph["dst"]]
        keep = (s >= 0) & (t >= 0)
        xs.append(features.windows(d, nodes))
        src.append(s[keep] + offset)
        dst.append(t[keep] + offset)
        etype.append(graph["type"][keep])
        weight.append(graph["weight"][keep])
        ys.append(panel.labels[horizon][d, nodes])
        at.append(np.stack([np.full(len(nodes), d), nodes], axis=1))
        offset += len(nodes)
    y = np.concatenate(ys)
    return {
        "x": torch.from_numpy(np.concatenate(xs)),
        "edge_index": torch.from_numpy(np.stack([np.concatenate(src), np.concatenate(dst)])).long(),
        "edge_type": torch.from_numpy(np.concatenate(etype)).long(),
        "edge_weight": torch.from_numpy(np.concatenate(weight)),
        "y": torch.from_numpy(np.nan_to_num(y)).float(),
        "target": torch.from_numpy(~np.isnan(y)),
        "node_day": torch.from_numpy(np.concatenate(at)[:, 0].copy()),
        "at": np.concatenate(at),
    }


def _chunks(days: np.ndarray, size: int):
    return [days[i: i + size] for i in range(0, len(days), size)]


@dataclass(frozen=True)
class Fitted:
    """A trained fold model and what it needs to score new batches (attention, #42)."""

    model: object  # GATLSTM, in eval mode, at the early-stopping checkpoint
    features: object  # FoldFeatures fitted on the fold's training days
    graph: dict  # fold_edges output the model was trained on
    test_idx: np.ndarray  # panel day indices of the test window with labelled rows
    horizon: int
    batch_days: int


@dataclass(frozen=True)
class GraphSources:
    """Aligned source observations for rebuilding learned edges on the weight-fit partition."""

    prices: pd.DataFrame | None = None
    article_tickers: pd.DataFrame | None = None


def training_partitions(fold, cfg):
    from src.core.config import walk_forward_config
    from src.models.panel import split_validation

    embargo = cfg["embargo_days"] if "embargo_days" in cfg else walk_forward_config()["embargo_days"]
    if embargo < 5:
        raise ValueError("The model requires an internal embargo of at least five trading sessions")
    return split_validation(fold.train, cfg["train"]["validation_fraction"], embargo)


def fold_inputs(panel, edges: pd.DataFrame | GraphSources | None, fold, cfg: dict, ablate=()):
    """Prepare outer-training inputs for legacy runs or weight-fit-only inputs for new runs."""
    from src.eval.splitter import Fold
    from src.models.panel import fit_features

    feat, gat = cfg["features"], cfg["gat"]
    edge_types = tuple(t for t in gat["edge_types"] if t not in ablate)
    scope = cfg["train"].get("preprocessing_scope", "outer_train")
    if scope not in ("outer_train", "weight_fit"):
        raise ValueError("preprocessing_scope must be outer_train or weight_fit")
    fit_dates = fold.train
    if scope == "weight_fit":
        fit_dates, _ = training_partitions(fold, cfg)
        if edge_types:
            from src.graph.build_edges import comention_edges, corr_edges

            if not isinstance(edges, GraphSources) or set(edge_types) - {"corr", "comention"}:
                raise ValueError("weight_fit graphs require raw GraphSources and corr/comention edges only")
            fit_fold = Fold(fold.index, fit_dates, fold.test)
            built = []
            if "corr" in edge_types:
                if edges.prices is None:
                    raise ValueError("weight_fit correlation edges require prices")
                built.append(corr_edges(edges.prices, [fit_fold]))
            if "comention" in edge_types:
                if edges.article_tickers is None:
                    raise ValueError("weight_fit co-mention edges require aligned article links")
                built.append(comention_edges(edges.article_tickers, [fit_fold]))
            edges = pd.concat(built, ignore_index=True)
    drop_pairs = DEAL_ARB_PAIRS if DEAL_ARB in ablate else ()
    no_news = NEWS in ablate
    features = fit_features(
        panel, fit_dates, window=cfg["window_days"], price=tuple(feat["price"]),
        news=() if no_news else tuple(feat["news"]),
        embeddings=bool(feat.get("embeddings", True)) and not no_news, embedding_dim=feat["embedding_dim"],
    )
    graph = fold_edges(edges, panel, fold.index, edge_types, drop_corr_pairs=drop_pairs)
    return edge_types, features, graph


def build_model(n_features: int, edge_types: tuple[str, ...], cfg: dict):
    from src.models.gat_lstm import GATLSTM

    enc, gat = cfg["encoder"], cfg["gat"]
    return GATLSTM(n_features, enc["hidden"], gat["heads"], edge_types, layers=enc["layers"],
                   dropout=enc["dropout"], signed_messages=gat.get("signed_messages", False),
                   center_graph_inputs=gat.get("center_graph_inputs", False))


def train_fold(panel, edges: pd.DataFrame | GraphSources | None, fold, horizon: int, cfg: dict, *,
               ablate=(), predict_test=True,
               log=print, return_model=False):
    """Fit one model on `fold`'s training days and predict its test days.

    Returns (predictions in SCHEMA.md `predictions` shape, info dict with timing, epochs and the
    validation scores). `predict_test=False` (tuning, #58) never touches the test window and
    returns `None` for the predictions. `return_model=True` appends a `Fitted`.
    """
    import torch

    from src.models.panel import choose_threshold

    start = time.perf_counter()
    tr = cfg["train"]
    edge_types, features, graph = fold_inputs(panel, edges, fold, cfg, ablate)
    fit_days, val_days = training_partitions(fold, cfg)

    def days_with_rows(dates) -> np.ndarray:
        return np.flatnonzero(panel.row_mask(dates, horizon).any(axis=1))

    fit_idx, val_idx = days_with_rows(fit_days), days_with_rows(val_days)
    if not len(fit_idx) or not len(val_idx):
        raise ValueError("Both training and validation require eligible labelled rows")
    val_batches = [make_batch(c, panel, features, graph, horizon) for c in _chunks(val_idx, tr["batch_days"])]
    probe_batches = [
        make_batch(c, panel, features, graph, horizon)
        for c in _chunks(fit_idx[::5], tr["batch_days"])
    ] if tr.get("record_history") else []

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = build_model(len(features.names), edge_types, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    def run(b):
        logits, _ = model(b["x"], b["edge_index"], b["edge_type"], b["edge_weight"], node_day=b["node_day"])
        return logits

    from src.models.ladder_report import within_day_auc

    def validate(batches) -> tuple[float, float]:
        """Validation (loss, within-day AUC). Loss also rewards each day's overall level (market
        timing); within-day AUC scores only how the day's tickers are ranked: the headline."""
        model.eval()
        total, n, parts = 0.0, 0, []
        with torch.no_grad():
            for b in batches:
                logits, y = run(b)[b["target"]], b["y"][b["target"]]
                total += float(loss_fn(logits, y))
                n += int(b["target"].sum())
                parts.append(pd.DataFrame({"date": b["at"][b["target"].numpy()][:, 0],
                                           "p_up": torch.sigmoid(logits).numpy(), "y_true": y.numpy()}))
        model.train()
        return total / n, within_day_auc(pd.concat(parts, ignore_index=True))

    # Early stopping (#58): checks every `validate_every_steps` optimiser steps (default once per
    # epoch) and keeps the checkpoint with the best `early_stopping_metric`; the patience is in
    # epochs. The other metric's optimum is logged too, to measure how far apart they are.
    stop_metric = tr.get("early_stopping_metric", "within_day_auc")
    if stop_metric not in ("within_day_auc", "loss"):
        raise ValueError(f"early_stopping_metric must be within_day_auc or loss, not {stop_metric!r}")
    steps_per_epoch = math.ceil(len(fit_idx) / tr["batch_days"])
    every = min(tr.get("validate_every_steps") or steps_per_epoch, steps_per_epoch)
    patience = math.ceil(tr["early_stopping_patience"] * steps_per_epoch / every)

    def score(loss: float, wd: float) -> float:  # higher is better; NaN never wins
        s = wd if stop_metric == "within_day_auc" else -loss
        return -math.inf if math.isnan(s) else s

    best, best_state, chosen, stale, history, step, epoch = -math.inf, None, None, 0, [], 0, 0
    model.train()
    while epoch < tr["epochs"] and (not history or stale < patience):
        epoch += 1
        for chunk in _chunks(rng.permutation(fit_idx), tr["batch_days"]):
            b = make_batch(chunk, panel, features, graph, horizon)
            loss = loss_fn(run(b)[b["target"]], b["y"][b["target"]]) / b["target"].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            if tr.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
            step += 1
            if step % every:
                continue
            v, wd = validate(val_batches)
            check = {"step": step, "loss": v, "within_day_auc": wd}
            if probe_batches:
                check["train_probe_loss"], check["train_probe_within_day_auc"] = validate(probe_batches)
            history.append(check)
            if best_state is None or score(v, wd) > best + 1e-6:
                best, stale = score(v, wd), 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                chosen = check
            else:
                stale += 1
            log(f"  h={horizon} fold {fold.index} epoch {epoch} step {step}: val loss {v:.4f}, within-day AUC {wd:.3f}")
            if stale >= patience:
                break
    if chosen is None:
        raise ValueError("Training ended before any checkpoint was evaluated")
    model.load_state_dict(best_state)
    loss_opt = min(history, key=lambda r: r["loss"])

    model.eval()

    def predict(day_idx: np.ndarray) -> pd.DataFrame:
        parts = []
        with torch.no_grad():
            for chunk in _chunks(day_idx, tr["batch_days"]):
                b = make_batch(chunk, panel, features, graph, horizon)
                p = torch.sigmoid(run(b))[b["target"]].numpy()
                at = b["at"][b["target"].numpy()]
                parts.append(pd.DataFrame({
                    "date": panel.days[at[:, 0]], "ticker": panel.tickers[at[:, 1]].astype("string"),
                    "p_up": p, "y_true": b["y"][b["target"]].numpy().astype("int8"),
                }))
        return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

    # The operating threshold is chosen on the validation days, never the test window.
    val = predict(val_idx)
    threshold = choose_threshold(val["y_true"].to_numpy(), val["p_up"].to_numpy())
    import warnings

    from src.eval.metrics import classification_metrics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # undefined MCC is reported as NaN
        val_metrics = classification_metrics(val["y_true"].to_numpy(), val["p_up"].to_numpy(), threshold=threshold)
    info = {
        "fold": fold.index, "horizon": horizon, "seconds": time.perf_counter() - start,
        "stop_metric": stop_metric, "epochs": epoch, "steps": step, "steps_per_epoch": steps_per_epoch,
        "checks": len(history), "best_step": chosen["step"], "best_epoch": round(chosen["step"] / steps_per_epoch, 2),
        "best_val_within_day_auc": chosen["within_day_auc"], "val_loss_at_best": chosen["loss"],
        "best_val_loss": chosen["loss"],  # kept for older run records; same as val_loss_at_best
        "loss_optimal_step": loss_opt["step"], "val_within_day_auc_at_loss_optimum": loss_opt["within_day_auc"],
        "min_val_loss": loss_opt["loss"], "threshold": threshold, "val_base_rate": float(val["y_true"].mean()),
        "val_mcc": val_metrics["mcc"], "fit_days": len(fit_idx), "val_days": len(val_idx),
        "features": len(features.names), "edges_per_day": len(graph["src"]),
    }
    if probe_batches:
        info["history"] = history
        info["validation"] = val_metrics
    if not predict_test:
        return None, info
    test_idx = days_with_rows(fold.test)
    name = model_name(ablate, cfg.get("model", MODEL))
    preds = predict(test_idx).assign(model=name, horizon=horizon, fold=fold.index)
    info["test_rows"] = len(preds)
    if return_model:
        return preds, info, Fitted(model, features, graph, test_idx, horizon, tr["batch_days"])
    return preds, info


def fold_report(preds: pd.DataFrame, info: dict) -> dict:
    """Test metrics for one fold-horizon: within-day AUC (the headline, stock selection) and fold
    AUC (no threshold) first, then accuracy, F1 and MCC at the validation-chosen threshold, with
    the base rate that always predicting "up" scores."""
    import warnings

    from src.eval.metrics import classification_metrics
    from src.models.ladder_report import within_day_auc

    y, p = preds["y_true"].to_numpy(), preds["p_up"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # undefined MCC is reported as NaN
        m = classification_metrics(y, p, threshold=info["threshold"])
    return {"within_day_auc": within_day_auc(preds), "auc": m["auc"], "threshold": info["threshold"],
            "accuracy": m["accuracy"], "base_rate": float(y.mean()),
            "f1": m["f1"], "mcc": m["mcc"], "share_predicted_up": float((p >= info["threshold"]).mean())}


def folds_for(panel, wf: dict | None = None) -> list:
    """The shared walk-forward folds (same as every other model) on the panel's calendar."""
    from src.graph.comention_report import shared_folds

    days = pd.DataFrame({"date": panel.days, "in_index": panel.in_index.any(axis=1)})
    return shared_folds(days, wf)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/gat_lstm.yaml"))
    p.add_argument("--horizon", type=int, choices=[1, 5], default=None, help="default: both")
    p.add_argument("--ablate", nargs="*", default=[], help="edge types, 'news' or 'deal_arb' to drop (#41)")
    p.add_argument("--legacy-labels", action="store_true", help="verify and map legacy return columns in memory")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="train only these folds (smoke run): prints metrics, writes no predictions")
    args = p.parse_args()

    from src.core.config import load_config
    from src.core.io import read_table, write_predictions
    from src.core.paths import predictions_path
    from src.models.panel import build_panel, lag_on_sessions, load_panel
    from src.models.price_features import EXTENDED_PRICE

    cfg = load_config(args.config)
    bad = set(args.ablate) - set(cfg["gat"]["edge_types"]) - {NEWS, DEAL_ARB}
    if bad:
        p.error(f"--ablate: unknown {sorted(bad)}; choose from {cfg['gat']['edge_types'] + [NEWS, DEAL_ARB]}")
    if DEAL_ARB in args.ablate and "corr" in args.ablate:
        p.error("--ablate: deal_arb drops some corr edges; with corr dropped too it changes nothing")
    t0 = time.perf_counter()
    name = model_name(args.ablate, cfg.get("model", MODEL))
    strict = cfg["train"].get("preprocessing_scope") == "weight_fit"
    output = predictions_path(name)
    if strict and args.folds is None and (
        name == MODEL or output.exists() or output.with_name(f"{name}_run.json").exists()
    ):
        p.error("weight_fit runs require a new model name and unused output paths")
    embeddings = bool(cfg["features"].get("embeddings", True)) and NEWS not in args.ablate
    lag = cfg.get("news_lag_sessions", 1 if strict else 0)
    extended = bool(set(cfg["features"]["price"]) & set(EXTENDED_PRICE))
    if args.legacy_labels:
        from src.eval.run_baselines import load_inputs

        prices, labels, news = load_inputs(include_news=True, legacy_labels=True)
        calendar = pd.DatetimeIndex(sorted(prices.date.unique()))
        panel = build_panel(prices, labels, lag_on_sessions(news, calendar, lag),
                            embeddings=embeddings, extended_price=extended)
    else:
        panel = load_panel(embeddings=embeddings, news_lag_sessions=lag, extended_price=extended)
    kinds = set(cfg["gat"]["edge_types"]) - set(args.ablate)
    if strict:
        edges = GraphSources(
            prices=read_table("prices") if "corr" in kinds else None,
            article_tickers=lag_on_sessions(read_table("article_tickers"), panel.days, lag)
            if "comention" in kinds else None,
        )
    else:
        edges = read_table("edges") if kinds else None
    folds = folds_for(panel)
    chosen = folds if args.folds is None else [folds[i] for i in args.folds]
    print(f"Loaded inputs in {time.perf_counter() - t0:.0f}s: {len(panel.days)} days x {len(panel.tickers)} tickers, "
          f"{len(folds)} folds, model {name}")

    horizons = [args.horizon] if args.horizon else cfg["horizons"]
    out, infos = [], []
    for h in horizons:
        for fold in chosen:
            preds, info = train_fold(panel, edges, fold, h, cfg, ablate=tuple(args.ablate))
            info["test"] = m = fold_report(preds, info)
            print(f"WALL CLOCK h={h} fold {fold.index}: {info['seconds'] / 60:.1f} min "
                  f"({info['epochs']} epochs, best {info['best_epoch']}) | within-day AUC {m['within_day_auc']:.3f}, "
                  f"fold AUC {m['auc']:.3f} | "
                  f"threshold {m['threshold']:.3f} (validation): acc {m['accuracy']:.3f} vs base rate "
                  f"{m['base_rate']:.3f}, MCC {m['mcc']:.3f}, F1 {m['f1']:.3f}, "
                  f"{m['share_predicted_up']:.0%} predicted up | {info['test_rows']} test rows")
            out.append(preds)
            infos.append(info)
    total = sum(i["seconds"] for i in infos)
    print(f"Total training time {total / 60:.1f} min for {len(infos)} fold-horizons "
          f"({total / 60 / len(infos):.1f} min each)")
    if args.folds is not None:
        print("Smoke run (--folds): predictions not written.")
        return
    path = write_predictions(pd.concat(out, ignore_index=True), name)
    sidecar = write_run_record(path, cfg, args.ablate, infos)
    print(f"Wrote {path} and {sidecar.name}")


def write_run_record(predictions_path: Path, cfg: dict, ablate, infos: list[dict]) -> Path:
    """`<model>_run.json` beside the predictions: per fold-horizon the validation-chosen threshold,
    base rates, test metrics and timing, plus the config and code version. The threshold is not
    a `predictions` column (SCHEMA.md), so Evaluation reads it from here."""
    import json
    from datetime import datetime, timezone

    from src.graph.build_edges import _code_version

    name = model_name(ablate, cfg.get("model", MODEL))
    record = {
        "model": name, "ablate": sorted(ablate), "written_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"), **_code_version(), "config": cfg, "folds": infos,
    }
    out = predictions_path.with_name(f"{name}_run.json")
    out.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
