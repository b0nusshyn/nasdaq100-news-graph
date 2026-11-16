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
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "gat_lstm"
NEWS = "news"  # --ablate value that drops every news feature, embeddings included


def model_name(ablate=(), base: str = MODEL) -> str:
    """`base` (the config's `model`, default `gat_lstm`), or `<base>_no_<a>_no_<b>` for an
    ablation. The file name and the `model` column of the predictions."""
    return base + "".join(f"_no_{a}" for a in sorted(ablate))


def fold_edges(edges: pd.DataFrame, panel, fold: int, edge_types: tuple[str, ...]) -> dict[str, np.ndarray]:
    """This fold's edges (its own rows plus the shared `sector` rows) as ticker-index arrays."""
    from src.core.schema import ALL_FOLDS

    if not edge_types:  # Baseline 6 and the all-edges ablation: no graph, edges not needed
        empty = np.array([], dtype=np.int64)
        return {"src": empty, "dst": empty, "type": empty, "weight": np.array([], dtype=np.float32)}
    e = edges[edges["fold"].isin([fold, ALL_FOLDS]) & edges["edge_type"].isin(edge_types)]
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
        "at": np.concatenate(at),
    }


def _chunks(days: np.ndarray, size: int):
    return [days[i: i + size] for i in range(0, len(days), size)]


def train_fold(panel, edges: pd.DataFrame, fold, horizon: int, cfg: dict, *, ablate=(), log=print):
    """Fit one model on `fold`'s training days and predict its test days.

    Returns (predictions in SCHEMA.md `predictions` shape, info dict with timing and epochs).
    """
    import torch

    from src.core.config import walk_forward_config
    from src.models.gat_lstm import GATLSTM
    from src.models.panel import choose_threshold, fit_features, split_validation

    start = time.perf_counter()
    feat, enc, gat, tr = cfg["features"], cfg["encoder"], cfg["gat"], cfg["train"]
    edge_types = tuple(t for t in gat["edge_types"] if t not in ablate)
    no_news = NEWS in ablate
    features = fit_features(
        panel, fold.train, window=cfg["window_days"], price=tuple(feat["price"]),
        news=() if no_news else tuple(feat["news"]),
        embeddings=bool(feat.get("embeddings", True)) and not no_news, embedding_dim=feat["embedding_dim"],
    )
    graph = fold_edges(edges, panel, fold.index, edge_types)
    embargo = cfg.get("embargo_days", walk_forward_config()["embargo_days"])
    fit_days, val_days = split_validation(fold.train, tr["validation_fraction"], embargo)

    def days_with_rows(dates) -> np.ndarray:
        return np.flatnonzero(panel.row_mask(dates, horizon).any(axis=1))

    fit_idx, val_idx, test_idx = days_with_rows(fit_days), days_with_rows(val_days), days_with_rows(fold.test)
    val_batches = [make_batch(c, panel, features, graph, horizon) for c in _chunks(val_idx, tr["batch_days"])]

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = GATLSTM(len(features.names), enc["hidden"], gat["heads"], edge_types,
                    layers=enc["layers"], dropout=enc["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    def run(b):
        logits, _ = model(b["x"], b["edge_index"], b["edge_type"], b["edge_weight"])
        return logits

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            total = sum(float(loss_fn(run(b)[b["target"]], b["y"][b["target"]])) for b in val_batches)
        return total / sum(int(b["target"].sum()) for b in val_batches)

    best, best_state, stale, history = math.inf, None, 0, []
    for epoch in range(tr["epochs"]):
        model.train()
        for chunk in _chunks(rng.permutation(fit_idx), tr["batch_days"]):
            b = make_batch(chunk, panel, features, graph, horizon)
            loss = loss_fn(run(b)[b["target"]], b["y"][b["target"]]) / b["target"].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            if tr.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
        v = val_loss()
        history.append(v)
        if v < best - 1e-6:
            best, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
        log(f"  h={horizon} fold {fold.index} epoch {epoch + 1}: val loss {v:.4f}")
        if stale >= tr["early_stopping_patience"]:
            break
    model.load_state_dict(best_state)

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
    name = model_name(ablate, cfg.get("model", MODEL))
    preds = predict(test_idx).assign(model=name, horizon=horizon, fold=fold.index)
    info = {
        "fold": fold.index, "horizon": horizon, "seconds": time.perf_counter() - start,
        "epochs": len(history), "best_epoch": int(np.argmin(history)) + 1, "best_val_loss": best,
        "threshold": threshold, "val_base_rate": float(val["y_true"].mean()),
        "fit_days": len(fit_idx), "val_days": len(val_idx), "test_rows": len(preds),
        "features": len(features.names), "edges_per_day": len(graph["src"]),
    }
    return preds, info


def fold_report(preds: pd.DataFrame, info: dict) -> dict:
    """Test metrics for one fold-horizon: AUC (no threshold) first, then accuracy, F1 and MCC at
    the validation-chosen threshold, with the base rate that always predicting "up" scores."""
    import warnings

    from src.eval.metrics import classification_metrics

    y, p = preds["y_true"].to_numpy(), preds["p_up"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # undefined MCC is reported as NaN
        m = classification_metrics(y, p, threshold=info["threshold"])
    return {"auc": m["auc"], "threshold": info["threshold"], "accuracy": m["accuracy"], "base_rate": float(y.mean()),
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
    p.add_argument("--ablate", nargs="*", default=[], help="edge types or 'news' to drop (#41)")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="train only these folds (smoke run): prints metrics, writes no predictions")
    args = p.parse_args()

    from src.core.config import load_config
    from src.core.io import read_table, write_predictions
    from src.models.panel import load_panel

    cfg = load_config(args.config)
    bad = set(args.ablate) - set(cfg["gat"]["edge_types"]) - {NEWS}
    if bad:
        p.error(f"--ablate: unknown {sorted(bad)}; choose from {cfg['gat']['edge_types'] + [NEWS]}")
    t0 = time.perf_counter()
    name = model_name(args.ablate, cfg.get("model", MODEL))
    embeddings = bool(cfg["features"].get("embeddings", True)) and NEWS not in args.ablate
    panel = load_panel(embeddings=embeddings)
    edges = read_table("edges") if set(cfg["gat"]["edge_types"]) - set(args.ablate) else None
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
                  f"({info['epochs']} epochs, best {info['best_epoch']}) | AUC {m['auc']:.3f} | "
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
