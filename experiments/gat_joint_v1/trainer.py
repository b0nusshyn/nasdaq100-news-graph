"""Isolated copy of the shared fold trainer with an experimental objective injection."""
import math
import time

import numpy as np
import pandas as pd

from src.models.train import (
    MODEL,
    Fitted,
    GraphSources,
    _chunks,
    build_model,
    fold_inputs,
    make_batch,
    model_name,
    training_partitions,
)


def train_fold(panel, edges: pd.DataFrame | GraphSources | None, fold, horizon: int, cfg: dict, *,
               ablate=(), predict_test=True,
               log=print, return_model=False, objective=None):
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
            logits = run(b)
            loss = (objective(logits, b) if objective is not None else
                    loss_fn(logits[b["target"]], b["y"][b["target"]]) / b["target"].sum().clamp(min=1))
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
        state = Fitted(model, features, graph, np.empty(0, dtype=int), horizon, tr["batch_days"])
        return (None, info, state) if return_model else (None, info)
    test_idx = days_with_rows(fold.test)
    name = model_name(ablate, cfg.get("model", MODEL))
    preds = predict(test_idx).assign(model=name, horizon=horizon, fold=fold.index)
    info["test_rows"] = len(preds)
    if return_model:
        return preds, info, Fitted(model, features, graph, test_idx, horizon, tr["batch_days"])
    return preds, info

