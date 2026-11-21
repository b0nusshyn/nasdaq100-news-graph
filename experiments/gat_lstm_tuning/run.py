"""Bounded retrospective GAT-LSTM study. Run from repository root with python -B -m."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.metrics import roc_auc_score

from src.core.config import load_config
from src.core.io import read_table
from src.core.paths import table_path
from src.eval.metrics import classification_metrics
from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold, walk_forward
from src.graph.build_edges import comention_edges, corr_edges
from src.models.panel import build_panel, choose_threshold, fit_features
from src.models.train import build_model, fold_edges, make_batch

ROOT = Path(__file__).resolve().parent
CANDIDATES = {
    "t1_reference": {},
    "epoch_selection": {"cadence": "epoch"},
    "lower_lr": {"cadence": "epoch", "lr": 0.0003},
    "no_comention": {"cadence": "epoch", "edges": ["corr"]},
    "no_graph": {"cadence": "epoch", "edges": []},
    "rank_loss": {"cadence": "epoch", "objective": "within_day_pairwise"},
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def daily_auc(y, p, day):
    return {
        int(d): float(roc_auc_score(y[day == d], p[day == d]))
        for d in np.unique(day)
        if np.unique(y[day == d]).size == 2
    }


def evaluate(model, batches, stride=1):
    model.eval()
    ys, ps, ds = [], [], []
    loss = count = 0
    with torch.no_grad():
        for batch in batches:
            logits = forward(model, batch)
            target = batch["target"]
            y = batch["y"][target]
            z = logits[target]
            loss += float(torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="sum"))
            count += len(y)
            ys.append(y.numpy().astype(int))
            ps.append(z.sigmoid().numpy().astype(float))
            ds.append(batch["at"][target.numpy(), 0])
    y, p, day = map(np.concatenate, (ys, ps, ds))
    by_day = daily_auc(y, p, day)
    phases = [np.mean([v for d, v in by_day.items() if d % stride == phase]) for phase in range(stride)]
    return {
        "bce": loss / count,
        "within_day_auc": float(np.mean(list(by_day.values()))),
        "brier": float(np.mean((p - y) ** 2)),
        "positive_rate": float(np.mean(y)),
        "rows": count,
        "daily_auc": by_day,
        "nonoverlap_phase_auc": phases,
    }, y, p


def forward(model, b):
    return model(b["x"], b["edge_index"], b["edge_type"], b["edge_weight"])[0]


def classification(y, p, threshold):
    metrics = classification_metrics(y, p, threshold=threshold)
    return {key: value if math.isfinite(value) else None for key, value in metrics.items()}


def loss_for(logits, batch, objective):
    target = batch["target"]
    z, y = logits[target], batch["y"][target]
    if objective == "bce":
        return torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    days = batch["at"][target.numpy(), 0]
    terms = []
    for day in np.unique(days):
        on_day = days == day
        positive = z[on_day & (y.numpy() == 1)]
        negative = z[on_day & (y.numpy() == 0)]
        if positive.numel() and negative.numel():
            terms.append(torch.nn.functional.softplus(negative[None, :] - positive[:, None]).mean())
    return torch.stack(terms).mean() if terms else z.sum() * 0


def run_fit(panel, features, edges, fold, val, horizon, name, config, seed, cap):
    start = time.perf_counter()
    tr = config["train"]
    settings = {
        "cadence": 14, "lr": tr["lr"], "edges": ["comention", "corr"], "objective": "bce",
        **CANDIDATES[name],
    }
    kinds = tuple(settings["edges"])
    graph = fold_edges(edges, panel, fold.index, kinds)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(len(features.names), kinds, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=tr["weight_decay"])
    size = tr["batch_days"]

    def batches(days):
        idx = panel.day_index(days)
        return [make_batch(idx[i:i + size], panel, features, graph, horizon) for i in range(0, len(idx), size)]

    validation = batches(val)
    train_probe = batches(fold.train[::5])
    per_epoch = math.ceil(len(fold.train) / size)
    every = per_epoch if settings["cadence"] == "epoch" else settings["cadence"]
    patience_steps = tr["early_stopping_patience"] * per_epoch
    history = []
    step, best_step, best_score = 0, 0, -math.inf
    best_state = None
    finished = False
    for epoch in range(1, cap + 1):
        model.train()
        epoch_losses = []
        for batch in batches(rng.permutation(fold.train)):
            loss = loss_for(forward(model, batch), batch, settings["objective"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            optimizer.step()
            step += 1
            epoch_losses.append(float(loss.detach()))
            if step % every and step != cap * per_epoch:
                continue
            scores, _, _ = evaluate(model, validation)
            score = scores["within_day_auc"]
            history.append({
                "step": step, "epoch": epoch, "validation_auc": score, "validation_bce": scores["bce"],
                "mean_objective_this_epoch_so_far": float(np.mean(epoch_losses)),
            })
            if score > best_score + 1e-6:
                best_step, best_score = step, score
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            finished = step - best_step >= patience_steps
            model.train()
            if finished:
                break
        if finished or epoch == 1 or epoch % 2 == 0 or epoch == cap:
            train, _, _ = evaluate(model, train_probe)
            history.append({
                "step": step, "epoch": epoch, "training_probe_auc": train["within_day_auc"],
                "training_probe_bce": train["bce"],
            })
        if finished:
            break
    if best_state is None:
        raise RuntimeError("No finite validation AUC")
    model.load_state_dict(best_state)
    vs, y, p = evaluate(model, validation, horizon)
    threshold = choose_threshold(y, p)
    vs.update(classification(y, p, threshold))
    assess, y, p = evaluate(model, batches(fold.test), horizon)
    assess.update(classification(y, p, threshold))
    return {
        "candidate": name, "settings": settings, "seed": seed, "fold": fold.index, "horizon": horizon,
        "steps": step, "best_step": best_step, "steps_per_epoch": per_epoch, "interval": every,
        "stop": "patience" if finished else "cap", "threshold": threshold, "seconds": time.perf_counter() - start,
        "validation": vs, "assessment": assess, "history": history, "edge_count": len(graph["src"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--folds", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--horizons", nargs="+", type=int, choices=[1, 5], default=[1, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=16)
    args = parser.parse_args()
    if args.cap <= 0 or Path(args.output).name != args.output or not args.output.endswith(".jsonl"):
        parser.error("Use a positive cap and a new JSONL filename inside the experiment directory.")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config_path = Path("configs/gat_lstm.yaml")
    cfg = load_config(config_path)
    prices, labels, news = load_inputs(include_news=True, legacy_labels=True)
    membership = read_table("membership")
    links = read_table("article_tickers")
    prices = prices[prices.date <= "2020-12-31"]
    calendar = pd.DatetimeIndex(sorted(prices.date.unique()))
    days = calendar[calendar >= "2019-01-01"]
    # Enforce the recorded historical member spells for every observed quote.
    joined = prices[["date", "ticker", "in_index"]].reset_index(drop=True).reset_index().merge(membership, on="ticker")
    active = joined.date.ge(joined.start) & (joined.end.isna() | joined.date.le(joined.end))
    flags = active.groupby(joined["index"]).any().reindex(range(len(prices)), fill_value=False)
    if flags.tolist() != prices.in_index.tolist():
        raise ValueError("Historical membership disagrees with price flags")
    next_session = pd.Series(calendar[1:], index=calendar[:-1])
    news = news.loc[news.date.isin(next_session.index)].assign(date=lambda f: f.date.map(next_session))
    links = links.loc[links.date.isin(next_session.index)].assign(date=lambda f: f.date.map(next_session))
    panel = build_panel(prices, labels[labels.date <= "2020-12-31"], news)
    outer = walk_forward(days, train_days=302, test_days=55, step_days=65, horizon=5, embargo_days=5)[:2]
    paths = [table_path(name) for name in ("prices", "labels", "membership", "news_features", "article_tickers")]
    paths += [config_path, Path(__file__), *sorted(Path("src").rglob("*.py"))]
    manifest = {
        "arguments": vars(args), "original_t1": cfg, "candidate_definitions": CANDIDATES,
        "inputs_and_source_sha256": {str(p): digest(p) for p in paths},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "pyg": torch_geometric.__version__,
                    "platform": platform.platform(), "device": "cpu", "torch_threads": 2},
    }
    with (ROOT / args.output).open("x") as out:
        def emit(row):
            print(json.dumps(row, allow_nan=False), file=out, flush=True)
        emit(manifest)
        for index in args.folds:
            split = outer[index]
            fold = Fold(index, split.train[:-50], split.test)
            val = split.train[-45:]
            for before, after in [(fold.train, val), (val, fold.test)]:
                assert calendar.get_loc(after[0]) - calendar.get_loc(before[-1]) == 6
            assert calendar.get_loc(fold.test[-1]) + 5 < len(calendar)
            ranges = {key: [str(d[0].date()), str(d[-1].date()), len(d)]
                      for key, d in [("fit", fold.train), ("checkpoint", val), ("assessment", fold.test)]}
            features = fit_features(panel, fold.train, window=cfg["window_days"], embedding_dim=64)
            edges = pd.concat([corr_edges(prices, [fold]), comention_edges(links, [fold])], ignore_index=True)
            for h in args.horizons:
                for name in args.candidates:
                    row = run_fit(panel, features, edges, fold, val, h, name, cfg, args.seed, args.cap)
                    emit({**row, "ranges": ranges})
                    print(name, index, h, row["steps"], row["best_step"],
                          round(row["assessment"]["within_day_auc"], 5), round(row["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
