"""Verify run completeness and replay the best recorded checkpoint without fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval.run_baselines import load_inputs
from src.eval.splitter import Fold
from src.models.panel import build_panel
from src.models.train import _chunks, build_model, make_batch

ROOT = Path(__file__).parent / "runs"


def main():
    torch.set_num_threads(2)
    records = [json.loads(line) for line in (ROOT / "ledger.jsonl").read_text().splitlines()]
    assert len(records) == len({r["id"] for r in records}) == 24
    common = {}
    for run in records:
        data = pd.read_parquet(ROOT / f'{run["id"]}.parquet')
        actual = data[["date", "ticker", "y_true"]]
        key = run["fold"], run["horizon"]
        pd.testing.assert_frame_equal(actual, common.setdefault(key, actual))
        assert np.isfinite(data.p_up).all()
        assert (ROOT / f'{run["id"]}.pt').is_file()
    best = max(records, key=lambda r: r["metrics"]["within_day_auc"])
    prices, labels, empty_news = load_inputs(include_news=False, legacy_labels=True)
    panel = build_panel(prices, labels, empty_news, embeddings=False)
    cfg = best["config"]
    fit_start = pd.Timestamp(best["dates"]["fit"][0])
    val_end = pd.Timestamp(best["dates"]["validation"][1])
    assess_start, assess_end = map(pd.Timestamp, best["dates"]["assessment"][:2])
    fold = Fold(best["fold"], panel.days[(panel.days >= fit_start) & (panel.days <= val_end)],
                panel.days[(panel.days >= assess_start) & (panel.days <= assess_end)])
    # FoldInputs with a prebuilt graph still requires the exact fit-only dates for features.
    from src.models.panel import fit_features
    from src.models.train import fold_edges

    fit_end = pd.Timestamp(best["dates"]["fit"][1])
    dates = panel.days[(panel.days >= fit_start) & (panel.days <= fit_end)]
    cfg_feat = cfg["features"]
    features = fit_features(panel, dates, window=cfg["window_days"], price=tuple(cfg_feat["price"]),
                            news=(), embeddings=False)
    types = tuple(cfg["gat"]["edge_types"])
    edges = pd.read_parquet(ROOT / f'graph_{best["fold"]}.parquet')
    graph = fold_edges(edges, panel, best["fold"], types)
    model = build_model(len(features.names), types, cfg)
    model.load_state_dict(torch.load(ROOT / f'{best["id"]}.pt', map_location="cpu", weights_only=True))
    model.eval()
    horizon = best["horizon"]
    indices = np.flatnonzero(panel.row_mask(fold.test, horizon).any(axis=1))
    scores = []
    with torch.inference_mode():
        for chunk in _chunks(indices, cfg["train"]["batch_days"]):
            b = make_batch(chunk, panel, features, graph, horizon)
            logits, _ = model(b["x"], b["edge_index"], b["edge_type"], b["edge_weight"], node_day=b["node_day"])
            scores.append(logits.sigmoid()[b["target"]].numpy())
    expected = pd.read_parquet(ROOT / f'{best["id"]}.parquet').p_up
    np.testing.assert_allclose(np.concatenate(scores), expected, atol=1e-7, rtol=0)
    print(f'24 model outputs on matched keys verified. Checkpoint replay matched {best["id"]}.')


if __name__ == "__main__":
    main()
