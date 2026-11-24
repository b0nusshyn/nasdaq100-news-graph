"""PROTOTYPE, throwaway (branch prototype/demo-ui). Not production code: no tests, minimal error handling.

Question it answers: what should a local demo UI for the project look like?
Three structurally different variants of one page, switchable with ?variant=A|B|C or the bottom bar:
  A  Walkthrough  - the notebook's 8 steps as a guided stepper
  B  Dashboard    - the within-day AUC ladder and ablations on one screen
  C  Explorer     - pick a day and a company, trace its prediction through the graph

Run from the repo root:  python notebooks/demo_ui_prototype/serve.py   (then open http://localhost:8765)
Reads precomputed artifacts (predictions, 3.10 attention files, results/) and never trains. The what-if
reruns the saved 3.10 fold checkpoints (needs torch) on your edited inputs, in memory only.
Uses $NDX_DATA_DIR, else data/, else falls back to tests/fixtures (marked SYNTHETIC in the UI).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.core import paths  # noqa: E402

LADDER = {"lstm_price": "B5 price only", "lstm_news": "B6 + news", "gat_lstm": "Proposed + graph"}
EXAMPLES = [("SGEN", "2023-05-17", 1), ("GOOG", "2023-05-09", 1), ("LCID", "2023-04-17", 1)]


def pick_data_dir() -> tuple[Path, bool]:
    if os.environ.get(paths.DATA_DIR_ENV):
        d = paths.data_dir()
        return d, d.resolve() == paths.FIXTURES_DIR.resolve()
    if (REPO / "data" / "predictions").is_dir() and any((REPO / "data" / "predictions").glob("*.parquet")):
        return REPO / "data", False
    os.environ[paths.DATA_DIR_ENV] = str(paths.FIXTURES_DIR)
    return paths.FIXTURES_DIR, True


DATA, SYNTHETIC = pick_data_dir()
os.environ[paths.DATA_DIR_ENV] = str(DATA)
ATT = DATA / "attention"

from src.core.config import load_config  # noqa: E402
from src.core.io import read_table  # noqa: E402
from src.models.ladder_report import auc, within_day_auc  # noqa: E402


def records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return json.loads(df.to_json(orient="records"))


def clean(x):
    """NaN/inf -> None and numpy scalars -> Python, so the browser's JSON.parse accepts it."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return x


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# ---------------------------------------------------------------- loaded once, kept in memory
print(f"[prototype] data dir: {DATA}{'  (SYNTHETIC fixtures)' if SYNTHETIC else ''}")
PREDS = {p.stem: read_table("predictions", p) for p in sorted((DATA / "predictions").glob("*.parquet"))}
MAIN = "gat_lstm" if "gat_lstm" in PREDS else next(iter(PREDS))
CONTRIB_PATH = ATT / "gat_lstm_contribution.parquet"
CONTRIB = pd.read_parquet(CONTRIB_PATH) if CONTRIB_PATH.is_file() else None
if CONTRIB is not None:
    CONTRIB["date"] = pd.to_datetime(CONTRIB["date"])
LINKS = read_table("article_tickers")
PRICES = read_table("prices", columns=["date", "ticker", "in_index"])
CALENDAR = pd.DatetimeIndex(sorted(PRICES["date"].unique()))
EDGES = read_table("edges")
print(f"[prototype] loaded {len(PREDS)} prediction files, contribution={'yes' if CONTRIB is not None else 'no'}")


def frame() -> pd.DataFrame:
    """Per (horizon, date, ticker): the main model's prediction, plus graph contribution when available."""
    if CONTRIB is not None:
        return CONTRIB
    p = PREDS[MAIN].copy()
    p["date"] = pd.to_datetime(p["date"])
    return p


@lru_cache(maxsize=1)
def meta() -> dict:
    from src.graph.comention_report import shared_folds

    wf = load_config("walk_forward.yaml")
    try:
        folds = [{"fold": f.index, "train_start": f.train[0], "train_end": f.train[-1],
                  "test_start": f.test[0], "test_end": f.test[-1], "train_days": len(f.train),
                  "test_days": len(f.test)}
                 for f in shared_folds(PRICES.groupby("date", as_index=False)["in_index"].any())]
    except Exception as e:  # fixtures are too short for a full fold
        print(f"[prototype] folds unavailable: {e}")
        folds = []
    f = frame()
    run = read_json(DATA / "predictions" / "gat_lstm_run.json") or {}
    cfg = run.get("config") or load_config("gat_lstm.yaml")
    per_fold = [{k: r.get(k) for k in ("fold", "horizon", "seconds", "epochs", "best_epoch", "best_val_within_day_auc",
                                       "threshold", "fit_days", "val_days", "features", "edges_per_day")}
                | {"test_" + k: (r.get("test") or {}).get(k)
                   for k in ("within_day_auc", "auc", "accuracy", "base_rate")}
                for r in run.get("folds", [])]
    edge_counts = EDGES.groupby(["fold", "edge_type"]).size().rename("n").reset_index()
    members = read_table("membership")
    return {
        "data_dir": str(DATA), "synthetic": SYNTHETIC, "main_model": MAIN,
        "models": list(PREDS), "ladder": {k: v for k, v in LADDER.items() if k in PREDS},
        "has_attention": CONTRIB is not None, "whatif": wi_ready(),
        "horizons": sorted(int(h) for h in f["horizon"].unique()),
        "dates": sorted(f["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "tickers": sorted(f["ticker"].unique().tolist()),
        "folds": records(pd.DataFrame(folds)) if folds else [],
        "walk_forward": {k: wf.get(k) for k in ("train_days", "test_days", "embargo_days", "step_days")},
        "config": cfg, "training": per_fold,
        "counts": {"companies_ever_in_index": int(members["ticker"].nunique()),
                   "trading_days": len(CALENDAR), "article_links": len(LINKS),
                   "articles": int(LINKS["article_id"].nunique()),
                   "test_rows": int(len(f)), "news_share": float(f["has_news"].mean()) if "has_news" in f else None},
        "edges": records(edge_counts),
        "examples": [{"ticker": t, "date": d, "horizon": h} for t, d, h in EXAMPLES],
        "replay": read_json(REPO / "experiments" / "historical_demo_v1" / "development_replay.json"),
    }


@lru_cache(maxsize=1)
def results() -> dict:
    """Within-day AUC (headline) and fold AUC per model, horizon and fold, computed from the predictions."""
    rows = []
    for name, df in PREDS.items():
        for (h, fold), g in df.groupby(["horizon", "fold"]):
            rows.append({"model": name, "horizon": int(h), "fold": int(fold), "within_day_auc": within_day_auc(g),
                         "auc": auc(g["y_true"].to_numpy(), g["p_up"].to_numpy()),
                         "base_rate": float(g["y_true"].mean()), "rows": int(len(g))})
    verdict = {}
    if set(LADDER) <= set(PREDS):  # the pre-registered 3.8 flag, computed by the repo's own code (never reworded)
        from src.models.ladder_report import auc_table, horizon_pattern, steps

        lad = {k: PREDS[k] for k in LADDER}
        verdict = {kind: horizon_pattern(steps(auc_table(lad, kind))) for kind in ("within_day", "fold")}
    return {"rows": rows, "labels": LADDER, "verdict": verdict}


def day(date: str, h: int) -> dict:
    f = frame()
    d = f[(f["date"] == pd.Timestamp(date)) & (f["horizon"] == h)]
    cols = [c for c in ("ticker", "fold", "p_up", "p_up_no_graph", "y_true", "has_news", "graph_logit",
                        "graph_logit_within_day") if c in d]
    out = {"date": date, "horizon": h, "rows": records(d[cols].sort_values("p_up", ascending=False))}
    if len(d) > 1:
        out["within_day_auc"] = auc(d["y_true"].to_numpy(), d["p_up"].to_numpy())
        if "p_up_no_graph" in d:
            out["within_day_auc_no_graph"] = auc(d["y_true"].to_numpy(), d["p_up_no_graph"].to_numpy())
    return out


def ticker(t: str, h: int) -> dict:
    f = frame()
    d = f[(f["ticker"] == t) & (f["horizon"] == h)].sort_values("date")
    cols = [c for c in ("date", "fold", "p_up", "p_up_no_graph", "y_true", "has_news", "graph_logit") if c in d]
    return {"ticker": t, "horizon": h, "rows": records(d[cols])}


def news_days(src: str, days: pd.DatetimeIndex) -> list[int]:
    a = LINKS[(LINKS["ticker"] == src) & LINKS["date"].isin(days)]
    counts = a.groupby("date")["article_id"].nunique()
    return [int(counts.get(d, 0)) for d in days]


def trace(t: str, date: str, h: int) -> dict:
    from src.nlp import interpret

    n = int(meta()["config"].get("window_days", 20))
    days = interpret.window_days(CALENDAR, date, n)
    base = {"ticker": t, "date": date, "horizon": h, "window": [d.strftime("%Y-%m-%d") for d in days],
            "own_news": news_days(t, days)}
    if CONTRIB is None:  # fixtures: no 3.10 outputs, show the static graph neighbours only
        e = EDGES[(EDGES["dst"] == t) & (EDGES["fold"] == -1)]
        nb = [{"src": r.src, "edge_type": r.edge_type, "weight": float(r.weight), "alpha_mean": None,
               "news": news_days(r.src, days)} for r in e.itertuples()]
        return base | {"available": False, "neighbours": nb}
    row, edges = interpret.load_prediction(date, t, h, ATT)
    order = interpret.type_order(row, edges["edge_type"].unique())
    top = interpret.top_neighbours(edges, order, top=5)
    nb = [{"src": r.src, "edge_type": r.edge_type, "weight": float(r.weight), "alpha_mean": float(r.alpha_mean),
           "alpha_heads": [float(getattr(r, f"alpha_{i}")) for i in range(4) if hasattr(r, f"alpha_{i}")],
           "news": news_days(r.src, days)} for r in top.itertuples()]
    types = []
    for et in order:
        k = row.get(f"in_degree_{et}")
        types.append({"edge_type": et, "graph_logit": float(row.get(f"graph_logit_{et}", float("nan"))),
                      "in_degree": int(k) if pd.notna(k) else int((edges["edge_type"] == et).sum())})
    heads = REPO / "results" / "interpretability" / f"{t.lower()}_{date}_h{h}.csv"
    headlines = []
    if heads.is_file():  # NLP's precomputed traces carry headlines and FinBERT sentiment
        hd = pd.read_csv(heads).dropna(subset=["title"])
        hd = hd.assign(_s=hd["sentiment"].abs()).sort_values("_s", ascending=False).drop_duplicates(["src", "title"])
        top3 = hd.groupby(["edge_type", "src"]).head(3)
        headlines = records(top3[["edge_type", "src", "date", "sentiment", "title"]])
    pred = {k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else (v.item() if hasattr(v, "item") else v))
            for k, v in row.items()}
    return base | {"available": True, "prediction": pred, "types": types, "neighbours": nb, "headlines": headlines}


# ---------------------------------------------------------------- what-if: your own news, rerun the saved model
# Loads the panel and the 3.10 fold checkpoints (torch) on first use, edits one company's news inputs in
# memory, reruns the model on that day, then puts the original inputs back. Nothing is written to disk.
WI_LOCK = threading.Lock()
WI: dict = {}


def wi_ready() -> bool:
    return CONTRIB is not None and (ATT / "checkpoints" / "checkpoints.json").is_file()


def wi_fitted(h: int, fold: int):
    from src.models.attention import load_fitted
    from src.models.panel import load_panel

    if "panel" not in WI:
        WI["panel"] = load_panel()
    if (h, fold) not in WI:
        WI[(h, fold)] = load_fitted(h, fold, panel=WI["panel"], edges=EDGES)
    return WI["panel"], WI[(h, fold)]


def whatif(t: str, date: str, h: int, has_news: bool, sentiment: float, confidence: float, days: int) -> dict:
    import numpy as np

    from src.models.attention import explain_days

    if not wi_ready():
        raise ValueError("needs the saved fold checkpoints in <data>/attention/checkpoints (real data only)")
    f = frame()
    on = f[(f["date"] == pd.Timestamp(date)) & (f["horizon"] == h)]
    if on.empty or t not in set(on["ticker"]):
        raise ValueError(f"no {h}d test prediction for {t} on {date}")
    fold = int(on["fold"].iloc[0])
    sentiment, confidence = min(1.0, max(-1.0, sentiment)), min(1.0, max(0.0, confidence))
    with WI_LOCK:
        panel, fitted = wi_fitted(h, fold)
        feat, names = fitted.features, fitted.features.names
        d, k = int(panel.day_index([date])[0]), int(panel.tickers.get_loc(t))
        cells = list(range(max(0, d - min(days, feat.window) + 1), d + 1))
        emb = [i for i, n in enumerate(names) if n.startswith("emb_pca_")]
        had_text = bool((panel.news[cells, k, 0] > 0).any())
        saved_x, saved_news = feat.x[cells, k].copy(), panel.news[cells, k].copy()
        before, _ = explain_days(fitted, panel, np.array([d]))
        try:
            for c in cells:
                if has_news:  # scaled exactly as fit_features does, with this fold's training statistics
                    raw = {"has_news": 1.0, "sentiment": sentiment, "confidence": confidence}
                    for name, v in raw.items():
                        if name in names:
                            st = feat.stats.get(name)
                            feat.x[c, k, names.index(name)] = v if st is None else (v - st["mean"]) / (st["std"] or 1.0)
                    panel.news[c, k] = [1.0, sentiment, confidence]
                else:
                    for name in ("has_news", "sentiment", "confidence"):
                        if name in names:
                            feat.x[c, k, names.index(name)] = 0.0
                    feat.x[c, k, emb] = 0.0
                    panel.news[c, k] = 0.0
            after, edges = explain_days(fitted, panel, np.array([d]))
        finally:
            feat.x[cells, k], panel.news[cells, k] = saved_x, saved_news
    rows = before[["ticker", "p_up", "y_true"]].merge(after[["ticker", "p_up"]], on="ticker",
                                                      suffixes=("_before", "_after"))
    rows["delta"] = rows["p_up_after"] - rows["p_up_before"]
    out_edges = edges[edges["src"] == t].groupby("dst")["edge_type"].agg(lambda s: ", ".join(sorted(set(s))))
    rows["link"] = rows["ticker"].map(out_edges).fillna("")
    rows.loc[rows["ticker"] == t, "link"] = "edited"
    official = on.set_index("ticker")["p_up"]
    y, pb, pa = rows["y_true"].to_numpy(), rows["p_up_before"].to_numpy(), rows["p_up_after"].to_numpy()
    return {
        "ticker": t, "date": date, "horizon": h, "fold": fold,
        "edited_days": [panel.days[c].strftime("%Y-%m-%d") for c in cells],
        "input": {"has_news": has_news, "sentiment": sentiment, "confidence": confidence},
        "note": ("article-text embeddings kept from the real articles on those days; only has_news, sentiment and "
                 "confidence were changed" if has_news and had_text else
                 "no article text on the added days, so the embedding features stay 0" if has_news else
                 "all news features, including the text embeddings, set to 0 (no coverage)"),
        "matches_official": float((rows.set_index("ticker")["p_up_before"] - official).abs().max()),
        "within_day_auc_before": auc(y, pb), "within_day_auc_after": auc(y, pa),
        "neighbours_changed": int((~rows["link"].isin(["", "edited"]) & (rows["delta"].abs() > 1e-6)).sum()),
        "others_changed": int(((rows["link"] == "") & (rows["delta"].abs() > 1e-6)).sum()),
        "rows": records(rows.reindex(rows["delta"].abs().sort_values(ascending=False).index)),
    }


# ---------------------------------------------------------------- HTTP
ROUTES = {
    "/api/meta": lambda q: meta(),
    "/api/results": lambda q: results(),
    "/api/day": lambda q: day(q["date"], int(q.get("h", 1))),
    "/api/ticker": lambda q: ticker(q["ticker"], int(q.get("h", 1))),
    "/api/trace": lambda q: trace(q["ticker"], q["date"], int(q.get("h", 1))),
    "/api/whatif": lambda q: whatif(q["ticker"], q["date"], int(q.get("h", 1)), q.get("has_news", "1") == "1",
                                    float(q.get("sentiment", 0)), float(q.get("confidence", 0.9)),
                                    int(q.get("days", 1))),
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        route = ROUTES.get(url.path)
        if route is None:
            return self._send(404, b"not found", "text/plain")
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            body = json.dumps(clean(route(q)), default=str, allow_nan=False).encode()
            self._send(200, body, "application/json")
        except Exception as e:  # surface the error in the page, it's a prototype
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    def log_message(self, fmt, *args) -> None:
        print("[prototype]", self.address_string(), fmt % args)


def main() -> None:
    ap = argparse.ArgumentParser(description="PROTOTYPE local demo UI (three variants).")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if wi_ready():
        try:  # torch's first import must happen on the main thread (segfaults when imported from a worker)
            import torch  # noqa: F401
        except ImportError:
            print("[prototype] torch not installed: the what-if is disabled")
            globals()["wi_ready"] = lambda: False

    def warm() -> None:  # one thread, in order: caches, then the fold-4 1d checkpoint for the first what-if
        meta(), results()
        if wi_ready():
            with WI_LOCK:
                wi_fitted(1, 4)
            print("[prototype] what-if model ready")

    threading.Thread(target=warm, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}/?variant=A"
    print(f"[prototype] serving {url}  (variants A, B, C; Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
