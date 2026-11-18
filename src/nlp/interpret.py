"""Trace a GAT + LSTM prediction back to the edges and source articles behind it. Owner: NLP. Issue #37.

Modelling's task 3.10 (`src.models.attention`, #42) writes, for every test prediction of the
official proposed model, its graph contribution and its incoming edges with their attention
(`<data>/attention/`, columns in `results/attention/README.md`). This module adds the news side:
for one prediction it lists each neighbour's articles in the model's look-back window, with the
FinBERT headline sentiment, giving the chain the report's worked examples need (task 2.10):

    article (headline, sentiment) -> src --[edge type, attention, weight]--> ticker -> P(up)

    python -m src.nlp.interpret --date 2023-05-17 --ticker SGEN --horizon 1
    python -m src.nlp.interpret --example 1 --out results/interpretability/example_1.md

`--example N` takes the N-th row of `results/attention/top_contributions.csv` (largest
within-day graph contribution first).

Reading a trace:
- Attention is a softmax over a company's incoming edges of one type: it ranks neighbours of
  that type and sums to 1 per (ticker, edge type), not across types. In this model it is close
  to uniform (`results/attention/summary.md`), so it mostly says that an edge exists.
- How much the graph moved the prediction is `graph_logit` (and `graph_logit_<type>` per edge
  type); only `graph_logit_within_day` can change the day's ranking, which within-day AUC scores.
  Edge types are listed in order of their contribution, so start reading from the top.
- An edge passes on the neighbour's LSTM state, which summarises its whole window
  (`window_days` trading days up to and including the date), so its articles are listed over
  the window, not just that day. A neighbour without news still passes on its price history.
- One GAT layer: only direct neighbours can move a prediction (no 2-hop paths).
- The direction of a neighbour's effect is not in the attention; compare the sentiment with the
  sign of `graph_logit_<type>`. Treat a trace as a pointer for the examples, not as proof.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODEL = "gat_lstm"
TOP_NEIGHBOURS = 3  # neighbours shown per edge type
TOP_ARTICLES = 3  # articles shown per neighbour (strongest |sentiment| first)
EVIDENCE_COLUMNS = ["edge_type", "src", "alpha_mean", "weight", "article_id", "date"]


def attention_dir() -> Path:
    from src.models.attention import attention_dir as modelling_dir

    return modelling_dir()


def window_days(calendar, date, n: int) -> pd.DatetimeIndex:
    """The `n` trading days of `calendar` up to and including `date` (the model's input window)."""
    cal = pd.DatetimeIndex(pd.to_datetime(pd.Series(calendar).unique())).sort_values()
    date = pd.Timestamp(date)
    if date not in cal:
        raise ValueError(f"{date.date()} is not a trading day")
    return cal[cal <= date][-n:]


def load_prediction(date, ticker: str, horizon: int, directory: Path | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """(contribution row, incoming edges) of one test prediction, from 3.10's Parquet files."""
    directory = directory or attention_dir()
    date = pd.Timestamp(date)
    keys = [("ticker", "==", ticker), ("horizon", "==", horizon)]
    contrib = pd.read_parquet(directory / f"{MODEL}_contribution.parquet", filters=keys)
    contrib = contrib[pd.to_datetime(contrib["date"]) == date]
    if contrib.empty:
        raise ValueError(f"no {horizon}d test prediction for {ticker} on {date.date()}")
    edges = pd.read_parquet(directory / f"{MODEL}_edges.parquet",
                            filters=[("dst", "==", ticker), ("horizon", "==", horizon)])
    edges = edges[pd.to_datetime(edges["date"]) == date]
    return contrib.iloc[0], edges.reset_index(drop=True)


def type_order(row: pd.Series, edge_types) -> list[str]:
    """Edge types by |graph_logit_<type>|, largest first (types without the column last)."""
    def size(t):
        v = row.get(f"graph_logit_{t}")
        return -abs(float(v)) if v is not None and pd.notna(v) else 0.0
    return sorted(edge_types, key=size)


def top_neighbours(edges: pd.DataFrame, order, top: int = TOP_NEIGHBOURS) -> pd.DataFrame:
    """The `top` incoming edges per type by `alpha_mean`, types in `order`."""
    rank = {t: i for i, t in enumerate(order)}
    e = edges.assign(_t=edges["edge_type"].map(rank))
    e = e.sort_values(["_t", "alpha_mean", "weight"], ascending=[True, False, False])
    return e.groupby("edge_type", sort=False).head(top).drop(columns="_t").reset_index(drop=True)


def source_articles(article_tickers: pd.DataFrame, ticker: str, days,
                    articles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Articles assigned to `ticker` on any of `days` (the look-back window), newest first.

    `articles` optionally adds per-article columns keyed by `article_id` (`article_info`).
    """
    window = pd.DatetimeIndex(pd.to_datetime(days))
    rows = article_tickers[(article_tickers["ticker"] == ticker) & article_tickers["date"].isin(window)]
    rows = rows[["article_id", "date", "ticker"]]
    if articles is not None:
        rows = rows.merge(articles, on="article_id", how="left")
    return rows.sort_values(["date", "article_id"], ascending=[False, True], ignore_index=True)


def evidence(neighbours: pd.DataFrame, article_tickers: pd.DataFrame, days,
             articles: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (shown neighbour, article in its window), in the neighbours' order."""
    parts = []
    for _, e in neighbours.iterrows():
        arts = source_articles(article_tickers, e["src"], days, articles).drop(columns="ticker")
        if len(arts):
            parts.append(arts.assign(edge_type=e["edge_type"], src=e["src"],
                                     alpha_mean=e["alpha_mean"], weight=e["weight"]))
    if not parts:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    return out[EVIDENCE_COLUMNS + [c for c in out.columns if c not in EVIDENCE_COLUMNS]]


def article_info(article_ids, raw: Path | None = None, titles: bool = True) -> pd.DataFrame:
    """article_id, sentiment, confidence (FinBERT, task 2.5) and, with `titles`, the headline.

    Headlines come from `raw/fnspid_filtered.csv` (about 1 GB, read in chunks: slow).
    """
    from src.core.paths import raw_dir
    from src.nlp.match_tickers import article_id

    raw = raw or raw_dir()
    ids = sorted(set(article_ids))
    info = pd.read_parquet(raw / "article_features.parquet", columns=["article_id", "sentiment", "confidence"],
                           filters=[("article_id", "in", ids)]) if ids else \
        pd.DataFrame(columns=["article_id", "sentiment", "confidence"])
    if titles and ids:
        wanted, found = set(ids), []
        for chunk in pd.read_csv(raw / "fnspid_filtered.csv", usecols=["Url", "Article_title"], dtype=str,
                                 chunksize=200_000):
            chunk = chunk.dropna(subset=["Url"])
            chunk = chunk.assign(article_id=chunk["Url"].map(article_id))
            found.append(chunk[chunk["article_id"].isin(wanted)][["article_id", "Article_title"]])
        heads = pd.concat(found).drop_duplicates("article_id").rename(columns={"Article_title": "title"})
        info = info.merge(heads, on="article_id", how="left")
    return info


def trace(row: pd.Series, edges: pd.DataFrame, article_tickers: pd.DataFrame, days,
          articles: pd.DataFrame | None = None, top: int = TOP_NEIGHBOURS) -> dict:
    """The chain for one prediction: {"prediction", "neighbours", "evidence", "window"}."""
    order = type_order(row, edges["edge_type"].unique())
    neighbours = top_neighbours(edges, order, top)
    return {"prediction": row, "neighbours": neighbours, "window": pd.DatetimeIndex(days),
            "evidence": evidence(neighbours, article_tickers, days, articles)}


def _f(v, digits: int = 3) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{float(v):+.{digits}f}"


def format_trace(t: dict, max_articles: int = TOP_ARTICLES) -> str:
    """Markdown for one trace: the prediction, then each edge type's neighbours and their news."""
    r, ev = t["prediction"], t["evidence"]
    date = pd.Timestamp(r["date"]).date()
    w = t["window"]
    lines = [
        f"## {r['ticker']} on {date}, {int(r['horizon'])}-day horizon (fold {int(r['fold'])})",
        "",
        f"- P(up) **{float(r['p_up']):.3f}**, without the graph {float(r['p_up_no_graph']):.3f}; "
        f"realised label: {'up' if int(r['y_true']) == 1 else 'down'}; own news that day: "
        f"{'yes' if bool(r['has_news']) else 'no'}",
        f"- Graph logit {_f(r['graph_logit'])}, of which within-day (changes the ranking) "
        f"{_f(r.get('graph_logit_within_day'))}",
        f"- News window: {w[0].date()} to {w[-1].date()} ({len(w)} trading days)",
        "",
    ]
    for etype, nb in t["neighbours"].groupby("edge_type", sort=False):
        k = r.get(f"in_degree_{etype}")
        k = int(k) if k is not None and pd.notna(k) and int(k) > 0 else len(nb)
        lines += [f"### {etype}: graph logit {_f(r.get(f'graph_logit_{etype}'))} "
                  f"({k} incoming edges, uniform attention {1 / k:.3f})", ""]
        for _, e in nb.iterrows():
            arts = ev[(ev["edge_type"] == etype) & (ev["src"] == e["src"])]
            lines.append(f"- **{e['src']}** -> {r['ticker']}: attention {float(e['alpha_mean']):.3f}, "
                         f"edge weight {float(e['weight']):.3f}, {len(arts)} articles in window "
                         f"on {arts['date'].nunique() if len(arts) else 0} days")
            if len(arts) and "sentiment" in arts:
                lines.append(f"  - mean headline sentiment {arts['sentiment'].mean():+.3f}")
                arts = arts.assign(_s=arts["sentiment"].abs()).sort_values(["_s", "date"], ascending=False)
            if "title" in arts:  # the same headline is often syndicated under several URLs
                arts = arts.drop_duplicates("title")
            for _, a in arts.head(max_articles).iterrows():
                title = a.get("title") if isinstance(a.get("title"), str) else a["article_id"]
                lines.append(f"  - {pd.Timestamp(a['date']).date()} [{_f(a.get('sentiment'), 2)}] {title}")
        lines.append("")
    return "\n".join(lines)


def explain(date, ticker: str, horizon: int, *, top: int = TOP_NEIGHBOURS, titles: bool = True,
            directory: Path | None = None) -> dict:
    """Load one prediction from 3.10's outputs and trace it to its neighbours' articles."""
    from src.core.config import load_config
    from src.core.io import read_table
    from src.core.paths import CONFIGS_DIR

    row, edges = load_prediction(date, ticker, horizon, directory)
    n = load_config(CONFIGS_DIR / f"{MODEL}.yaml")["window_days"]
    days = window_days(read_table("prices", columns=["date"])["date"], row["date"], n)
    links = read_table("article_tickers")
    neighbours = top_neighbours(edges, type_order(row, edges["edge_type"].unique()), top)
    ids = links[links["ticker"].isin(set(neighbours["src"])) & links["date"].isin(days)]["article_id"]
    return trace(row, edges, links, days, article_info(ids, titles=titles), top)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", help="prediction day, YYYY-MM-DD")
    p.add_argument("--ticker")
    p.add_argument("--horizon", type=int, choices=[1, 5], default=1)
    p.add_argument("--example", type=int, default=None,
                   help="use row N (1-based) of results/attention/top_contributions.csv instead of --date/--ticker")
    p.add_argument("--top", type=int, default=TOP_NEIGHBOURS, help="neighbours shown per edge type")
    p.add_argument("--articles", type=int, default=TOP_ARTICLES, help="articles shown per neighbour")
    p.add_argument("--no-titles", action="store_true", help="skip reading headlines from the 1 GB raw CSV")
    p.add_argument("--out", type=Path, default=None, help="also write the trace (.md) and its evidence (.csv)")
    args = p.parse_args()

    if args.example is not None:
        from src.models.attention import results_dir

        top = pd.read_csv(results_dir() / "top_contributions.csv")
        ex = top.iloc[args.example - 1]
        args.date, args.ticker, args.horizon = ex["date"], ex["ticker"], int(ex["horizon"])
    if not (args.date and args.ticker):
        p.error("give --date and --ticker, or --example")

    t = explain(args.date, args.ticker, args.horizon, top=args.top, titles=not args.no_titles)
    text = format_trace(t, args.articles)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        t["evidence"].to_csv(args.out.with_suffix(".csv"), index=False)
        print(f"wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
