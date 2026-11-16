"""Co-mention measurement report. Owner: Modelling. Issue #21 (task 3.3).

The evidence behind SCHEMA.md decision 6 (Jaccard weight, 1 / (k - 1) article weight,
top-k = 10 after min_count = 2). Reads `article_tickers` and `prices`, builds the shared
walk-forward folds, and writes:

- `results/tables/comention_report.md`: tickers per article, graph density by min_count and
  roundup treatment, Jaccard by co-mention count, and top pairs by raw count vs by weight.
- `results/tables/comention_top_pairs.csv`: each fold's 20 strongest `comention` edges at the
  SCHEMA.md defaults, for the report and presentation.

    python -m src.graph.comention_report --output results/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.paths import RESULTS_DIR

TREATMENTS = {"plain": {"inverse": False}, "cap 8": {"inverse": False, "max_companies": 8}, "inverse": {}}


def md(df: pd.DataFrame, index: bool = True, digits: int = 1) -> str:
    """GitHub Markdown table (pandas' `to_markdown` needs the optional `tabulate`)."""
    d = df.reset_index() if index else df

    def cell(v) -> str:
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{v:,.0f}" if float(v).is_integer() and digits <= 1 else f"{v:,.{digits}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    rows += ["| " + " | ".join(cell(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def companies_per_article(article_tickers: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    from src.core.tickers import company

    at = article_tickers.loc[article_tickers["date"].isin(days), ["article_id", "ticker"]]
    return at.assign(company=at["ticker"].map(company)).groupby("article_id")["company"].nunique()


def roundup_table(k: pd.Series) -> tuple[pd.DataFrame, str]:
    pairs = k * (k - 1) / 2
    bucket = pd.cut(k, [0, 1, 2, 3, 5, 8, 15, 10**6], labels=["1", "2", "3", "4–5", "6–8", "9–15", "16+"])
    t = pd.DataFrame({
        "articles": k.groupby(bucket, observed=False).size(),
        "pairs": pairs.groupby(bucket, observed=False).sum(),
    })
    t["% of multi-company articles"] = 100 * t["articles"] / (k > 1).sum()
    t.loc["1", "% of multi-company articles"] = np.nan
    t["% of pairs"] = 100 * t["pairs"] / pairs.sum()
    multi = k[k > 1].sort_values(ascending=False, kind="stable")
    top = multi.iloc[: int(np.ceil(0.1 * len(multi)))]
    note = (
        f"{len(k):,} articles, {(k > 1).sum():,} naming two or more companies; "
        f"p99 = {k.quantile(0.99):g}, max = {k.max()} companies. The top decile of multi-company "
        f"articles by k (k ≥ {top.min()}) contributes {100 * (top * (top - 1) / 2).sum() / pairs.sum():.1f}% "
        "of all pairs."
    )
    return t.rename_axis("companies named"), note


def degree_row(edges: pd.DataFrame) -> dict:
    """Degree stats of undirected `a`/`b` pairs or directed `src`/`dst` rows stored both ways."""
    deg = pd.concat([edges["a"], edges["b"]]).value_counts() if "a" in edges else edges["src"].value_counts()
    n_edges = len(edges) if "a" in edges else len(edges) // 2
    if deg.empty:
        return {"edges": 0, "nodes": 0}
    return {
        "edges": n_edges, "nodes": len(deg), "mean degree": deg.mean(), "median": deg.median(),
        "p90": deg.quantile(0.9), "max": f"{deg.max()} ({deg.idxmax()})",
    }


def period_days(prices: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DatetimeIndex, dict]:
    """Trading days of `prices` inside the study period, and the walk-forward settings used.

    `cfg` has the shape of `configs/walk_forward.yaml` and defaults to it.
    """
    from src.core.config import walk_forward_config

    cfg = walk_forward_config() if cfg is None else cfg
    days = pd.DatetimeIndex(sorted(prices.loc[prices["in_index"], "date"].unique()))
    days = days[(days >= pd.Timestamp(cfg["period"]["start"])) & (days <= pd.Timestamp(cfg["period"]["end"]))]
    return days, cfg


def shared_folds(prices: pd.DataFrame, cfg: dict | None = None) -> list:
    """The walk-forward folds every model uses (`period_days` explains `cfg`)."""
    from src.eval.splitter import walk_forward

    days, cfg = period_days(prices, cfg)
    settings = {k: cfg[k] for k in ("train_days", "test_days", "step_days", "expanding", "embargo_days")}
    return walk_forward(days, horizon=5, **settings)


def build(article_tickers: pd.DataFrame, prices: pd.DataFrame, cfg: dict | None = None) -> tuple[str, pd.DataFrame]:
    """Report text and top pairs. `cfg` defaults to `configs/walk_forward.yaml`."""
    from src.core.schema import COMENTION_MIN_COUNT, COMENTION_TOP_K
    from src.graph.build_edges import comention_edges, comention_pairs

    days, _ = period_days(prices, cfg)
    folds = shared_folds(prices, cfg)
    shown = [folds[0], folds[-1]]
    out = [
        "# Co-mention edges: measurements (task 3.3, #21)",
        "",
        "Generated by `python -m src.graph.comention_report`. Evidence for SCHEMA.md decision 6. "
        "Share classes of one company count as one company and never pair with each other.",
        "",
        "| fold | training window | days |",
        "|---|---|---|",
        *(f"| {f.index} | {f.train[0]:%Y-%m-%d} → {f.train[-1]:%Y-%m-%d} | {len(f.train)} |" for f in folds),
    ]

    table, note = roundup_table(companies_per_article(article_tickers, days))
    out += ["", "## 1 · Companies per article (2019–2023)", "", md(table), "", note]

    stats = {
        (f.index, name): comention_pairs(article_tickers, f.train, **kw)
        for f in shown for name, kw in TREATMENTS.items()
    }
    out += ["", "## 2 · Density by min_count, before top-k", "",
            "min_count counts raw articles naming both firms, so inverse weighting changes weights, "
            "not this table; `cap 8` drops articles naming more than 8 companies."]
    for f in shown:
        for name in ("plain", "cap 8"):
            p = stats[f.index, name]
            rows = {m: degree_row(p[p["n_articles"] >= m]) for m in (1, 2, 5, 10)}
            out += ["", f"**Fold {f.index}, {name}**", "", md(pd.DataFrame(rows).T.rename_axis("min_count"))]

    edges = comention_edges(article_tickers, folds)
    rows = {f.index: degree_row(edges[edges["fold"] == f.index]) for f in folds}
    out += ["", f"**SCHEMA.md defaults (inverse weight, min_count {COMENTION_MIN_COUNT}, "
            f"top_k {COMENTION_TOP_K}), every fold**", "", md(pd.DataFrame(rows).T.rename_axis("fold"))]
    ablation = comention_edges(article_tickers, folds, min_count=5)
    rows = {f.index: degree_row(ablation[ablation["fold"] == f.index]) for f in folds}
    out += ["", "**Ablation: min_count 5, otherwise the defaults**", "", md(pd.DataFrame(rows).T.rename_axis("fold"))]

    out += ["", "## 3 · Jaccard weight by raw co-mention count", ""]
    for f in shown:
        for name in ("plain", "inverse"):
            p = stats[f.index, name]
            b = pd.cut(p["n_articles"], [0, 1, 4, 9, 49, 10**9], labels=["1", "2–4", "5–9", "10–49", "50+"])
            t = p.groupby(b, observed=False)["jaccard"].describe(percentiles=[0.1, 0.5, 0.9])
            t = t[["count", "10%", "50%", "90%", "max"]].rename(columns={"count": "pairs"})
            t["pairs"] = t["pairs"].astype(int)
            out += [f"**Fold {f.index}, {name}**", "", md(t.rename_axis("articles naming both"), digits=4), ""]

    last = folds[-1].index
    cols = ["a", "b", "n_articles", "jaccard"]
    plain, inverse = stats[last, "plain"], stats[last, "inverse"]
    inverse = inverse[inverse["n_articles"] >= COMENTION_MIN_COUNT]
    out += [f"## 4 · Top 15 pairs, fold {last}: raw count vs weight", "",
            "**By raw co-mention count**", "", md(plain.nlargest(15, "n_articles")[cols], index=False, digits=3),
            "", f"**By weight (inverse Jaccard, min_count {COMENTION_MIN_COUNT}) — the edges the model sees**",
            "", md(inverse.nlargest(15, "jaccard")[cols], index=False, digits=3), ""]

    und = edges[edges["src"] < edges["dst"]]
    top = und.sort_values(["fold", "weight"], ascending=[True, False]).groupby("fold").head(20)
    return "\n".join(out), top[["fold", "src", "dst", "weight"]].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "tables", help="directory for the report files")
    args = p.parse_args()

    from src.core.io import read_table

    report, top = build(read_table("article_tickers"), read_table("prices", columns=["date", "in_index"]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comention_report.md").write_text(report + "\n", encoding="utf-8")
    top.to_csv(args.output / "comention_top_pairs.csv", index=False, float_format="%.4f")
    print(f"wrote {args.output / 'comention_report.md'} and comention_top_pairs.csv")


if __name__ == "__main__":
    main()
