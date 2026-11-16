"""Company graph edges. Owner: Modelling. Issues #8, #21, #22, #28.

Output: `edges.parquet` (SCHEMA.md) and `edges_manifest.json` beside it. `sector` edges are
static (fold = -1, shared by every fold); `comention` and `corr` weights are computed per
walk-forward fold from that fold's training window only.

    python -m src.graph.build_edges --types sector comention corr            # every fold
    python -m src.graph.build_edges --types sector comention corr --fold 0   # one fold

Without `--fold`, every fold of `configs/walk_forward.yaml` is rebuilt and the requested types
are replaced wholesale. With `--fold N`, only fold N's `comention` / `corr` rows are replaced;
`sector` rows, if requested, are rewritten too, and those are the rows every other fold reads.
The other folds keep rows from earlier runs, so the file then mixes builds: the manifest records
each (fold, edge type)'s parameters, inputs and build time, and the run warns. Rebuild in full
before handing the file off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.schema import (
    ALL_FOLDS,
    COMENTION_MIN_COUNT,
    COMENTION_TOP_K,
    CORR_MIN_OVERLAP,
    CORR_RESIDUAL,
    CORR_TOP_K,
    EDGE_TYPES,
    conform,
)
from src.core.tickers import canonical_ticker, company, same_company
from src.eval.splitter import Fold

# Static input, not a SCHEMA.md artifact: committed with the code so every fold and every
# teammate builds the same sector edges.
GICS_CSV = Path(__file__).with_name("gics.csv")
GICS_LEVELS = ("gics_sector", "gics_industry", "gics_sub_industry")
SECTOR_LEVEL = "gics_industry"
MANIFEST_NAME = "edges_manifest.json"
EDGE_COLUMNS = ["fold", "src", "dst", "edge_type", "weight"]
WEIGHT_RANGE = {"sector": (1.0, 1.0), "comention": (0.0, 1.0), "corr": (-1.0, 1.0)}


def load_gics(path: Path = GICS_CSV) -> pd.DataFrame:
    """The hand-checked GICS table, one row per company, keyed by current ticker."""
    gics = pd.read_csv(path, dtype="string")
    gics["ticker"] = gics["ticker"].map(canonical_ticker)
    dupes = sorted(set(gics.loc[gics["ticker"].duplicated(), "ticker"]))
    if dupes:
        raise ValueError(f"{path}: companies listed more than once: {dupes}")
    blank = gics.loc[gics[list(GICS_LEVELS)].isna().any(axis=1), "ticker"].tolist()
    if blank:
        raise ValueError(f"{path}: missing GICS classification for {blank}")
    return gics


def check_gics_coverage(gics: pd.DataFrame, membership: pd.DataFrame) -> None:
    """Raise if any ticker that was ever a member has no GICS row (it would drop out of the graph)."""
    missing = sorted(set(membership["ticker"]) - set(gics["ticker"]))
    if missing:
        raise ValueError(f"no GICS classification for former or current members: {missing}")


def sector_edges(gics: pd.DataFrame, level: str = SECTOR_LEVEL) -> pd.DataFrame:
    """Both directions between every pair sharing a GICS `level`; weight 1, fold -1."""
    tickers = gics[["ticker", level]]
    pairs = tickers.merge(tickers, on=level, suffixes=("_src", "_dst"))
    pairs = pairs[pairs["ticker_src"] != pairs["ticker_dst"]]
    edges = pd.DataFrame({
        "fold": ALL_FOLDS,
        "src": pairs["ticker_src"],
        "dst": pairs["ticker_dst"],
        "edge_type": "sector",
        "weight": 1.0,
    })
    return conform(edges, "edges")


def comention_pairs(
    article_tickers: pd.DataFrame,
    days: pd.DatetimeIndex,
    *,
    max_companies: int | None = None,
    inverse: bool = True,
) -> pd.DataFrame:
    """Co-mention statistics for every unordered pair (`a` < `b`) over the trading `days` given.

    Only articles dated inside `days` count: pass a fold's training window, never more. k is the
    number of distinct *companies* an article names (share classes count once). Each article
    weighs w = 1 / (k - 1), so a roundup naming twenty firms is weak evidence about any one
    pair; with `inverse=False` every article weighs 1. Articles naming more than `max_companies`
    companies are dropped. Two share classes of one company are never a pair.

    Columns: `n_articles` (raw count of articles naming both), `co` (their summed weight),
    `n_a` / `n_b` (summed weight of all articles naming each firm) and
    `jaccard` = co / (n_a + n_b - co), in [0, 1] because co <= min(n_a, n_b).
    """
    at = article_tickers.loc[article_tickers["date"].isin(days), ["article_id", "ticker"]]
    at = at.drop_duplicates().assign(company=lambda d: d["ticker"].map(company))
    k = at.groupby("article_id")["company"].transform("nunique")
    if max_companies is not None:
        at, k = at[k <= max_companies], k[k <= max_companies]
    at = at.assign(w=1.0 / (k - 1).clip(lower=1) if inverse else 1.0)

    n = at.groupby("ticker")["w"].sum()
    multi = at[k > 1]
    pairs = multi.merge(multi[["article_id", "ticker", "company"]], on="article_id", suffixes=("_a", "_b"))
    pairs = pairs[(pairs["ticker_a"] < pairs["ticker_b"]) & (pairs["company_a"] != pairs["company_b"])]
    out = (
        pairs.groupby(["ticker_a", "ticker_b"])["w"]
        .agg(n_articles="size", co="sum")
        .reset_index()
        .rename(columns={"ticker_a": "a", "ticker_b": "b"})
    )
    out["n_a"] = out["a"].map(n).astype(float)
    out["n_b"] = out["b"].map(n).astype(float)
    out["jaccard"] = out["co"] / (out["n_a"] + out["n_b"] - out["co"])
    return out


def top_k_both_ways(pairs: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """Undirected `a`, `b`, `weight` -> directed `src`, `dst`, `weight` rows, both directions.

    Each ticker ranks its peers by `weight` (ties by peer name, so the result is deterministic)
    and keeps the top `k`; a pair picked by either side is kept, in both directions. `k=None`
    keeps every pair. Shared by `comention` and `corr` edges (SCHEMA.md).
    """
    cols = ["src", "dst", "weight"]
    both = pd.concat(
        [pairs[["a", "b", "weight"]].set_axis(cols, axis=1), pairs[["b", "a", "weight"]].set_axis(cols, axis=1)],
        ignore_index=True,
    )
    both = both.sort_values(["src", "weight", "dst"], ascending=[True, False, True], kind="stable")
    if k is not None:
        keep = set(_undirected(both.groupby("src").head(k)))
        both = both[np.array([p in keep for p in _undirected(both)], dtype=bool)]
    return both.sort_values(["src", "dst"]).reset_index(drop=True)


def _undirected(edges: pd.DataFrame) -> list[tuple[str, str]]:
    """(lower, higher) ticker of each `src`, `dst` row, so both directions share one key."""
    src, dst = edges["src"].tolist(), edges["dst"].tolist()
    return [(s, d) if s < d else (d, s) for s, d in zip(src, dst, strict=True)]


def comention_edges(
    article_tickers: pd.DataFrame,
    folds: list[Fold],
    *,
    top_k: int | None = COMENTION_TOP_K,
    min_count: int = COMENTION_MIN_COUNT,
    max_companies: int | None = None,
    inverse: bool = True,
) -> pd.DataFrame:
    """Per fold: weight = co-mention Jaccard over the training window (`comention_pairs`).

    A pair needs `min_count` training-window articles naming both firms to be a candidate; each
    ticker then keeps its `top_k` peers by weight (`top_k_both_ways`). Defaults are SCHEMA.md
    decision 6; the keyword arguments exist for the 3.9 ablations.
    """
    parts = []
    for fold in folds:
        p = comention_pairs(article_tickers, fold.train, max_companies=max_companies, inverse=inverse)
        p = p[p["n_articles"] >= min_count].rename(columns={"jaccard": "weight"})
        parts.append(top_k_both_ways(p, top_k).assign(fold=fold.index, edge_type="comention"))
    columns = ["fold", "src", "dst", "edge_type", "weight"]
    edges = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return conform(edges, "edges")


def market_model(prices: pd.DataFrame, days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """Daily returns over the trading `days` given, with the equal-weight market removed.

    Pass a fold's training window, never more. Candidates are the tickers `in_index` on at least
    one of `days`; each uses every return it has on those days, in the index or not (all past
    data). Returns are close-to-close on the `prices` calendar, so the first day's return uses
    the previous close. The market m_t is the mean return of the tickers in the index on day t;
    each candidate's beta is the OLS slope of its returns on m over its own observed days.

    Returns (`residual`, `beta`): residual r_i - alpha_i - beta_i * m (dates x tickers, NaN where
    a ticker has no return), and beta per ticker.
    """
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    member = prices.pivot(index="date", columns="ticker", values="in_index").reindex_like(close)
    member = member.fillna(False).astype(bool)
    ret = close.pct_change(fill_method=None)
    ret, member = ret[ret.index.isin(days)], member[member.index.isin(days)]
    market = ret.where(member).mean(axis=1)

    ret = ret.loc[:, member.any()]
    seen = ret.notna()
    m = pd.DataFrame(np.where(seen, market.to_numpy()[:, None], np.nan), ret.index, ret.columns)
    dm, dr = m - m.mean(), ret - ret.mean()
    beta = (dm * dr).sum() / (dm**2).sum()
    alpha = ret.mean() - beta * m.mean()
    residual = ret - alpha - m.mul(beta, axis=1)
    return residual, beta


def corr_pairs(prices: pd.DataFrame, days: pd.DatetimeIndex, *, residual: bool = CORR_RESIDUAL) -> pd.DataFrame:
    """Return correlation for every unordered pair (`a` < `b`) of candidates over the trading `days`.

    Candidates and returns as in `market_model`; pass a fold's training window, never more. With
    `residual` (the default) the weight is the partial correlation given the equal-weight
    market: each stock's fitted market beta is removed before correlating, so two high-beta
    names are not related merely because both move a lot with the market. `residual=False`
    correlates raw returns (ablation). Two share classes of one company are never a pair.

    Columns: `n_obs` (days on which both have a return) and `corr` (Pearson over those days).
    """
    resid, _ = market_model(prices, days)
    if not residual:
        close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
        resid = close.pct_change(fill_method=None).reindex(index=resid.index, columns=resid.columns)
    seen = resid.notna().astype(float)
    n_obs = seen.T @ seen
    corr = resid.corr(min_periods=2)
    tickers = resid.columns.tolist()
    ia, ib = np.triu_indices(len(tickers), k=1)
    out = pd.DataFrame({
        "a": np.array(tickers, dtype=object)[ia],
        "b": np.array(tickers, dtype=object)[ib],
        "n_obs": n_obs.to_numpy()[ia, ib].astype(int),
        "corr": corr.to_numpy()[ia, ib],
    })
    keep = out["corr"].notna() & ~np.array([same_company(a, b) for a, b in zip(out["a"], out["b"], strict=True)])
    return out[keep].reset_index(drop=True)


def corr_edges(
    prices: pd.DataFrame,
    folds: list[Fold],
    *,
    top_k: int | None = CORR_TOP_K,
    min_overlap: int = CORR_MIN_OVERLAP,
    residual: bool = CORR_RESIDUAL,
) -> pd.DataFrame:
    """Per fold: weight = training-window return correlation (`corr_pairs`), stored both ways.

    A pair needs `min_overlap` training days on which both firms have a return to be a
    candidate; each ticker then keeps its `top_k` peers by weight (`top_k_both_ways`). Share
    classes of one company never pair. Defaults are SCHEMA.md decision 7; the keyword arguments
    exist for the 3.9 ablations.
    """
    parts = []
    for fold in folds:
        p = corr_pairs(prices, fold.train, residual=residual)
        p = p[p["n_obs"] >= min_overlap].rename(columns={"corr": "weight"})
        parts.append(top_k_both_ways(p, top_k).assign(fold=fold.index, edge_type="corr"))
    columns = ["fold", "src", "dst", "edge_type", "weight"]
    edges = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return conform(edges, "edges")


# --------------------------------------------------------------------------- edges.parquet

# Inputs whose content determines each edge type. `prices` sets the fold calendar, so the
# per-fold types depend on it even when, like `comention`, they read no price.
INPUTS = {"sector": ("gics.csv", "membership"), "comention": ("article_tickers", "prices"), "corr": ("prices",)}


def default_params() -> dict[str, dict]:
    """Builder keyword arguments per edge type, at the SCHEMA.md defaults (decisions 4, 6, 7)."""
    return {
        "sector": {"level": SECTOR_LEVEL},
        "comention": {
            "top_k": COMENTION_TOP_K, "min_count": COMENTION_MIN_COUNT, "max_companies": None, "inverse": True,
        },
        "corr": {"top_k": CORR_TOP_K, "min_overlap": CORR_MIN_OVERLAP, "residual": CORR_RESIDUAL},
    }


def check_edges(edges: pd.DataFrame) -> None:
    """Raise ValueError listing every way `edges` breaks the SCHEMA.md graph rules.

    `validate` checks columns and dtypes; this checks the graph: one row per key, no self-loops,
    `sector` only at fold -1 and the per-fold types only at folds >= 0, no `comention` / `corr`
    edge between share classes of one company, every edge stored in both directions with the
    same weight, and every weight inside its type's range.
    """
    errors = []
    key = ["fold", "src", "dst", "edge_type"]
    if n := edges.duplicated(key).sum():
        errors.append(f"{n} duplicate (fold, src, dst, edge_type) rows")
    if n := (edges["src"] == edges["dst"]).sum():
        errors.append(f"{n} self-loops")
    sector = edges["edge_type"] == "sector"
    if n := (sector & (edges["fold"] != ALL_FOLDS)).sum():
        errors.append(f"{n} sector rows outside fold {ALL_FOLDS}")
    if n := (~sector & (edges["fold"] < 0)).sum():
        errors.append(f"{n} comention/corr rows with a negative fold")
    per_fold = edges[~sector]
    if n := sum(same_company(s, d) for s, d in zip(per_fold["src"], per_fold["dst"], strict=True)):
        errors.append(f"{n} comention/corr rows between share classes of one company")
    if n := edges["weight"].isna().sum():
        errors.append(f"{n} missing weights")
    reverse = edges.rename(columns={"src": "dst", "dst": "src"})
    both = edges.merge(reverse, on=key, how="left", suffixes=("", "_rev"))
    if n := both["weight_rev"].isna().sum():
        errors.append(f"{n} edges not stored in both directions")
    paired = both.dropna(subset=["weight", "weight_rev"])
    if n := (~np.isclose(paired["weight"], paired["weight_rev"])).sum():
        errors.append(f"{n} edges whose two directions have different weights")
    for t, (lo, hi) in WEIGHT_RANGE.items():
        w = edges.loc[edges["edge_type"] == t, "weight"]
        if n := ((w < lo) | (w > hi)).sum():
            errors.append(f"{n} {t} weights outside [{lo:g}, {hi:g}]")
    if errors:
        raise ValueError("edges: " + "; ".join(errors))


def summarize(edges: pd.DataFrame) -> pd.DataFrame:
    """One row per (fold, edge_type): rows, nodes, weight range and the highest-degree node.

    Edges are stored both ways, so a node's degree is its count of `src` rows; ties go to the
    first ticker alphabetically.
    """
    rows = {}
    for (fold, t), e in edges.groupby(["fold", "edge_type"], sort=True):
        deg = e["src"].value_counts().sort_index()
        rows[(fold, t)] = {
            "rows": len(e),
            "nodes": pd.concat([e["src"], e["dst"]]).nunique(),
            "weight min": float(e["weight"].min()),
            "weight max": float(e["weight"].max()),
            "max degree": f"{deg.max()} ({deg.idxmax()})",
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["fold", "edge_type"])


def fold_graphs(edges: pd.DataFrame) -> pd.DataFrame:
    """The graph each walk-forward fold trains on: its own rows plus the shared `sector` rows."""
    rows = {}
    for fold in sorted(f for f in edges["fold"].unique() if f != ALL_FOLDS):
        e = edges[edges["fold"].isin([fold, ALL_FOLDS])]
        deg = e["src"].value_counts().sort_index()
        rows[fold] = {
            "rows": len(e),
            **{t: int((e["edge_type"] == t).sum()) for t in EDGE_TYPES},
            "nodes": pd.concat([e["src"], e["dst"]]).nunique(),
            "mean degree": deg.mean(),
            "max degree": f"{deg.max()} ({deg.idxmax()})",
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("fold")


def summary_text(edges: pd.DataFrame, n_folds: int | None = None) -> str:
    """Markdown summary of an `edges` table that has passed `check_edges`."""
    from src.graph.comention_report import md

    present = sorted(int(f) for f in edges["fold"].unique())
    lines = [f"Folds present: {', '.join(map(str, present))}"]
    if n_folds is not None:
        missing = sorted(set(range(n_folds)) - set(present))
        status = f"MISSING {missing}" if missing else "all present"
        lines[0] += f" (walk_forward.yaml gives folds 0 to {n_folds - 1}: {status})"
    sector = edges[edges["edge_type"] == "sector"]
    classes = sum(s < d and same_company(s, d) for s, d in zip(sector["src"], sector["dst"], strict=True))
    lines += [
        "", "**Per fold and edge type**", "", md(summarize(edges), digits=3),
        "", "**Graph each fold trains on** (its own rows + the shared `sector` rows)", "",
        md(fold_graphs(edges), digits=1),
        "", "Checks passed (`check_edges`): no duplicate keys, no self-loops, no `comention` / `corr` "
        "edge between share classes of one company, every edge stored both ways with equal weight, "
        "every weight inside its type's range. "
        f"`sector` pairs between share classes, which SCHEMA.md allows: {classes}.",
    ]
    return "\n".join(lines)


def manifest_path() -> Path:
    from src.core.paths import table_path

    return table_path("edges").with_name(MANIFEST_NAME)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_version() -> dict:
    from src.core.paths import REPO_ROOT

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        commit, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
        return {"git_commit": commit, "git_dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _load_manifest(edges_path: Path) -> tuple[dict, list[str]]:
    """Entries keyed (fold, edge_type) from the manifest beside `edges_path`, and any notes.

    A manifest describes the file it was written with; if the file has changed since (edited by
    hand, or copied from another build), its entries no longer apply and are discarded.
    """
    path = edges_path.with_name(MANIFEST_NAME)
    if not path.exists() or not edges_path.exists():
        return {}, []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("edges_sha256") != _sha256(edges_path):
        return {}, [f"{edges_path.name} changed since {MANIFEST_NAME} was written; its old entries were discarded"]
    return {(e["fold"], e["edge_type"]): e for e in manifest["entries"]}, []


def provenance_conflicts(entries: dict, keys: set[tuple[int, str]]) -> list[str]:
    """Ways the (fold, edge_type) row groups `keys` of one edges file were not built alike.

    Compares each type's parameters across folds, and every input hash, the walk-forward
    settings and the code version across all entries; also flags row groups with no entry.
    """
    out = []
    unknown = sorted(keys - set(entries))
    if unknown:
        out.append("no manifest entry, so unknown provenance, for " + ", ".join(f"fold {f} {t}" for f, t in unknown))

    def differs(label: str, ents: list[dict], get) -> None:
        seen: dict[str, list[str]] = {}
        for e in ents:
            seen.setdefault(json.dumps(get(e), sort_keys=True), []).append(f"fold {e['fold']} {e['edge_type']}")
        if len(seen) > 1:
            out.append(f"{label} differ: " + "; ".join(f"{v} for {', '.join(who)}" for v, who in seen.items()))

    ents = sorted(entries.values(), key=lambda e: (e["fold"], e["edge_type"]))
    for t in EDGE_TYPES:
        differs(f"{t} parameters", [e for e in ents if e["edge_type"] == t], lambda e: e["params"])
    for name in sorted({n for e in ents for n in e["inputs"]}):
        differs(f"{name} inputs", [e for e in ents if name in e["inputs"]], lambda e, n=name: e["inputs"][n][:12])
    differs("walk-forward settings", [e for e in ents if e["edge_type"] != "sector"], lambda e: e["walk_forward"])
    differs("code versions", ents, lambda e: e["git_commit"])
    dirty = [f"fold {e['fold']} {e['edge_type']}" for e in ents if e["git_dirty"]]
    if dirty:
        out.append("built from uncommitted changes: " + ", ".join(dirty))
    return out


@dataclass(frozen=True)
class EdgeBuild:
    """What `build_edges_file` wrote: the full `edges` table, the fold count, and warnings."""

    edges: pd.DataFrame
    n_folds: int
    warnings: list[str]


def build_edges_file(
    types=EDGE_TYPES,
    fold: int | None = None,
    *,
    cfg: dict | None = None,
    params: dict[str, dict] | None = None,
    gics_path: Path = GICS_CSV,
) -> EdgeBuild:
    """Build `types` and merge them into `edges.parquet`, recording each in `edges_manifest.json`.

    `fold=None` builds every fold of `cfg` (default `configs/walk_forward.yaml`) and replaces
    every row of the requested types. `fold=N` replaces only fold N's `comention` / `corr` rows,
    plus the `sector` rows if requested, and warns that the file now mixes builds. `params`
    overrides `default_params()` per type (ablations, short test fixtures). The merged table is
    checked (`check_edges`) before anything is written, so a bad build leaves the old file.
    """
    from src.core.config import walk_forward_config
    from src.core.io import read_table, write_table
    from src.core.paths import table_path
    from src.graph.comention_report import shared_folds

    types = [t for t in EDGE_TYPES if t in types]
    unknown = set(params or {}) - set(EDGE_TYPES)
    if unknown:
        raise ValueError(f"params for unknown edge types: {sorted(unknown)}")
    kw = default_params()
    for t, override in (params or {}).items():
        kw[t] = {**kw[t], **override}
    cfg = walk_forward_config() if cfg is None else cfg
    settings = ("train_days", "test_days", "step_days", "expanding", "embargo_days")
    # YAML reads the period bounds as dates; the manifest stores them as ISO strings.
    wf = json.loads(json.dumps({"period": cfg["period"], **{k: cfg[k] for k in settings}}, default=str))

    prices = read_table("prices", columns=["date", "ticker", "close", "in_index"])
    all_folds = shared_folds(prices, cfg)
    if not all_folds:
        raise ValueError("the walk-forward settings give no folds on these prices")
    if fold is not None and not 0 <= fold < len(all_folds):
        raise ValueError(f"fold {fold} does not exist: the walk-forward settings give folds 0 to {len(all_folds) - 1}")
    folds = all_folds if fold is None else [all_folds[fold]]

    parts, sources = [], {}
    if "sector" in types:
        gics = load_gics(gics_path)
        check_gics_coverage(gics, read_table("membership"))
        parts.append(sector_edges(gics, **kw["sector"]))
        sources["gics.csv"] = gics_path
    if "comention" in types:
        parts.append(comention_edges(read_table("article_tickers"), folds, **kw["comention"]))
    if "corr" in types:
        parts.append(corr_edges(prices, folds, **kw["corr"]))
    new = pd.concat(parts, ignore_index=True)

    path = table_path("edges")
    existing = read_table("edges") if path.exists() else conform(pd.DataFrame(columns=EDGE_COLUMNS), "edges")
    entries, warnings = _load_manifest(path)
    built = {f.index for f in folds}

    def rebuilt(f: int, t: str) -> bool:
        return t in types and (t == "sector" or fold is None or f in built)

    keep = ~np.array([rebuilt(f, t) for f, t in zip(existing["fold"], existing["edge_type"], strict=True)], dtype=bool)
    edges = pd.concat([existing[keep], new], ignore_index=True)
    edges = edges.sort_values(["fold", "edge_type", "src", "dst"], kind="stable").reset_index(drop=True)
    check_edges(conform(edges, "edges"))

    for name in {n for t in types for n in INPUTS[t]} - {"gics.csv"}:
        sources[name] = table_path(name)
    hashes = {name: _sha256(p) for name, p in sources.items()}
    stamp, code = datetime.now(timezone.utc).isoformat(timespec="seconds"), _code_version()
    entries = {k: e for k, e in entries.items() if not rebuilt(*k)}
    windows = {ALL_FOLDS: None} | {
        f.index: {"start": f"{f.train[0]:%Y-%m-%d}", "end": f"{f.train[-1]:%Y-%m-%d}", "days": len(f.train)}
        for f in folds
    }
    for t in types:
        for f in [ALL_FOLDS] if t == "sector" else sorted(built):
            entries[(f, t)] = {
                "fold": f, "edge_type": t, "built_at_utc": stamp,
                "rows": int(((new["fold"] == f) & (new["edge_type"] == t)).sum()),
                "params": kw[t], "train_window": windows[f],
                "walk_forward": None if t == "sector" else wf,
                "inputs": {n: hashes[n] for n in INPUTS[t]}, **code,
            }

    # Serialise first: an entry JSON cannot store fails here, before the old table is replaced.
    body = json.loads(json.dumps([entries[k] for k in sorted(entries)]))
    out = write_table(edges, "edges")
    manifest = {"edges_sha256": _sha256(out), "written_at_utc": stamp, "entries": body}
    out.with_name(MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    keys = set(zip(edges["fold"].astype(int), edges["edge_type"], strict=True))
    warnings += provenance_conflicts(entries, keys)
    if fold is not None:
        per_fold = " and ".join(t for t in types if t != "sector") or "no per-fold"
        what = f"only fold {fold}'s {per_fold} rows were rebuilt"
        if "sector" in types:
            what += f", and the sector rows (fold {ALL_FOLDS}) that every fold shares were rewritten"
        warnings.insert(0, f"--fold {fold}: {what}. Other folds keep rows from earlier runs, so {out.name} now "
                           f"mixes builds (see {MANIFEST_NAME}). Rebuild in full, without --fold, before a hand-off.")
    return EdgeBuild(edges, len(all_folds), warnings)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--types", nargs="+", choices=EDGE_TYPES, default=list(EDGE_TYPES))
    p.add_argument(
        "--fold", type=int, default=None,
        help="rebuild only this walk-forward fold (0-based); requested sector rows, shared by every fold, are "
             "rewritten too. Omit to rebuild every fold, which is what a hand-off needs.",
    )
    args = p.parse_args()
    result = build_edges_file(args.types, args.fold)
    from src.core.paths import table_path

    print(f"Wrote {table_path('edges')} and {MANIFEST_NAME}\n")
    print(summary_text(result.edges, result.n_folds))
    for w in result.warnings:
        print(f"WARNING: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
