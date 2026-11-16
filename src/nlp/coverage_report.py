"""News coverage report: share of in-index ticker-days with news, by year and by sector. Owner: NLP. Issue #30.

Feeds the "sparse news" argument: most ticker-days have no article, so `has_news` must stay
distinct from neutral sentiment. Reads `news_features.parquet` (every in-index ticker-day in
the study window) and GICS sectors from `src/graph/gics.csv`.

    python -m src.nlp.coverage_report

Writes to `results/tables/`: `news_coverage_by_year.csv`, `news_coverage_by_sector.csv`,
`news_coverage_year_sector.csv` (share, year x sector); and `results/figures/news_coverage.png`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.core.paths import REPO_ROOT, RESULTS_DIR

GICS_CSV = REPO_ROOT / "src" / "graph" / "gics.csv"
UNKNOWN_SECTOR = "Unknown"


def _summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    per_ticker = df.groupby([by, "ticker"], observed=True)["has_news"].any().groupby(level=0)
    out = pd.DataFrame({
        "ticker_days": g.size(),
        "with_news": g["has_news"].sum(),
        "share_with_news": g["has_news"].mean(),
        "articles_per_covered_day": g.apply(lambda x: x.loc[x["has_news"], "n_articles"].mean(), include_groups=False),
        "tickers": g["ticker"].nunique(),
        "tickers_without_news": per_ticker.apply(lambda s: int((~s).sum())),
    })
    return out.reset_index()


def coverage_tables(news_features: pd.DataFrame, sectors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Coverage by year, by sector, and the year x sector share grid (plus an "All" row/column)."""
    df = news_features[["date", "ticker", "has_news", "n_articles"]].copy()
    df["year"] = df["date"].dt.year
    sector = sectors.drop_duplicates("ticker").set_index("ticker")["gics_sector"]
    df["sector"] = df["ticker"].map(sector).fillna(UNKNOWN_SECTOR)

    by_year = _summary(df, "year")
    by_sector = _summary(df, "sector").sort_values("share_with_news", ascending=False, ignore_index=True)
    grid = df.pivot_table(index="sector", columns="year", values="has_news", aggfunc="mean")
    grid["All"] = df.groupby("sector")["has_news"].mean()
    grid.loc["All"] = [*df.groupby("year")["has_news"].mean(), df["has_news"].mean()]
    order = [*by_sector["sector"], "All"]
    grid = grid.loc[order]
    grid.columns = [str(c) for c in grid.columns]
    return {"by_year": by_year, "by_sector": by_sector, "year_sector": grid.reset_index()}


def plot_grid(grid: pd.DataFrame, counts: pd.Series, path: Path) -> None:
    """Heatmap of share with news, sector x year; one blue sequential ramp, values printed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Reference sequential ramp (blue 100 -> 700), light end = little coverage.
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("coverage", ramp)
    ink, muted = "#1f1f1e", "#6b6a66"

    values = grid.set_index("sector")
    labels = [f"{s} ({counts.get(s, 0)})" if s != "All" else "All sectors" for s in values.index]
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(values) + 1.6), dpi=200)
    fig.patch.set_facecolor("white")
    cmap.set_bad("#f0efec")  # no in-index ticker of that sector that year
    arr = values.to_numpy(dtype=float)
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            text, color = ("—", muted) if pd.isna(v) else (f"{v:.0%}", "white" if v >= 0.5 else ink)
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color,
                    fontweight="bold" if values.index[i] == "All" else "normal")
    # Separate the "All" row and column with a surface-colored gap.
    ax.axhline(len(values) - 1.5, color="white", lw=3)
    ax.axvline(values.shape[1] - 1.5, color="white", lw=3)
    ax.set_xticks(range(values.shape[1]), values.columns, fontsize=8, color=ink)
    ax.set_yticks(range(len(values)), labels, fontsize=8, color=ink)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Share of in-index ticker-days with at least one news article",
                 fontsize=10, color=ink, loc="left", pad=10)
    fig.text(0.01, 0.01, "FNSPID, 2019–2023. Sector = GICS; (n) = tickers in the sector; "
             "— = no index member in that sector that year.",
             fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=RESULTS_DIR, help="results directory (tables/, figures/)")
    args = p.parse_args()

    from src.core.io import read_table

    news = read_table("news_features", columns=["date", "ticker", "has_news", "n_articles"])
    sectors = pd.read_csv(GICS_CSV, usecols=["ticker", "gics_sector"], dtype=str)
    tables = coverage_tables(news, sectors)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(args.out / "tables" / f"news_coverage_{name}.csv", index=False, float_format="%.4f")
    counts = tables["by_sector"].set_index("sector")["tickers"]
    plot_grid(tables["year_sector"], counts, args.out / "figures" / "news_coverage.png")

    print(tables["by_year"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(tables["by_sector"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote tables and figure under {args.out}")


if __name__ == "__main__":
    main()
