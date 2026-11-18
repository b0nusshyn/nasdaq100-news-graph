"""Worked interpretability examples for the report (task 2.10, #43). Owner: NLP.

Each example is one test prediction of the official proposed model, traced with
`src.nlp.interpret` (2.9) on Modelling's 3.10 outputs, and written to `results/interpretability/`:

- `<name>.md`: the trace (prediction, graph contribution per edge type, neighbours, headlines)
- `<name>.csv`: every (shown neighbour, article in the window) row behind it
- `<name>.png`: the figure for the report (see `figure`)

    python -m src.nlp.examples              # all of EXAMPLES
    python -m src.nlp.examples --only sgen  # one, by name prefix

The examples come from `results/attention/top_contributions.csv` (largest within-day graph
contribution, 1-day horizon) and were picked to show three different situations; the reading of
each is in `results/interpretability/README.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.nlp.interpret import explain, format_trace

# name, date, ticker, horizon
EXAMPLES = [
    ("sgen_2023-05-17_h1", "2023-05-17", "SGEN", 1),  # target without news, one comention edge, right
    ("lcid_2023-04-17_h1", "2023-04-17", "LCID", 1),  # intuitive corr pair (RIVN), right
    ("goog_2023-05-09_h1", "2023-05-09", "GOOG", 1),  # concentrated attention on META after earnings, wrong
]

# Reference data-viz palette: first three categorical slots (validated all-pairs), diverging blue/red.
TYPE_COLORS = {"sector": "#2a78d6", "comention": "#eb6834", "corr": "#1baf7a"}
POS, NEG, NEUTRAL = "#2a78d6", "#e34948", "#c3c2b7"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def results_dir() -> Path:
    from src.core.paths import RESULTS_DIR

    return RESULTS_DIR / "interpretability"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(t: dict, path: Path | None = None):
    """Three panels: how much each edge type moved the logit, which neighbours the GAT weighted
    (attention against the uniform share), and the main neighbour's daily headline sentiment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r, nb, ev, window = t["prediction"], t["neighbours"], t["evidence"], t["window"]
    types = list(dict.fromkeys(nb["edge_type"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    date = pd.Timestamp(r["date"]).date()
    fig.suptitle(f"{r['ticker']} on {date}, {int(r['horizon'])}-day horizon: P(up) {float(r['p_up']):.2f} "
                 f"(without graph {float(r['p_up_no_graph']):.2f}), realised "
                 f"{'up' if int(r['y_true']) == 1 else 'down'}", x=0.01, ha="left", fontsize=11, color=INK)

    # (a) graph contribution to the logit
    ax = axes[0]
    _style(ax)
    rows = [(f"{e}", float(r[f"graph_logit_{e}"]), TYPE_COLORS.get(e, NEUTRAL)) for e in types]
    rows += [("all edges", float(r["graph_logit"]), INK_2),
             ("all, within-day", float(r["graph_logit_within_day"]), INK_2)]
    labels, values, colors = zip(*rows[::-1], strict=True)
    ax.barh(labels, values, color=colors, height=0.55)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    lim = max(abs(v) for v in values) * 1.6
    ax.set_xlim(-lim, lim)
    for i, v in enumerate(values):
        ax.text(v + (0.04 * lim if v >= 0 else -0.04 * lim), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.set_title("(a) Graph's push on the logit", loc="left", fontsize=9, color=INK)
    ax.set_xlabel("logit change (edge type removed)", fontsize=8, color=INK_2)

    # (b) attention of the shown neighbours vs the uniform share
    ax = axes[1]
    _style(ax)
    nb = nb.iloc[::-1].reset_index(drop=True)
    ylab = [f"{s}  ({e})" for s, e in zip(nb["src"], nb["edge_type"], strict=True)]
    for i, e in nb.iterrows():
        k = r.get(f"in_degree_{e['edge_type']}")
        uniform = 1 / int(k) if k is not None and pd.notna(k) and int(k) > 0 else None
        c = TYPE_COLORS.get(e["edge_type"], NEUTRAL)
        ax.hlines(i, 0, e["alpha_mean"], color=c, linewidth=2)
        ax.plot(e["alpha_mean"], i, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        if uniform is not None:
            ax.plot(uniform, i, "|", color=INK, markersize=12, markeredgewidth=1.5)
        arts = ev[(ev["src"] == e["src"]) & (ev["edge_type"] == e["edge_type"])]
        note = f"w {float(e['weight']):.3f}; " + (f"{arts['date'].nunique()} news days" if len(arts) else "no news")
        if len(arts) and "sentiment" in arts:
            note += f", sent. {arts['sentiment'].mean():+.2f}"
        ax.text(1.03, i, note, va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(nb)), ylab, fontsize=8)
    ax.set_xlim(0, 1.75)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("(b) Attention per neighbour (| = uniform 1/k)", loc="left", fontsize=9, color=INK)
    ax.set_xlabel("attention, mean of heads", fontsize=8, color=INK_2)

    # (c) daily headline sentiment of the neighbour on the strongest edge type
    ax = axes[2]
    _style(ax)
    top = t["neighbours"].iloc[0]
    arts = ev[(ev["src"] == top["src"]) & (ev["edge_type"] == top["edge_type"])]
    daily = arts.groupby("date")["sentiment"].mean() if len(arts) and "sentiment" in arts else pd.Series(dtype=float)
    daily = daily.reindex(window)
    x = range(len(window))
    vals = daily.fillna(0).to_numpy()
    ax.bar(x, vals, color=[POS if v > 0 else NEG for v in vals], width=0.7)
    for i, v in enumerate(daily):
        if pd.isna(v):
            ax.plot(i, 0, "o", color=NEUTRAL, markersize=3)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ticks = [0, len(window) // 2, len(window) - 1]
    ax.set_xticks(ticks, [window[i].strftime("%d %b") for i in ticks], fontsize=8)
    ax.set_title(f"(c) {top['src']} headline sentiment, input window", loc="left", fontsize=9, color=INK)
    ax.set_ylabel("mean FinBERT p_pos - p_neg", fontsize=8, color=INK_2)
    ax.text(0, -1.0, "dot = no news that day", fontsize=7, color=INK_2)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if path is not None:
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
    return fig


def sentiment_alignment(directory: Path | None = None, window: int = 20) -> pd.DataFrame:
    """Does the graph push a prediction the way its neighbours' news points? Over every test
    prediction: Spearman correlation and sign agreement between `graph_logit_within_day` and the
    attention-weighted mean headline sentiment of its neighbours over the input window (also
    demeaned within the day). One row per horizon."""
    import numpy as np

    from src.core.io import read_table
    from src.nlp.interpret import attention_dir

    directory = directory or attention_dir()
    contrib = pd.read_parquet(directory / "gat_lstm_contribution.parquet",
                              columns=["horizon", "date", "ticker", "graph_logit_within_day"])
    edges = pd.read_parquet(directory / "gat_lstm_edges.parquet",
                            columns=["horizon", "date", "src", "dst", "alpha_mean"])
    nf = read_table("news_features", columns=["date", "ticker", "sentiment", "n_articles"])
    for df in (contrib, edges, nf):
        df["date"] = pd.to_datetime(df["date"])
    nf = nf.sort_values(["ticker", "date"]).assign(s=lambda d: d["sentiment"] * d["n_articles"])
    roll = nf.groupby("ticker")[["s", "n_articles"]].rolling(window, min_periods=1).sum()
    roll = roll.reset_index(level=0, drop=True)
    nf["win_sent"] = roll["s"] / roll["n_articles"].replace(0, np.nan)  # NaN = no news in the window
    src_sent = nf[["date", "ticker", "win_sent"]].rename(columns={"ticker": "src"})
    e = edges.merge(src_sent, on=["date", "src"], how="left")
    e = e.dropna(subset=["win_sent"]).assign(w=lambda d: d["alpha_mean"] * d["win_sent"])
    nbr = e.groupby(["horizon", "date", "dst"]).agg(w=("w", "sum"), a=("alpha_mean", "sum")).reset_index()
    nbr = nbr.assign(nbr_sent=nbr["w"] / nbr["a"]).rename(columns={"dst": "ticker"})
    m = contrib.merge(nbr[["horizon", "date", "ticker", "nbr_sent"]], on=["horizon", "date", "ticker"])
    m["nbr_sent_wd"] = m["nbr_sent"] - m.groupby(["horizon", "date"])["nbr_sent"].transform("mean")
    rows = []
    for h, d in m.groupby("horizon"):
        g = d["graph_logit_within_day"]
        rows.append({"horizon": h, "predictions": len(d),
                     "spearman": g.corr(d["nbr_sent_wd"], method="spearman"),
                     "sign_agreement": (np.sign(g) == np.sign(d["nbr_sent_wd"])).mean()})
    return pd.DataFrame(rows)


def run(examples=EXAMPLES, out: Path | None = None, titles: bool = True) -> list[Path]:
    out = out or results_dir()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, date, ticker, horizon in examples:
        t = explain(date, ticker, horizon, titles=titles)
        (out / f"{name}.md").write_text(format_trace(t) + "\n", encoding="utf-8")
        t["evidence"].to_csv(out / f"{name}.csv", index=False)
        figure(t, out / f"{name}.png")
        written.append(out / f"{name}.png")
        print(f"{name}: P(up) {float(t['prediction']['p_up']):.3f}, graph logit "
              f"{float(t['prediction']['graph_logit']):+.2f} -> {out / name}.{{md,csv,png}}", flush=True)
    align = sentiment_alignment()
    align.to_csv(out / "sentiment_alignment.csv", index=False)
    print(align.round(4).to_string(index=False))
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", default=None, help="run only the examples whose name starts with this")
    p.add_argument("--no-titles", action="store_true", help="skip reading headlines from the 1 GB raw CSV")
    args = p.parse_args()
    chosen = [e for e in EXAMPLES if args.only is None or e[0].startswith(args.only)]
    if not chosen:
        p.error(f"no example starts with {args.only!r}")
    run(chosen, titles=not args.no_titles)


if __name__ == "__main__":
    main()
