"""Offline: point-in-time SEC fundamentals from filtered FSDS extracts (SEC_LAGGED_PROXY policy).

Three steps, each auditable on its own output:

1. **Filings ledger.** Every original periodic/annual report (amendments are listed but never
   used) with its eligible session from `sec_lagged_proxy.build.eligible_session`: two
   complete NASDAQ sessions after the later of the EDGAR filing date and the Eastern
   acceptance date. This is the same assumption as the earlier proxy, not evidence of exact
   public dissemination time.
2. **Identity.** A ticker-date maps to the CIK of the most recent filing (filed on or before
   that date, within 400 days) whose XBRL instance file prefix equals the ticker or one of its
   historical symbols (`src.core.tickers.RENAMES`, share classes). EDGAR asks filers to use
   their ticker as that prefix, so it is a dated, per-filing record of the symbol in use. Only
   past filings are consulted. Conflicting CIKs inside the window are flagged and unmapped.
3. **As-of join with two separate states.**
   - *event state*: latest eligible original filing of the mapped CIK (any facts or none):
     `sec_sessions_since_filing`, `sec_filing_window` (<= 5 sessions), `sec_is_annual`.
   - *fundamental state*: latest eligible original filing of that CIK **that has verified
     facts**, aged <= MAX_FUND_AGE sessions. A newer filing without extractable facts does
     not erase it. Values: revenue growth, operating margin and its change, net-margin change,
     and a bounded EPS change, every comparison taken from the same filing (current period vs
     the prior-year comparative it reports), so nothing is backfilled or restated later.
   Missing stays NaN with explicit `sec_has_*` flags; nothing is imputed here.

No price level enters any feature: `prices.close` is adjusted for splits after the sample, so
EPS/price would leak future corporate actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.sec_lagged_proxy.build import eligible_session
from src.core.tickers import RENAMES, SHARE_CLASSES, canonical_ticker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = {"10-K", "10-Q", "20-F", "40-F", "10-KT", "10-QT"}
REVENUE = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
           "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense", "Revenue"]
OPINC = ["OperatingIncomeLoss", "ProfitLossFromOperatingActivities"]
NETINC = ["NetIncomeLoss", "ProfitLossAttributableToOwnersOfParent", "ProfitLoss"]
EPS = ["EarningsPerShareDiluted", "DilutedEarningsLossPerShare", "EarningsPerShareBasicAndDiluted"]
FUND = ["sec_rev_growth", "sec_op_margin", "sec_op_margin_chg", "sec_net_margin_chg", "sec_eps_chg"]
EVENT = ["sec_sessions_since_filing", "sec_filing_window", "sec_is_annual"]
MAX_FUND_AGE = 130   # sessions; a quarter is ~63, so one missed report is tolerated, two are not
MAX_EVENT_AGE = 130
IDENTITY_DAYS = 400
REVIEW = HERE / "identity_review.csv"


def load_review(path: Path = REVIEW) -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    """Reviewed identity decisions: blocked (ticker, cik) pairs and extra symbol aliases."""
    r = pd.read_csv(path, dtype=str)
    if not r.action.isin(["block", "alias"]).all() or r.reason.isna().any():
        raise ValueError("identity review rows need action block|alias and a reason")
    blocked = {(t, c) for t, c in r.loc[r.action.eq("block"), ["ticker", "cik"]].itertuples(index=False)}
    aliases: dict[str, set[str]] = {}
    for t, v in r.loc[r.action.eq("alias"), ["ticker", "value"]].itertuples(index=False):
        aliases.setdefault(t, set()).add(v.lower())
    return blocked, aliases
LAG = 2


# ---------------------------------------------------------------- loading
def load_extracts(extract_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    records = [json.loads(p.read_text()) for p in sorted(extract_dir.glob("*.json"))]
    if not records:
        raise FileNotFoundError(f"No FSDS extracts in {extract_dir}")
    subs, nums = [], []
    for r in records:
        q = r["quarter"]
        for kind, bucket in (("sub", subs), ("num", nums)):
            path = extract_dir / f"{q}.{kind}.parquet"
            if _sha(path) != r[f"{kind}_sha256"]:
                raise ValueError(f"{path.name} does not match its recorded hash")
            bucket.append(pd.read_parquet(path).assign(quarter=q))
    sub = pd.concat(subs, ignore_index=True)
    if sub.adsh.duplicated().any():
        raise ValueError("An accession appears in two quarterly data sets")
    return sub, pd.concat(nums, ignore_index=True), records


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 1. filings ledger
def instance_prefix(instance: str) -> str | None:
    m = re.match(r"([a-z][a-z0-9]*)[-_]", str(instance).strip().lower())
    return m.group(1) if m else None


def filings_ledger(sub: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    f = sub.copy()
    f["filed_date"] = pd.to_datetime(f.filed, format="%Y%m%d")
    f["period_end"] = pd.to_datetime(f.period, format="%Y%m%d")
    f["accepted_et"] = pd.to_datetime(f.accepted, errors="coerce")  # EDGAR acceptance is Eastern
    f["original"] = f.form.isin(ORIGINAL)
    f["prefix"] = f.instance.map(instance_prefix)
    f["eligible_date"] = [eligible_session(d, a, "America/New_York", sessions, LAG)
                          for d, a in zip(f.filed_date, f.accepted_et, strict=True)]
    for col in ("filed_date", "period_end", "eligible_date"):
        f[col] = pd.to_datetime(f[col]).astype("datetime64[ns]")
    bad = f.period_end > f.filed_date
    f.loc[bad, "original"] = False  # a report cannot cover a period after its filing date
    return f


# ---------------------------------------------------------------- 2. identity
def ticker_prefixes(ticker: str, aliases: dict[str, set[str]] | None = None) -> set[str]:
    group = next((g for g in SHARE_CLASSES if ticker in g), (ticker,))
    names = {t for t in group} | {old for old in RENAMES if canonical_ticker(old) in group}
    extra = set().union(*((aliases or {}).get(t, set()) for t in group))
    return {n.lower().replace(".", "") for n in names} | extra


def _interleaved(ciks: np.ndarray) -> bool:
    """True if the date-ordered filer sequence alternates (A, B, A): two issuers using one
    symbol concurrently. A clean succession (A..A, B..B) is not a conflict."""
    runs = [c for i, c in enumerate(ciks) if i == 0 or c != ciks[i - 1]]
    return len(runs) > len(set(runs))


def prefix_matches(prefix: str, names: set[str]) -> bool:
    """Exact symbol, or symbol + digits (e.g. 'csx03312010', 'goog10') for symbols of 3+ chars,
    so that short symbols cannot capture other filers' date-style prefixes ('a10')."""
    return any(prefix == n or (len(n) >= 3 and prefix.startswith(n) and prefix[len(n):][:1].isdigit())
               for n in names)


def map_identity(keys: pd.DataFrame, filings: pd.DataFrame,
                 review: tuple[set, dict] | None = None) -> pd.DataFrame:
    """keys: ticker, date. Returns keys + cik, identity_accession, identity_conflict.
    `review` = (blocked (ticker, cik) pairs, extra aliases); default none."""
    blocked, aliases = review or (set(), {})
    out = []
    keys = keys.assign(date=pd.to_datetime(keys.date).astype("datetime64[ns]"))
    ev = filings.dropna(subset=["prefix"]).sort_values("filed_date")
    for ticker, g in keys.groupby("ticker", sort=False):
        names = ticker_prefixes(ticker, aliases)
        cand = ev.loc[ev.prefix.map(lambda x, n=names: prefix_matches(x, n)), ["filed_date", "cik", "adsh"]]
        cand = cand.loc[[(ticker, c) not in blocked for c in cand.cik]]
        g = g.sort_values("date")
        if cand.empty:
            out.append(g.assign(cik=pd.NA, identity_accession=pd.NA, identity_conflict=False))
            continue
        j = pd.merge_asof(g, cand.rename(columns={"filed_date": "id_date"}), left_on="date",
                          right_on="id_date", direction="backward",
                          tolerance=pd.Timedelta(days=IDENTITY_DAYS))
        # conflict: filers of this symbol alternate within 2x the identity look-back
        dates, ciks = cand.filed_date.to_numpy(), cand.cik.to_numpy()
        lo = np.searchsorted(dates, (g.date - pd.Timedelta(days=2 * IDENTITY_DAYS)).to_numpy(), "left")
        hi = np.searchsorted(dates, g.date.to_numpy(), "right")
        cache: dict[tuple[int, int], bool] = {}
        conflict = np.array([cache.setdefault((a, b), _interleaved(ciks[a:b]))
                             for a, b in zip(lo, hi, strict=True)], dtype=bool)
        j["identity_conflict"] = conflict
        j.loc[conflict, ["cik", "adsh"]] = pd.NA
        out.append(j.drop(columns="id_date").rename(columns={"adsh": "identity_accession"}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 3a. facts per filing
def _pick(num: pd.DataFrame, tags: list[str], cur, prior, monetary: bool):
    """(current, prior, uom) for the first tag with an unambiguous current value."""
    for tag in tags:
        t = num.loc[num.tag.eq(tag)]
        if monetary:
            t = t.loc[~t.uom.str.contains("/", regex=False)]
        c = t.loc[t.ddate.eq(cur)]
        if c.value.nunique() != 1 or c.uom.nunique() != 1:
            continue
        uom = c.uom.iloc[0]
        p = t.loc[t.ddate.eq(prior) & t.uom.eq(uom)]
        pv = p.value.iloc[0] if p.value.nunique() == 1 else np.nan
        return c.value.iloc[0], pv, uom
    return np.nan, np.nan, None


def filing_facts(filings: pd.DataFrame, num: pd.DataFrame) -> pd.DataFrame:
    num = num.copy()
    num["ddate"] = pd.to_datetime(num.ddate, format="%Y%m%d")
    groups = dict(tuple(num.groupby("adsh")))
    rows = []
    for f in filings.loc[filings.original].itertuples(index=False):
        n = groups.get(f.adsh)
        rec = {"adsh": f.adsh, **dict.fromkeys(FUND, np.nan), "fact_status": "no_values"}
        if n is None:
            rows.append(rec)
            continue
        qtrs = 1 if f.form in ("10-Q", "10-QT") else 4
        n = n.loc[n.qtrs.eq(qtrs)]
        cur = f.period_end
        prior = (cur - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)
        rev, rev0, ru = _pick(n, REVENUE, cur, prior, True)
        op, op0, ou = _pick(n, OPINC, cur, prior, True)
        ni, ni0, nu = _pick(n, NETINC, cur, prior, True)
        eps, eps0, eu = _pick(n, EPS, cur, prior, False)  # FSDS EPS uom is the currency
        rec["qtrs"], rec["currency"] = qtrs, ru
        ok = ru is not None and rev > 0
        ok0 = ok and rev0 > 0
        if ok:
            if ou == ru:
                rec["sec_op_margin"] = op / rev
                if ok0 and np.isfinite(op0):
                    rec["sec_op_margin_chg"] = op / rev - op0 / rev0
            if ok0:
                rec["sec_rev_growth"] = rev / rev0 - 1
                if nu == ru and np.isfinite(ni0):
                    rec["sec_net_margin_chg"] = ni / rev - ni0 / rev0
        if eu == ru and np.isfinite(eps) and np.isfinite(eps0) and (abs(eps) + abs(eps0)) > 0:
            rec["sec_eps_chg"] = (eps - eps0) / (abs(eps) + abs(eps0))  # bounded in [-1, 1]
        rec["fact_status"] = "values" if any(np.isfinite(rec[k]) for k in FUND) else "no_values"
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3b. as-of join
def build_store(keys: pd.DataFrame, filings: pd.DataFrame, facts: pd.DataFrame,
                sessions: pd.DatetimeIndex, review: tuple[set, dict] | None = None) -> pd.DataFrame:
    """One row per key (ticker, date) with identity, event state and fundamental state."""
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    keys = keys.assign(date=pd.to_datetime(keys.date).astype("datetime64[ns]"))
    mapped = map_identity(keys[["ticker", "date"]], filings, review)
    orig = filings.loc[filings.original & filings.eligible_date.notna(),
                       ["adsh", "cik", "form", "eligible_date", "filed_date", "period_end"]]
    orig = orig.merge(facts, on="adsh", how="left")
    orig["eligible_date"] = pd.to_datetime(orig.eligible_date).astype("datetime64[ns]")
    out = mapped.copy()
    out["sec_has_issuer"] = out.cik.notna()
    for col in EVENT + FUND + ["sec_fund_age", "event_accession", "fund_accession"]:
        out[col] = np.nan if not col.endswith("accession") else pd.NA
    m = out.loc[out.sec_has_issuer].sort_values("date")
    m = m.assign(cik=m.cik.astype(object).map(str))
    orig["cik"] = orig.cik.astype(object).map(str)
    if not m.empty:
        ev = orig.sort_values("eligible_date")
        e = pd.merge_asof(m, ev[["cik", "eligible_date", "adsh", "form"]].rename(columns={"adsh": "e_adsh"}),
                          left_on="date", right_on="eligible_date", by="cik", direction="backward")
        age = e.date.map(pos).to_numpy() - e.eligible_date.map(pos).to_numpy()
        ok = np.isfinite(age) & (age <= MAX_EVENT_AGE)
        e["sec_sessions_since_filing"] = np.where(ok, age, np.nan)
        e["sec_filing_window"] = np.where(ok, (age <= 5).astype(float), np.nan)
        e["sec_is_annual"] = np.where(ok, (e.form != "10-Q").astype(float), np.nan)
        e["event_accession"] = e.e_adsh.where(ok)
        fv = ev.loc[ev.fact_status.eq("values"), ["cik", "eligible_date", "adsh", *FUND]]
        fj = pd.merge_asof(m[["ticker", "date", "cik"]], fv.rename(columns={"adsh": "f_adsh"}),
                           left_on="date", right_on="eligible_date", by="cik", direction="backward")
        fage = fj.date.map(pos).to_numpy() - fj.eligible_date.map(pos).to_numpy()
        fok = np.isfinite(fage) & (fage <= MAX_FUND_AGE)
        for col in FUND:
            fj[col] = fj[col].where(fok)
        fj["sec_fund_age"] = np.where(fok, fage, np.nan)
        fj["fund_accession"] = fj.f_adsh.where(fok)
        state = e[["ticker", "date", *EVENT, "event_accession"]].merge(
            fj[["ticker", "date", *FUND, "sec_fund_age", "fund_accession"]], on=["ticker", "date"])
        out = out.drop(columns=[*EVENT, *FUND, "sec_fund_age", "event_accession", "fund_accession"]).merge(
            state, on=["ticker", "date"], how="left", validate="one_to_one")
    out["sec_has_event"] = out.sec_sessions_since_filing.notna()
    out["sec_has_fund"] = out[FUND].notna().any(axis=1)
    return out


def check_point_in_time(store: pd.DataFrame, filings: pd.DataFrame) -> None:
    """Every accession feeding a row was eligible on or before that row's date, is an original
    report of the mapped CIK, and the identity filing itself was filed on or before the date."""
    f = filings.set_index("adsh")
    for col in ("event_accession", "fund_accession", "identity_accession"):
        used = store.loc[store[col].notna(), ["date", "cik", col]]
        src = f.loc[used[col]]
        when = src.filed_date if col == "identity_accession" else src.eligible_date
        if (when.to_numpy() > used.date.to_numpy()).any():
            raise AssertionError(f"{col}: accession used before it was available")
        if col != "identity_accession" and (
                (src.cik.to_numpy() != used.cik.to_numpy()).any() or not src.original.all()):
            raise AssertionError(f"{col}: wrong issuer or non-original filing")


# ---------------------------------------------------------------- CLI
def assessment_keys() -> pd.DataFrame:
    frames = []
    for i in (0, 1):
        p = ROOT / f"experiments/gat_edge_weight_v1/runs/w{i}_h5_s42_original.parquet"
        frames.append(pd.read_parquet(p, columns=["ticker", "date"]))
    keys = pd.concat(frames, ignore_index=True)
    keys["ticker"] = keys.ticker.astype(str)
    return keys


def all_member_keys() -> pd.DataFrame:
    pr = pd.read_parquet(ROOT / "data/prices.parquet", columns=["ticker", "date", "in_index"])
    k = pr.loc[pr.in_index & pr.date.dt.year.between(2019, 2023), ["ticker", "date"]]
    return k.assign(ticker=k.ticker.astype(str)).reset_index(drop=True)


def sessions_calendar() -> pd.DatetimeIndex:
    import pandas_market_calendars as mcal
    return mcal.get_calendar("NASDAQ").schedule("2018-06-01", "2024-03-31").index


def coverage(store: pd.DataFrame) -> pd.DataFrame:
    sectors = pd.read_csv(ROOT / "src/graph/gics.csv").set_index("ticker").gics_sector
    s = store.assign(year=store.date.dt.year, sector=store.ticker.map(sectors).fillna("unclassified"))
    rows = []
    for y, g in s.groupby("year"):
        row = {"year": y, "rows": len(g), "tickers": g.ticker.nunique()}
        for flag in ("sec_has_issuer", "sec_has_event", "sec_has_fund"):
            h = g.loc[g[flag]]
            row |= {f"{flag}_rows": len(h), f"{flag}_pct": round(100 * len(h) / len(g), 2),
                    f"{flag}_tickers": h.ticker.nunique(), f"{flag}_sectors": h.sector.nunique(),
                    f"{flag}_dates": h.date.nunique()}
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract", type=Path, default=HERE / "extract")
    p.add_argument("--output", type=Path, required=True, help="new directory (must not exist)")
    args = p.parse_args()
    sub, num, records = load_extracts(args.extract)
    sessions = sessions_calendar()
    keys = all_member_keys()
    # Only issuers whose filings ever carry one of our ticker prefixes can map; keeping all
    # filings of those CIKs gives the same store as the full ledger, much faster.
    review = load_review()
    names = set().union(*(ticker_prefixes(t, review[1]) for t in keys.ticker.unique()))
    hit = sub.instance.map(instance_prefix).map(lambda x: isinstance(x, str) and prefix_matches(x, names))
    sub = sub.loc[sub.cik.isin(set(sub.cik[hit]))]
    filings = filings_ledger(sub, sessions)
    facts = filing_facts(filings, num.loc[num.adsh.isin(set(filings.adsh))])
    store = build_store(keys, filings, facts, sessions, review)
    assess = assessment_keys().merge(store, on=["ticker", "date"], how="left", validate="one_to_one")
    if assess.sec_has_issuer.isna().any():
        raise ValueError("Assessment keys outside the member key set")
    check_point_in_time(store, filings)
    args.output.mkdir(parents=True, exist_ok=False)
    filings.drop(columns=["accepted_et"]).to_parquet(args.output / "filings.parquet", index=False)
    facts.to_parquet(args.output / "facts.parquet", index=False)
    store.to_parquet(args.output / "store.parquet", index=False)
    cov_all, cov_assess = coverage(store), coverage(assess)
    cov_all.to_csv(args.output / "coverage_members.csv", index=False)
    cov_assess.to_csv(args.output / "coverage_assessment.csv", index=False)
    unresolved = store.loc[~store.sec_has_issuer].groupby("ticker").size().rename("rows")
    unresolved.sort_values(ascending=False).to_csv(args.output / "unresolved_tickers.csv")
    manifest = {"policy": "SEC_LAGGED_PROXY", "complete_sessions_lag": LAG,
                "max_fund_age": MAX_FUND_AGE, "identity_days": IDENTITY_DAYS,
                "strict_public_verified": False, "extracts": records,
                "code_sha256": _sha(Path(__file__)), "identity_review_sha256": _sha(REVIEW),
                "inputs_sha256": {q: _sha(ROOT / q) for q in ("data/prices.parquet", "src/graph/gics.csv")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(cov_assess.to_string(index=False))


if __name__ == "__main__":
    main()
