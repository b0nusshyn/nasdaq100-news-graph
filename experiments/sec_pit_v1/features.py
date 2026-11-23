"""Offline feature join for independently audited EDGAR records."""
import bisect
import math
import re
from collections import defaultdict

import pandas as pd
import pandas_market_calendars as mcal

PERIODIC = {"10-K", "10-Q", "10-K/A", "10-Q/A"}


def _timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Expected a timezone-aware timestamp")
    return stamp.tz_convert("America/New_York")


def available_session(accepted_at, public_by, sessions):
    """Delay until the first session after both acceptance and proven availability."""
    boundary = max(_timestamp(accepted_at), _timestamp(public_by))
    index = sessions.searchsorted(pd.Timestamp(boundary.date()), side="right")
    return sessions[index] if index < len(sessions) else pd.NaT


def quarterly_margin(facts, filing):
    """Use matching consolidated US-GAAP USD quarter facts from a single accession."""
    contexts = {}
    for tag in ("RevenueFromContractWithCustomerExcludingAssessedTax", "OperatingIncomeLoss"):
        values = defaultdict(set)
        for fact in facts.get(tag, ()):
            if (fact["accession"] != filing["accession"] or fact["cik"] != filing["cik"]
                    or fact["unit"] != "USD" or fact["end"] != filing["period_end"]
                    or not fact.get("consolidated") or not fact.get("start")):
                continue
            duration = (pd.Timestamp(fact["end"]) - pd.Timestamp(fact["start"])).days + 1
            if 75 <= duration <= 105 and math.isfinite(fact["value"]):
                values[fact["start"]].add(fact["value"])
        contexts[tag] = values
    revenue, income = contexts.values()
    shared = revenue.keys() & income.keys()
    if len(shared) != 1:
        return None
    start = next(iter(shared))
    if len(revenue[start]) != 1 or len(income[start]) != 1:
        return None
    sales = next(iter(revenue[start]))
    return next(iter(income[start])) / sales if sales > 0 else None


def build_features(keys, filings, facts):
    """Preserve keys, using a historically audited per-row CIK (nullable).

    Filings must carry evidence of a historical public-by timestamp. Acceptance
    alone is insufficient. Sparse inputs cannot establish zero-event coverage.
    """
    if keys.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate assessment keys")
    if keys.empty:
        raise ValueError("No assessment keys")
    if keys.date.min() < pd.Timestamp("2019-01-01") or keys.date.max() > pd.Timestamp("2023-12-31"):
        raise ValueError("Only 2019–2023 feature rows are permitted")
    filings = list(filings)
    verified = [filing for filing in filings if filing.get("public_by")
                and filing.get("public_evidence") and filing.get("acceptance_source")]
    earliest = min((pd.Timestamp(_timestamp(row["accepted_at"]).date())
                    for row in verified), default=keys.date.min())
    sessions = mcal.get_calendar("NASDAQ").schedule(
        min(earliest, keys.date.min()), keys.date.max()).index
    if not keys.date.isin(sessions).all():
        raise ValueError("Every key must be a market session")
    events = defaultdict(list)
    seen = set()
    for filing in verified:
        form = filing["form"]
        if form not in PERIODIC and not (
            form in {"8-K", "8-K/A"} and "2.02" in filing.get("items", ())
        ):
            continue
        accession = filing["accession"]
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) or accession in seen:
            raise ValueError("Invalid or duplicate accession")
        seen.add(accession)
        if pd.Timestamp(filing["period_end"]).date() > _timestamp(filing["accepted_at"]).date():
            raise ValueError("Filing period ends after acceptance")
        session = available_session(filing["accepted_at"], filing["public_by"], sessions)
        if not pd.isna(session):
            boundary = max(_timestamp(filing["accepted_at"]), _timestamp(filing["public_by"]))
            events[filing["cik"]].append((session, boundary, accession, filing))

    histories = {}
    for cik, rows in events.items():
        last_event = last_periodic = last_margin = None
        history = []
        for session, _, accession, filing in sorted(rows):
            last_event = (session, accession, filing["form"])
            if filing["form"] in PERIODIC:
                period = pd.Timestamp(filing["period_end"])
                last_periodic = max(period, last_periodic) if last_periodic is not None else period
                margin = quarterly_margin(facts, filing)
                if margin is not None and (last_margin is None or period >= last_margin[0]):
                    last_margin = (period, margin, accession)
            history.append((session, last_event, last_periodic, last_margin))
        histories[cik] = ([row[0] for row in history], history)

    positions = {day: pos for pos, day in enumerate(sessions)}
    values = []
    for row in keys.itertuples(index=False):
        dates, history = histories.get(row.cik, ([], []))
        pos = bisect.bisect_right(dates, row.date) - 1
        missing = (None, None, None, None, None, None)
        if pos >= 0:
            day, accession, form = history[pos][1]
            _, _, period, margin = history[pos]
            missing = (positions[row.date] - positions[day], accession, form, period,
                       margin[1] if margin else None, margin[2] if margin else None)
        values.append(missing)
    columns = ["filing_session_age", "filing_accession", "filing_form", "last_periodic_end",
               "quarterly_operating_margin", "margin_accession"]
    result = keys.copy()
    for column, data in pd.DataFrame(values, columns=columns).items():
        result[column] = data.array
    return result
