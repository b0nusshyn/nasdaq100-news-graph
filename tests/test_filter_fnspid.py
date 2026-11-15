import io

import pandas as pd

from src.nlp.filter_fnspid import KEEP, OUT_COLUMNS, filter_chunk, filter_csv, universe

CSV = """Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,Lsa_summary
2020-06-05 06:30:54 UTC,Apple up,AAPL,u1,P,,,
2020-06-05 06:30:54 UTC,Facebook up,FB,u2,P,,body,
2020-06-05 06:30:54 UTC,Agilent up,A,u3,P,,,
2018-12-31 22:00:00 UTC,Late Dec 31 article,AAPL,u4,P,,,
2018-12-01 15:00:00 UTC,Too early,AAPL,u5,P,,,
2024-01-01 03:00:00 UTC,Still Dec 31 in New York,AAPL,u6,P,,,
2024-01-01 06:00:00 UTC,Jan 1 in New York,AAPL,u7,P,,,
2021-03-01 12:00:00 UTC,Joined after window,ARM,u8,P,,,
,No date,AAPL,u9,P,,,
"""


def membership():
    return pd.DataFrame({
        "ticker": ["AAPL", "META", "ARM", "CELG"],
        "start": pd.to_datetime(["2018-01-01", "2018-01-01", "2024-06-24", "2018-01-01"]),
        "end": pd.to_datetime([None, None, None, "2019-11-20"]),
    })


def chunks(size=3):
    return pd.read_csv(io.StringIO(CSV), usecols=KEEP, dtype=str, chunksize=size)


def test_universe_uses_spells_overlapping_window_not_current_list():
    # CELG left in 2019 but is kept; ARM joined after 2023 and is dropped.
    assert universe(membership(), "2019-01-01", "2023-12-31") == {"AAPL", "META", "CELG"}


def test_filter_chunk_tickers_and_et_dates():
    df = pd.read_csv(io.StringIO(CSV), usecols=KEEP, dtype=str)
    out = filter_chunk(df, {"AAPL", "META"}, "2019-01-01", "2023-12-31")
    assert list(out.columns) == OUT_COLUMNS
    assert out["Url"].tolist() == ["u1", "u2", "u4", "u6"]
    # FB is mapped to its current symbol; the raw symbol is kept for audit.
    assert out.set_index("Url").loc["u2", ["ticker", "Stock_symbol"]].tolist() == ["META", "FB"]


def test_filter_csv_streams_chunks_and_counts(tmp_path):
    out = tmp_path / "raw" / "fnspid_filtered.csv"
    stats = filter_csv(chunks(), {"AAPL", "META", "CELG"}, "2019-01-01", "2023-12-31", out)
    assert stats["rows_in"] == 9
    assert stats["rows_out"] == 4
    assert stats["rows_per_ticker"] == {"AAPL": 3, "META": 1}
    assert stats["tickers_without_news"] == ["CELG"]
    written = pd.read_csv(out, dtype=str)
    assert written["Url"].tolist() == ["u1", "u2", "u4", "u6"]
    assert not (tmp_path / "raw" / "fnspid_filtered.csv.part").exists()
