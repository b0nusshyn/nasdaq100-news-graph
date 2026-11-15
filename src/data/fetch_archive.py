"""Fetch selected FNSPID ZIP members with bounded HTTP Range requests, never the full ZIP."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

from src.core.paths import raw_dir

REVISION = "bf9189c41527198897d1af3e17b1a0095279fc45"
BASE_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve"


class RangeReader(io.RawIOBase):
    """Seekable, byte-budgeted HTTP reader for zipfile; refuses whole-file fallbacks."""

    def __init__(self, session, url: str, size: int, max_bytes: int):
        self.session, self.url, self.size, self.max_bytes = session, url, size, max_bytes
        self.position, self.downloaded = 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("Invalid seek mode")
        position = offset + (self.position if whence == io.SEEK_CUR else self.size if whence == io.SEEK_END else 0)
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        length = max(0, self.size - self.position)
        length = length if size is None or size < 0 else min(size, length)
        if not length:
            return b""
        if self.downloaded + length > self.max_bytes:
            raise ValueError("Archive download exceeds byte budget; full ZIP was not downloaded")
        first, last = self.position, self.position + length - 1
        response = self.session.get(
            self.url, headers={"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"},
            stream=True, timeout=45,
        )
        try:
            expected_range = f"bytes {first}-{last}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("Server did not honour exact HTTP Range; refusing full ZIP download")
            data = response.raw.read(length + 1)
            if len(data) != length:
                raise ValueError("Truncated or oversized HTTP Range response")
        finally:
            response.close()
        self.position += length
        self.downloaded += length
        return data


def download_members(tickers: list[str], *, revision: str = REVISION, max_bytes: int = 8_000_000) -> dict:
    """Raw archive symbols are not identity mappings; review before any pipeline import."""
    import requests

    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Use an immutable 40-character dataset revision")
    symbols = {t.strip().upper() for t in tickers}
    if not symbols or any(not re.fullmatch(r"[A-Z][A-Z0-9.\-]*", t) for t in symbols):
        raise ValueError("Invalid raw archive symbols")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    url = f"{BASE_URL}/{revision}/Stock_price/full_history.zip"
    folder = raw_dir() / "fnspid_recovery"
    folder.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers["User-Agent"] = "nasdaq100-news-graph/1.0 academic archive recovery"
        response = session.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        size = int(response.headers["Content-Length"])
        reader = RangeReader(session, url, size, max_bytes)
        found = []
        with zipfile.ZipFile(reader) as archive:
            for entry in archive.infolist():
                source = Path(entry.filename)
                if source.parent.as_posix() != "full_history" or source.suffix.lower() != ".csv":
                    continue
                symbol = source.stem.upper()
                if symbol not in symbols:
                    continue
                if entry.file_size > 3_000_000:
                    raise ValueError(f"Oversized uncompressed archive member: {symbol}")
                content = archive.read(entry)  # zipfile checks member CRC
                (folder / f"{symbol}.csv").write_bytes(content)
                found.append({"ticker": symbol, "archive_path": entry.filename, "bytes": len(content)})
    result = {"revision": revision, "url": url, "archive_size": size, "downloaded_bytes": reader.downloaded,
              "found": found, "not_found": sorted(symbols - {r['ticker'] for r in found})}
    (folder / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", required=True, help="exact raw archive symbols, before identity review")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-bytes", type=int, default=8_000_000)
    args = parser.parse_args()
    print(json.dumps(download_members(args.tickers, revision=args.revision, max_bytes=args.max_bytes), indent=2))


if __name__ == "__main__":
    main()
