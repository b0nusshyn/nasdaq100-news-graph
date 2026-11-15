import io
import zipfile
from unittest.mock import Mock

import pytest

from src.data.fetch_archive import RangeReader


def session_for(content, *, status=206):
    def get(url, *, headers, **kwargs):
        first, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return Mock(status_code=status, headers={"Content-Range": f"bytes {first}-{last}/{len(content)}"},
                    raw=io.BytesIO(content[first:last + 1]))
    return Mock(get=Mock(side_effect=get))


def test_reads_only_selected_compressed_member():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("full_history/AAA.csv", "date,close\n2021-01-04,100\n")
        archive.writestr("full_history/BBB.csv", b"x" * 100000)
    content = buffer.getvalue()
    session = session_for(content)
    reader = RangeReader(session, "https://example.org/archive.zip", len(content), 10000)
    with zipfile.ZipFile(reader) as archive:
        assert archive.read("full_history/AAA.csv").startswith(b"date,close")
    assert reader.downloaded < 10000
    assert all("Range" in call.kwargs["headers"] for call in session.get.call_args_list)


def test_whole_file_response_refused_before_body_read():
    response = Mock(status_code=200, headers={}, raw=Mock())
    session = Mock(get=Mock(return_value=response))
    reader = RangeReader(session, "https://example.org/archive.zip", 590000000, 1000000)
    with pytest.raises(ValueError, match="refusing full ZIP"):
        reader.read(22)
    response.raw.read.assert_not_called()
    response.close.assert_called_once()


def test_budget_enforced_before_request():
    session = session_for(b"abcdef")
    reader = RangeReader(session, "https://example.org/archive.zip", 6, 3)
    assert reader.read(2) == b"ab"
    with pytest.raises(ValueError, match="byte budget"):
        reader.read(2)
    assert session.get.call_count == 1
