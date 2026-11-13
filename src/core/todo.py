"""Marker for skeleton code that an issue will replace."""

from __future__ import annotations

from typing import NoReturn

REPO_URL = "https://github.com/b0nusshyn/nasdaq100-news-graph"


def todo(issue: int, what: str) -> NoReturn:
    raise NotImplementedError(f"{what}: not implemented yet, see {REPO_URL}/issues/{issue}")
