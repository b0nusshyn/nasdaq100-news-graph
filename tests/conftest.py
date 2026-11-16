"""Suite-wide pytest settings.

Tests that need a heavy dependency (torch, scikit-learn, statsmodels, xgboost, transformers) skip
when it is missing, so the light set (`requirements-dev.txt`) can still run the rest. A skipped
test is not a passed one, though: with `NDX_NO_SKIPS=1` (CI's full job) any skip fails the run
and is listed, so a missing dependency cannot quietly turn coverage into an empty pass.
"""

from __future__ import annotations

import os

NO_SKIPS = os.environ.get("NDX_NO_SKIPS") == "1"


def pytest_sessionfinish(session, exitstatus):
    if not NO_SKIPS:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if not skipped:
        return
    lines = [f"NDX_NO_SKIPS=1 but {len(skipped)} test(s) or module(s) skipped, so they did not run:"]
    for r in skipped:
        reason = r.longrepr[2] if isinstance(r.longrepr, tuple) else str(r.longrepr)
        lines.append(f"  {r.nodeid or r.fspath}: {reason}")
    if reporter:
        reporter.write_line("")
        for line in lines:
            reporter.write_line(line, red=True, bold=True)
    session.exitstatus = 1
