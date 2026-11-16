"""Suite-wide pytest settings.

Tests that need a heavy dependency (torch, scikit-learn, statsmodels, xgboost, transformers) skip
when it is missing, so the light set (`requirements-dev.txt`) can still run the rest. A skipped
test is not a passed one, though: with `NDX_NO_SKIPS=1` (CI's full job) any skip fails the run
and is listed, so a missing dependency cannot quietly turn coverage into an empty pass.

The one exception is a test marked `real_data`: it checks the team's private artifacts (the
Drive, gitignored), which CI can never have. Its skip is allowed but still listed by name.
"""

from __future__ import annotations

import os

NO_SKIPS = os.environ.get("NDX_NO_SKIPS") == "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "real_data: needs the team's private data directory; may skip in CI")


def _reason(report) -> str:
    return report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)


def pytest_sessionfinish(session, exitstatus):
    if not NO_SKIPS:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    allowed = [r for r in skipped if "real_data" in getattr(r, "keywords", {})]
    failing = [r for r in skipped if r not in allowed]
    if allowed:
        reporter.write_line("")
        reporter.write_line(f"{len(allowed)} real-data test(s) skipped (private data not present, allowed):",
                            yellow=True)
        for r in allowed:
            reporter.write_line(f"  {r.nodeid}: {_reason(r)}", yellow=True)
    if not failing:
        return
    reporter.write_line("")
    reporter.write_line(f"NDX_NO_SKIPS=1 but {len(failing)} test(s) or module(s) skipped, so they did not run:",
                        red=True, bold=True)
    for r in failing:
        reporter.write_line(f"  {r.nodeid or r.fspath}: {_reason(r)}", red=True, bold=True)
    session.exitstatus = 1
