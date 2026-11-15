<!--
Title: start with the task ID, like the issue and CHECKLIST.md line: [3.1] Build sector edges from GICS
Keep prose short; let the Summary picture do the work. Delete these comments before submitting.
-->

<!-- One or two lines: what this does and why. -->

<!-- "Closes #N" only if the task is fully done (merging closes the issue); otherwise "Part of #N". -->
Closes #

## Summary

<!--
The smallest picture that explains the change: a diff sketch, file tree, pseudocode,
or a table of what goes in and what comes out.
-->

## Evidence

- **Before:** <!-- failing test, error or output -->
  **After:** <!-- passing test or output; include `pytest` counts and `ruff check .` -->

## Leakage

<!--
Which leakage invariants from CLAUDE.md this touches (survivorship, look-ahead news,
look-ahead features, walk-forward, sparse news), and the test that guards each one.
Or "None touched".
-->

## Merge Danger

**Door:** <!-- two-way (easy to revert) / one-way (SCHEMA.md, shared artifacts, anything teammates build on) -->

**Blast Radius:** <!-- one word, then who downstream reads this output -->

**Hand-off:** <!-- if the issue has the `handoff` label: who is now unblocked. Otherwise delete this line. -->
