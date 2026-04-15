---
name: analyze
description: Generate progress.png from results.tsv — scatter of discarded/kept experiments with a running-min step line and annotated kept points. Use after (or during) an autoresearch run.
---

# Analyze

Generate `progress.png` summarizing autoresearch progress from `results.tsv`.

## Steps

1. Run:
   ```bash
   uv run analyze.py
   ```
   The script reads `results.tsv`, writes `progress.png`, and prints
   `total / keep / discard / crash` counts plus `baseline → best` delta.

2. If the script errors because `results.tsv` is missing or empty, tell the
   user to run at least one experiment first.

3. Relay the printed stats (and `progress.png` path) back to the user.
