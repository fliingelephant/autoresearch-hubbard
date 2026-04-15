---
name: rehash
description: Recompute sha256 for every file tracked in frozen-manifest.toml, write the new hashes back, then print the new manifest sha so the user can rotate the FROZEN_MANIFEST_SHA repo secret. Use after a reviewed edit to a frozen file.
---

# Rehash

After a human-reviewed edit to a frozen file (`prepare.py`,
`src/autoresearch_hubbard/hamiltonian.py`, `tests/test_hamiltonian.py`, or the
physics constants inside `frozen-manifest.toml` itself), update the manifest's
file hashes and guide the CI secret rotation.

## Steps

1. **Read** `frozen-manifest.toml` to list the files under `[hashes]`.

2. **For each tracked file, recompute its sha256** with
   `shasum -a 256 <path>`. If it differs from the manifest entry, use the
   Edit tool to replace the old sha with the new one in `frozen-manifest.toml`.

3. **Recompute the manifest's own sha:** `shasum -a 256 frozen-manifest.toml`.

4. **Report to the user** — list which files had hash changes (one line each,
   `<path>: <old[:12]> -> <new[:12]>`), then:

   > New `FROZEN_MANIFEST_SHA` = `<sha>`
   >
   > Rotate the GitHub repo secret so CI's `physics-freeze` check passes:
   >
   > ```
   > gh secret set FROZEN_MANIFEST_SHA --body <sha>
   > ```
   >
   > (or update via web UI: Settings → Secrets and variables → Actions.)

5. **Do NOT commit.** The user reviews the diff and commits themselves.
