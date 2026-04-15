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

4. **Report** — list which files had hash changes (one line each,
   `<path>: <old[:12]> -> <new[:12]>`), then the new manifest sha.

5. **Ask permission to rotate the CI secret via `gh`:**

   > Rotate `FROZEN_MANIFEST_SHA` repo secret now via `gh secret set`?

   If the user confirms, run this as a single Bash block. It pins the repo
   explicitly (gh auto-detection is ambiguous with multiple remotes) and
   verifies the timestamp moved (GitHub reads lag writes by 1–2 s):

   ```bash
   gh auth status >/dev/null 2>&1 || { echo "gh not authed — run 'gh auth login'"; exit 1; }
   REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
   BEFORE=$(gh secret list -R "$REPO" | grep '^FROZEN_MANIFEST_SHA' | cut -f2)
   gh secret set FROZEN_MANIFEST_SHA --body "<SHA>" -R "$REPO"
   sleep 2
   AFTER=$(gh secret list -R "$REPO" | grep '^FROZEN_MANIFEST_SHA' | cut -f2)
   [ "$AFTER" != "$BEFORE" ] && echo "rotated ($BEFORE -> $AFTER)" || echo "timestamp unchanged; retry or rotate via web UI"
   ```

   If the user declines or `gh` is unavailable, print the command for manual
   rotation (`gh secret set FROZEN_MANIFEST_SHA --body <sha>`) or direct them
   to Settings → Secrets and variables → Actions.

6. **Do NOT commit.** The user reviews the diff and commits themselves.
