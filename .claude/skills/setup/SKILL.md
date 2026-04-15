---
name: setup
description: Bootstrap the project environment. Installs uv if missing, detects CUDA, then runs uv sync with the cpu or cuda extra as appropriate.
---

# Setup

One-shot environment bootstrap for this repo.

## Steps

Run these as separate Bash calls so the user can see each action.

1. **Install uv if missing:**
   ```bash
   command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Detect CUDA and sync:**
   ```bash
   if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
     uv sync --extra cuda
   else
     uv sync --extra cpu
   fi
   ```

3. **Report** which extra was installed (cpu or cuda) in one line.
