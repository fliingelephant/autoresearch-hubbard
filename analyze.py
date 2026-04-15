"""Plot autoresearch progress from results.tsv → progress.png.

Reads the 8-column TSV written by the loop in program.md:
    commit  timestamp  final_energy  model  spring_steps  elapsed_sec  status  description

Usage:  uv run analyze.py [--input results.tsv] [--output progress.png]
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results.tsv")
    ap.add_argument("--output", default="progress.png")
    args = ap.parse_args()

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"{args.input} is empty — run at least one experiment first.")

    for r in rows:
        r["status"] = r["status"].strip().lower()
    valid = [r for r in rows if r["status"] != "crash"]
    for i, r in enumerate(valid):
        r["_i"] = i
        r["_e"] = float(r["final_energy"])
    if not valid:
        raise SystemExit("No non-crash experiments in results.tsv")

    baseline = valid[0]["_e"]
    best = min(r["_e"] for r in valid)
    n_keep = sum(1 for r in rows if r["status"] == "keep")
    n_disc = sum(1 for r in rows if r["status"] == "discard")
    n_crash = sum(1 for r in rows if r["status"] == "crash")

    print(f"total={len(rows)} keep={n_keep} discard={n_disc} crash={n_crash}")
    print(f"baseline={baseline:.6f} best={best:.6f} delta={baseline - best:.6f}")

    disc = [r for r in valid if r["status"] == "discard"]
    kept = [r for r in valid if r["status"] == "keep"]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter([r["_i"] for r in disc], [r["_e"] for r in disc],
               c="#cccccc", s=14, alpha=0.6, zorder=2, label="Discarded")
    ax.scatter([r["_i"] for r in kept], [r["_e"] for r in kept],
               c="#2ecc71", s=55, edgecolors="black", linewidths=0.5,
               zorder=4, label="Kept")

    running = []
    cur = float("inf")
    for r in kept:
        cur = min(cur, r["_e"])
        running.append(cur)
    ax.step([r["_i"] for r in kept], running, where="post",
            color="#27ae60", linewidth=2, alpha=0.7, zorder=3, label="Running best")

    for r in kept:
        label = f"{r['model']}: {r['description']}"
        if len(label) > 45:
            label = label[:42] + "..."
        ax.annotate(label, (r["_i"], r["_e"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8.0, color="#1a7a3a", alpha=0.9,
                    rotation=30, ha="left", va="bottom")

    margin = max(abs(baseline - best) * 0.15, 0.01)
    ax.set_ylim(best - margin, baseline + margin)
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel(r"final_energy = $\langle\psi|H|\psi\rangle/\langle\psi|\psi\rangle$ (lower is better)",
                  fontsize=12)
    ax.set_title(f"Autoresearch progress: {len(rows)} experiments, {n_keep} kept", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
