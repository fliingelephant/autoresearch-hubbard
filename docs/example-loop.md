# Example autoresearch loop

A worked example showing what a successful agent-driven loop looks like
on this repo. Numbers are illustrative — the actual ED ground state for
4×4 OBC half-filled U=8 is ≈ −8.5 (compute via `nk.exact.lanczos_ed` if
not cached).

## Branch setup

```
$ git checkout -b autoresearch/apr16
$ printf "commit\ttimestamp\tfinal_energy\tmodel\tspring_steps\telapsed_sec\tstatus\tdescription\n" > results.tsv
```

## Iteration 0 — baseline

Run `train.py` as-is. The first run establishes the reference number.

```
$ uv run train.py > run.log 2>&1
$ grep "^final_energy:\|^tsv_entry:" run.log
final_energy:    4.446313
tsv_entry:       2026-04-16T10:30:45	4.446313	SiT(d=128,L=4,K=4)	16	332.7	[STATUS]	[DESC]
```

Copy the `tsv_entry`, prepend `git rev-parse --short HEAD`, fill status
and description:

```
9edd47d	2026-04-16T10:30:45	4.446313	SiT(d=128,L=4,K=4)	16	332.7	keep	baseline
```

`final_energy = +4.4` is far from the ED ground state. SPRING got 16
timed steps; the optimization is still descending at the deadline.

## Iteration 1 — hyperparameter (smaller transformer)

The bottleneck is per-step SPRING cost. Halve `D_MODEL` and `N_LAYERS`
to see if a smaller model takes more steps and lands lower.

```diff
-D_MODEL = 128
-N_LAYERS = 4
+D_MODEL = 64
+N_LAYERS = 2
```

```
$ git commit -am "experiment: shrink SiT (d=64, L=2)"
$ uv run train.py > run.log 2>&1
$ grep "^final_energy:\|^spring_steps:" run.log
final_energy:    -2.118450
spring_steps:    62
```

```
4f8c3a1	2026-04-16T10:38:12	-2.118450	SiT(d=64,L=2,K=4)	62	312.4	keep	shrink SiT (d=64, L=2)
```

Improved (`+4.4 → −2.1`), more SPRING steps. Keep the commit.

## Iteration 2 — structural (drop pretraining)

Pretraining matches the SiT to the NNB warm-start, but the NNB itself
isn't very accurate at 4×4. Skip the pretrain phase entirely; reallocate
that time to SPRING. This also removes the need to keep the NNB output
structurally compatible with the SiT.

```diff
 NNB_FRACTION = 0.2
-PRETRAIN_FRACTION = 0.1
+PRETRAIN_FRACTION = 0.0
```

And in `main()`, skip the pretrain call (init SiT from a fresh PRNG).

```
$ git commit -am "experiment: drop pretraining; SPRING from random init"
$ grep "^final_energy:\|^spring_steps:" run.log
final_energy:    -1.892711
spring_steps:    71
```

```
2b1f7d4	2026-04-16T10:46:33	-1.892711	SiT(d=64,L=2,K=4)	71	305.8	discard	drop pretraining; SPRING from random init
```

Slightly worse (`−2.12 → −1.89`). Revert:

```
$ git reset --hard HEAD~1
```

Pretraining was earning its keep. Note this is structural feedback the
loop wouldn't get from hyperparameter tuning alone.

## Iteration 3 — structural (Slater + Jastrow, no neural net)

Per `program.md`'s "wavefunction class is fair game" guidance: at 4×4
with 16 sites, a parameter-light `Slater × Jastrow` may converge faster
than any transformer. Replace the SiT with `nk.models.Slater2nd` plus a
small density-density Jastrow correlator. Drop NNB and pretraining
since neither makes sense for this ansatz.

```diff
-from autoresearch_hubbard.ansatz import NNB, SiTBackflow
+import netket.models as nkm
...
-MODEL_ID = f"SiT(d={D_MODEL},L={N_LAYERS},K={N_DETERMINANTS})"
+MODEL_ID = "Slater+Jastrow"
...
 # main():
-    # ... NNB phase, pretraining, SiT init ...
-    spring_driver = nk.driver.VMC_SR(H, optax.sgd(SPRING_LR), variational_state=sit_state, ...)
+    model = nkm.Slater2nd(hilbert, restricted=False) * nkm.Jastrow()
+    state = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES, seed=SEED)
+    driver = nk.driver.VMC_SR(H, optax.sgd(SPRING_LR), variational_state=state, ...)
+    trace = run_vmc_phase(driver, TRIAL_SECONDS, "spring", LOG_EVERY_VMC)
```

```
$ git commit -am "experiment: pivot to Slater+Jastrow (no NN, no pretrain)"
$ grep "^final_energy:\|^spring_steps:" run.log
final_energy:    -7.412300
spring_steps:    487
```

```
8c9d2e1	2026-04-16T10:55:00	-7.412300	Slater+Jastrow	487	118.5	keep	pivot to Slater+Jastrow (no NN, no pretrain)
```

`−2.12 → −7.41`, closer to ED (≈ −8.5). 487 SPRING steps in the budget
because the model has ~300 params instead of ~150K. The agent's
structural pivot dominated three rounds of hyperparameter tuning. Per
the simplicity criterion, definitely keep — and the code shrunk by ~50
lines.

## Iteration 4 — refine the new ansatz

Now within the Slater+Jastrow family, tune the SR hyperparameters
(diag_shift, learning rate) for faster convergence.

```diff
-SPRING_LR = 1e-2
-DIAG_SHIFT = 1e-3
+SPRING_LR = 5e-2
+DIAG_SHIFT = 1e-4
```

```
$ git commit -am "experiment: SPRING_LR 1e-2->5e-2, diag_shift 1e-3->1e-4"
$ grep "^final_energy:" run.log
final_energy:    -8.241050
```

```
6d4f8e2	2026-04-16T11:01:30	-8.241050	Slater+Jastrow	502	115.2	keep	SPRING_LR 5e-2, diag_shift 1e-4
```

`−7.41 → −8.24`. Within ~3% of ED. Keep.

## Iteration 5 — crash and recover

Try increasing the LR more aggressively.

```diff
-SPRING_LR = 5e-2
+SPRING_LR = 5e-1
```

```
$ uv run train.py > run.log 2>&1
$ grep "^final_energy:" run.log
(empty)
$ tail -20 run.log
spring step 1 (warmup, 4.2s): energy=nan variance=nan
spring step 5: energy=nan variance=nan
...
ZeroDivisionError or NaN propagation
```

Crash:

```
9a1b3c5	2026-04-16T11:08:00	99.999999	Slater+Jastrow	0	5.0	crash	SPRING_LR=5e-1 (NaN'd out)
$ git reset --hard HEAD~1
```

## Pattern

- **Iter 0**: baseline (mandatory).
- **Iters 1, 4**: hyperparameter — incremental, low-risk.
- **Iter 2**: structural attempt that didn't pan out (discard but useful info).
- **Iter 3**: structural pivot that crushed three hyperparameter rounds.
- **Iter 5**: crash, log it, move on.

The loop expects all five flavors. Mix freely. When stuck on
hyperparameters, pivot the wavefunction class (per `program.md`).
