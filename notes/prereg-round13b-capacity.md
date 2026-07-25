# Pre-registration — Round 13b: does absorption depend on dictionary capacity, and does the L1-vs-TopK gap open under scarcity?

**Status: LOCKED before any 13b SAE is trained.** Lock = the commit adding this
file. Frozen scorer `analysis/round13b_scorer.py` and frozen evaluator
`analysis/analyze_round13b.py` are fixed at the same commit. Report what they yield.

## Motivation

Round 12 (registered): no L1-vs-TopK real-scale absorption gap, clean paired diff
+0.0030, CI [−0.0010,+0.0067]. Round 13a eliminated the leading alternative
explanation — the null is **not** a feature-splitting artifact; absorption
survives a family-based endpoint (0.0542, CI [0.0494,0.0592]) and the arch
contrast stays null under it (−0.0012, CI [−0.0081,+0.0049]).

That leaves **H1**: round 12 ran at m=16384 with **dead% 46–57%**, i.e. a
spare-capacity regime. The toy theory places absorption pressure under capacity
*scarcity*. The same trap already produced one null in this program — the GPT-2
POC at m=1536 was in the composition regime, "so the remedy had nothing to fix",
and the capacity-limited rerun at m=128/256 then showed true harmful absorption.
H1 says round 12 simply never entered the regime where the theory predicts
anything. It is untested.

## Design

Capacity sweep on the **same cached activations as round 12** (`acts_train.pt`,
`acts_eval.pt`, `words_pythia-1.4b_L12.pt` from
`gs://sae-identifiability-artifacts-ebd5a273/round12/`), Pythia-1.4B L12, d=2048.

- **Widths:** `EXPANSION ∈ {1,2,4,8}` → **m ∈ {2048, 4096, 8192, 16384}**
- **Arches:** TopK (k=32) and L1 (λ calibrated per width)
- **Seeds:** 0–7 in every cell → **64 SAEs** (8 cells × 8 seeds)
- STEPS=15000, TF32, held-out eval, Anthropic-style dead-latent resampling —
  the unmodified round-12 recipe (`experiments/real_train_sae.py`)
- **All four widths are retrained here.** Round 12's m=16384 SAEs are NOT reused:
  the m=16384 cell is the reference endpoint of every contrast below, and this
  program has repeatedly been burned by results that "do not transfer across
  harnesses". One script, one box, one activation cache, all cells.

**λ calibration.** Per width, `experiments/calibrate_lambda.py` at the *training*
step budget (15k — round 12's bug #1 was calibrating at 8k and training at 15k),
picking the λ whose L0 is closest to 32. Grid **{2, 3, 4, 4.5, 5, 6}**, widened
downward from round 12's {4.5,5,5.5,6,7} because round 12's chosen λ=4.5 sat at
the *lower edge* of its grid; smaller m plausibly needs smaller λ. Calibration
reads only L0 and is blind to absorption. TopK needs no calibration (k=32 fixes
L0=32 by construction).

**Primary endpoint: the FAMILY endpoint** (13a-validated), `τ_fam=0.30`, cap 32,
θ=0. The single-latent endpoint is reported alongside, since 13a showed it
inflates absorption by ~25%.

`analysis/round13b_scorer.py` is `round13a_family_endpoint.py` with the
single-latent and family counters **copied verbatim** plus added `lost` counters
(needed for P5). Adding a counter cannot change the existing ones, so 13a
harness fidelity carries over by construction.

## Manipulation check (GATE — not a prediction)

The whole round is uninterpretable if the capacity manipulation does not bite.

- **MC:** mean `dead_pct` at m=2048 must be **at least 15 percentage points below**
  mean `dead_pct` at m=16384, and live latents `m·(1−dead_pct)` must be
  monotonically increasing in m.
- If MC fails, P1/P2 are reported as **UNINTERPRETABLE** and no capacity claim is
  made, whatever the endpoint does.

## Pre-registered predictions

**P1 (PRIMARY) — absorption rises as capacity falls.**
Paired-by-seed contrast `rate_family(m=2048) − rate_family(m=16384)`, computed
within each arch and pooled, 10k bootstrap over seeds.
- **CONFIRMED** if CI lower > 0.
- **FALSIFIED-DIRECTION** if CI upper < 0.
- **NOT CONFIRMED** if it straddles 0.
Also reported: Spearman(live latents, `rate_family`) across the four widths within
each arch, and the full width profile.

**P2 (PRIMARY) — the architecture gap opens under scarcity.**
Interaction: `[L1−TopK]@m=2048 − [L1−TopK]@m=16384`, paired by seed, 10k bootstrap.
- **CONFIRMED** if CI lower > 0 (L1 absorbs relatively more when capacity is scarce).
- **FALSIFIED-DIRECTION** if CI upper < 0. **NOT CONFIRMED** if it straddles 0.

*Registered power caveat, stated in advance:* this is a difference-of-differences
on 8 seeds and is the **least-powered test in the round**. Round 12's within-width
gap CI was ±0.004; an interaction CI will be materially wider. A null on P2 is
weak evidence against H1, not strong evidence — and will be reported that way.
P2 is the test that decides whether round 12's null was regime-bound, so it is
registered as primary despite the power limitation, and its CI width is the
headline number for sizing any successor round.

**P3 (secondary) — splitting shrinks under scarcity.**
Mean `|F_L|` by width and arch, paired contrasts. Expectation: families shrink as
m falls (no room to split). 13a measured 2.61 (L1) vs 1.25 (TopK) at m=16384.

**P4 (secondary) — does the single-latent metric's inflation depend on capacity?**
`(rate_single − rate_family)/rate_single` by width. If splitting inflation is
larger at large m, that independently reinforces 13a's metric finding.

**P5 (secondary, CONFOUND CONTROL) — absorption vs outright loss.**
`absorbed` requires `retained` (the letter must survive in the reconstruction).
At small m reconstruction is necessarily worse, so scarcity can convert
absorption into outright **loss**, mechanically *depressing* P1's endpoint.
Report `loss_rate` and `(absorbed+lost)/present` by width. **If absorption falls
while loss rises, that is the retention confound and P1 must not be read as
"capacity does not drive absorption".** This is registered in advance because it
is the most likely way P1 gives a misleading negative.

Note also that FVU necessarily degrades as m falls. That is part of the
treatment, not a removable confound, and is reported per cell.

## Gates

1. **Conformance:** every SAE has model=EleutherAI/pythia-1.4b, layer=12,
   θ=0, τ_fam=0.30; TopK cells k=32; each L1 cell has a single constant λ equal
   to that width's calibrated value; `m` equals `EXPANSION·2048` as registered.
2. **Seeds / anti-contamination:** exactly seeds {0..7} in each of the 8 cells,
   64 SAEs, no duplicates; every scored basename must match
   `sae_pythia-1.4b_L12_{l1,topk}_x{1,2,4,8}_s{0..7}.pt` exactly, and the
   results directory is cleaned before collection. *(Round 12 was contaminated by
   a stale out-of-config file carrying a duplicate seed; this gate is
   load-bearing and is why the filename pattern is pinned.)*
3. **Matched L0 within width:** per width, |mean L0(L1) − mean L0(TopK)| ≤ 3 with
   both in [24,40]. A width failing this is **excluded from P2** (the arch
   contrast) but retained for P1 (within-arch), and the exclusion is reported.
4. **Provenance:** SHA256 of every scored weight file recorded.
5. **Manipulation check** as above.

## Amendments (2026-07-25, both PRE-RESULTS — declared before any 13b SAE was trained)

Both are based on **timing and L0 measurements only**. No absorption quantity of
any kind had been computed when these were written.

### Amendment 1 — adaptive λ calibration replaces the fixed grid

Registered: fixed grid {2,3,4,4.5,5,6}. Replaced by
`experiments/calibrate_lambda_adaptive.py`: evaluate λ=4.5 (round 12's answer),
double or halve until L0=32 is bracketed, then bisect on log λ; ≤6 evaluations;
the monotonicity assumption (L0 decreasing in λ) is checked and reported.

Why: a smoke run showed that at **m=2048, λ=4 gives L0 far above 32**. The
registered grid may therefore not *bracket* L0=32 at small widths, in which case
calibration returns an edge value, that width fails the matched-L0 gate, and the
round loses its most important cell. The adaptive search brackets by
construction. It is also cheaper (≈5 vs 6 full 15k-step trainings per width).

Outcome-blind: calibration reads **only** the reported held-out L0 and never sees
any absorption quantity — as the original design already stated. The target
(L0 closest to 32, evaluated at the 15k training budget) is unchanged; only the
search over λ changes.

### Amendment 2 — drop m=8192; widths become {2048, 4096, 16384}

Measured on the L4: 2000 steps takes 211.7s at m=16384 and 38.9s at m=2048, i.e.
≈24 min per SAE at m=16384 for the registered 15k steps. The 4-width design costs
**≈17h (~$12)**, over the registered ~12h (~$8) budget.

Dropping m=8192 gives ≈12h (~$8), on budget. The sweep still spans **8×** in
width and retains **three** points for the monotonicity read. Both endpoints that
define P1 and P2 (m=2048 and m=16384) are **unchanged**, so no registered
primary is weakened — only the interior resolution of the width profile is.

Cell count becomes 3 widths × 2 arches × 8 seeds = **48 SAEs**. Seeds are kept at
8 rather than trimmed, because P2 is already the round's least-powered test.
Gate 2's expected count changes 64 → 48 and the filename pattern's width field
becomes `x{1,2,8}`.

### Anti-contamination note (implementation, not a design change)

λ calibration invokes the real trainer at SEED=0, which writes files whose names
are *legitimate* (`..._l1_x{E}_s0.pt`) but whose λ is a discarded search point.
Training later overwrites them, but to remove the hazard entirely the driver now
**deletes all `sae_*.pt` after calibration and before training**. This is exactly
round 12's failure class (a stale file with a valid-looking name and a duplicate
seed), so it is handled explicitly rather than relied upon.

## What this cannot do

- It cannot rescue round 12's P1. Round 12 stands as NOT CONFIRMED.
- A P2 confirmation would establish that the L1-vs-TopK absorption difference is
  **regime-dependent**, i.e. real under scarcity and absent at 8× overcompleteness
  — not that the round-12 measurement was wrong.
- Single model, single layer, single task (first-letter). No claim generalises
  beyond Pythia-1.4B L12 without further work.

## Ops

- One L4 (`dev-gpu`), pinned stack per `ENVIRONMENT.md` (py3.10, torch 2.5.1+cu121).
- Inputs pulled from GCS; results pushed back to `round13b/`. The orchestrator
  reads GCS with **user ADC** (`ops/gcs_adc.sh`), never the service account —
  round 12's collection bug.
- Budget ~12h L4 (~$8): ~3h λ calibration (L1 only, 4 widths × 6 grid points),
  ~7.5h training 64 SAEs, ~1.5h scoring. **The box is deleted/stopped on
  completion.**
