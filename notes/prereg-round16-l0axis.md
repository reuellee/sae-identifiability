# Pre-registration — Round 16: the L0 axis on Pythia — does absorption fall as the active budget rises, at fixed width?

**Status: LOCKED before any round-16 SAE is trained.** Lock = the commit adding
this file. Frozen scorer `analysis/round16_scorer.py` and frozen evaluator
`analysis/analyze_round16.py` are fixed at the same commit. Report what they
yield.

## Motivation

Round 15's descriptive series D1 (Gemma Scope 2, fixed 65k width, three released
L0 settings) found **absorption falls monotonically as L0 rises** — rate_family
0.194 (L0≈25) → 0.157 (≈72) → 0.094 (≈165) — while sel-family size *rises*
(1.56 → 1.91 → 2.50). Combined with P1 (absorption rises with width), the two
capacity axes point in **opposite directions**: more total dictionary → more
absorption; more active budget per token → less.

This axis has never been varied on Pythia: rounds 12/13a/13b **matched L0=32 by
design** (it was a control there). D1 supplies a registered directional
prediction made before any Pythia L0-sweep number existed. Round 16 tests it
with the program's own suite, both architectures, and multi-seed CIs — none of
which the single-seed Gemma series could provide.

Secondary motivation (plan §"mechanism dissociation"): round 15's P2 showed the
width→absorption effect has no family-growth mediator on Gemma. The L0 sweep is
the cheapest place to start dissociating "family latents fire more because the
budget is bigger" (a threshold/fire-rate account) from "the letter feature earns
a dedicated slot" (a representation account); the D-control below is registered
for exactly that.

## Design

L0 sweep at **fixed width m=16384** (EXPANSION=8, the program's reference
width), on the **same cached activations as rounds 12/13b** (`acts_train.pt`,
`acts_eval.pt`, `words_pythia-1.4b_L12.pt` from
`gs://sae-identifiability-artifacts-ebd5a273/round12/`), Pythia-1.4B L12,
d=2048.

- **L0 targets:** `{16, 64}` — a 4× span centered (geometrically) on the
  program's standard L0=32.
- **Arches:** TopK (k = the target, exact by construction) and L1 (λ calibrated
  per target).
- **Seeds:** 0–7 in every cell → **32 fresh SAEs** (4 cells × 8 seeds).
- STEPS=15000, TF32, held-out eval, Anthropic-style dead-latent resampling —
  the unmodified round-12/13b recipe (`experiments/real_train_sae.py`).

**Interior reference cell (descriptive, no training):** round 13b's frozen
m=16384 cell (16 SAEs, both arches, L0≈32, same recipe, same caches, seeds 0–7,
weights in `gs://…/round13b/`) is **re-scored** with the round-16 scorer so the
interior point carries the new counters. It enters ONLY the three-point
monotonicity read and the D-control profile — **no registered contrast uses
it**. This is not a violation of 13b's "retrain everything" rule: both
endpoints of every registered contrast in this round are freshly trained in the
same run; the interior cell is annotation. Its identity is **pinned by SHA256**
in the frozen evaluator (the 16 hashes recorded in
`results/real/round13b_results.json` at lock time); a **non-empty interior
set that mismatches the 16 pinned hashes fails conformance** (that is
contamination). An **absent** interior set is not contamination: it is
printed as a named deficiency in the gate line, every interior-dependent
read reports ABSENT, and the round proceeds — the cell is descriptive
annotation, not an endpoint (wording settled at Gemini pre-lock finding 4).

**λ calibration.** Per L0 target, `experiments/calibrate_lambda_adaptive.py` at
the training step budget (15k), LAM0=4.5 (the known λ→L0=32 point at this
width), MAX_EVALS=6, TARGET ∈ {16, 64}, band = TARGET × [0.875, 1.125]
(BAND_LO/BAND_HI = 14/18 and 56/72). Calibration reads only held-out L0 and is
blind to every absorption quantity. All `sae_*.pt` files are **deleted after
calibration and before training** (round-12's stale-file failure class, handled
as in 13b).

**Naming (anti-contamination).** The trainer's output name does not encode k/λ,
so at fixed width every cell would collide on
`sae_pythia-1.4b_L12_{arch}_x8_s{seed}.pt`. The driver **renames each file
immediately after its training run** to

```
sae_pythia-1.4b_L12_{l1|topk}_x8_k{16|64}_s{0..7}.pt
```

where `k` is the **cell's L0 target** (for L1 too — it names the cell, not a
TopK parameter). Gate 2 pins this pattern; anything off-pattern in the results
directory is deleted before scoring. Interior-cell files are pulled into a
**separate directory** and scored in a separate pass; they never enter the
fresh results directory.

**Cell identity in scored rows.** For TopK rows the weight blob's `k` must
equal the filename's `k` (gate 1). For L1 rows the blob's `k` field is the
trainer's unused default and is ignored; the cell is the filename tag, and the
**realized held-out L0 must land in the cell's band** (manipulation check) —
fail-closed, as the config gate standard requires.

## Manipulation check (GATE — not a prediction)

The round is uninterpretable if the L0 manipulation does not bite.

- **MC:** per arch, BOTH of:
  (a) mean realized held-out L0 in the k64 cell ≥ **2.5×** the k16 cell's, and
  (b) each cell's mean realized held-out L0 within **[0.75, 1.25] × its
  target** (12–20 at k16, 48–80 at k64). (TopK satisfies both by
  construction; the check is really the L1 calibration. The band makes the
  fail-closed promise in §Design real — a ratio alone would pass an L1 pair
  calibrated to L0 = 2 and 5. Gemini pre-lock finding 2.)
- If MC fails in an arch, that arch's P1 is **UNINTERPRETABLE**; if it fails in
  both, the round is.

## Pre-registered predictions

Single primary; everything else is secondary or control. All CIs are 10k
bootstrap **over seeds** (the seed is the cluster — registered here explicitly
so the round cannot repeat 13b's pooled-CI correction of record).

**P1 (PRIMARY) — absorption falls as L0 rises at fixed width.**
Per arch and seed s: `d_s = rate_family(k16, s) − rate_family(k64, s)`.
Seed-level pooled value `u_s` = mean of `d_s` over the **MC-passing arches**
for seed s (an arch failing MC contributes no diffs anywhere). Bootstrap the
8 `u_s`.
- **CONFIRMED** if CI lower > 0 (matches D1's direction) **and the D-gate
  below holds**.
- **CONFIRMED-BUT-MECHANICAL** if CI lower > 0 and the D-gate fails: the
  statistical effect is real but indiscriminate family firing rose by
  enough that the round must NOT be read as confirming capacity dynamics.
- **FALSIFIED-DIRECTION** if CI upper < 0.
- **NOT CONFIRMED** if it straddles 0.

*D-gate (registered constant, pre-lock — Gemini pre-lock finding 1):* the
endpoint's fall must not be dominated by families simply firing more
everywhere. Pooled over MC-passing arches:
`rise_ffa = fam_fire_absent(k64) − fam_fire_absent(k16)` (cell means) and
`Δ_abs = rate_family(k16) − rate_family(k64)`. The D-gate **holds** iff
`rise_ffa ≤ 0.5 × Δ_abs`. The 0.5 factor is a registered convention: if
letter-indifferent family firing rises by more than half the endpoint's
fall, the mechanical account is too large to ignore in the headline.
Also reported: per-arch CIs (bootstrap over that arch's 8 `d_s`), and the
three-point profile k16 → interior(≈32) → k64 per arch with a monotonicity
note (descriptive; the interior cell is not part of the registered contrast).

**P2 (secondary) — split families grow with L0 (L1).**
D1's co-movement: fam_size rose 1.56 → 2.50 as L0 rose. **Letter-paired**
(Gemini pre-lock finding 3 — higher L0 lets more letters cross τ, so
unpaired cell means confound composition with growth): per seed, over the
letters that are clean in **both** the k16 and k64 L1 cells for that seed,
the mean per-letter `|F_L|(k64) − |F_L|(k16)`; then bootstrap the 8
seed-level means. **L1 only** (TopK's family size sits near its 1.25 floor
at this width and is reported descriptively). The number of paired letters
per seed is reported alongside.
- **CONFIRMED** if CI lower > 0; **FALSIFIED-DIRECTION** if CI upper < 0;
  else **NOT CONFIRMED**.

**P3 (secondary) — arch × L0 interaction.**
`[L1−TopK]@k16 − [L1−TopK]@k64` per seed, bootstrap over seeds.
- **CONFIRMED** if CI lower > 0 (L1's absorption is more L0-sensitive than
  TopK's); **FALSIFIED-DIRECTION** if CI upper < 0; **NOT CONFIRMED** if it
  straddles 0 (vocabulary registered here — GPT pre-lock P2.4).
Registered power caveat, stated in advance: this is the round's
least-powered test (13b's interaction CI half-width was ≈0.006 on 8 seeds);
a null is weak evidence. Excluded entirely if either target fails gate 3 or
either arch fails MC.

**P4 (CONFOUND CONTROL) — absorption vs outright loss.**
`absorbed` requires `retained`. At k16 reconstruction is necessarily worse, so
retention falls and the k16 endpoint is **depressed** — i.e. the confound is
**conservative for P1's predicted direction**. Registered readings:
- If P1 CONFIRMED and `rate_lost(k16) > rate_lost(k64)`: the confirmation
  survived a headwind; say so, and report `(absorbed+lost)/present` by cell.
- If P1 FALSIFIED-DIRECTION and loss rises at k16: the retention confound is
  the first-line explanation and P1 must NOT be read as "the L0 direction
  reverses on Pythia" without the combined endpoint.
FVU by cell is part of the treatment (as in 13b), reported per cell.

**D-control (registered descriptive — mechanism dissociation).**
From the two added scorer counters (raw-mask, probe-independent):
`fam_fire_present` = P(any family latent fires | letter-present word) and
`fam_fire_absent` = P(any family latent fires | letter-absent word), by cell.
- If the k64 cells' absorption drop is accompanied by a **comparable rise in
  `fam_fire_absent`** (families simply fire more everywhere), the mechanical
  fire-rate account is favored — the endpoint's fall with L0 would then be
  partially built into its construction, and the SUMMARY must say so.
- If `fam_fire_absent` stays low/flat while `fam_fire_present` rises, the
  budget is buying **letter-specific** firing — the slot account.
The quantitative form of this control is the **D-gate inside P1's verdict**
(above); beyond that gate the profiles constrain the interpretation
section.

**D-sensitivity (registered — GPT pre-lock P1.3): matched-budget re-score.**
The D-gate alone is weak armor: families are *selected* for low letter-absent
firing (`sel ≥ τ`), so `fam_fire_absent` can stay low by construction while
the endpoint still falls mechanically with the budget. So every SAE is
additionally re-scored under a **common firing mask**: its top-16 activations
per word (`MB_K=16`, still requiring `> θ`; words with fewer than 16 positive
activations keep all of them). Family selection (`sel ≥ τ`, cap 32) and the
family-miss endpoint are recomputed under that mask; reconstruction and
retention stay native, because the mask's purpose is to equalize firing
**opportunities** — exactly the mechanical concern. For a native-TopK k=16
cell the mask coincides with native firing, anchoring the comparison.

The registered read: the same seed-clustered pooled contrast
(`rate_family_mb(k16) − rate_family_mb(k64)`, `u_s` over MC-passing arches,
10k bootstrap).
- **SURVIVES** (CI lower > 0): representational-change language is licensed —
  the k16-trained and k64-trained dictionaries differ even at a common budget.
- **DOES NOT SURVIVE**: a native P1 confirmation establishes an **L0
  association only**; the SUMMARY must not use representational-change
  language.
This rule binds the write-up, not P1's verdict vocabulary.

## Gates

1. **Conformance (fail-closed):** every fresh row has
   model=EleutherAI/pythia-1.4b, layer=12, θ=0, τ_fam=0.30, m=16384,
   expansion=8; blob `arch` and `seed` equal the filename's; TopK rows' blob
   `k` equals the filename `k`; the row's `eval_src` is the held-out cache
   (the trainer records it — a non-held-out L0/FVU fails the row); the
   scorer-recorded `words_model`/`words_layer` equal the registered
   model/layer (the round-15 wrong-layer failure class, GPT pre-lock P1.2);
   `mb_k` = 16; each L1 cell has a single constant λ equal to that cell's
   calibrated value (checked against the shipped calibration output file
   when present; its absence is a named deficiency), and λ(k16 cell) >
   λ(k64 cell) (shrinkage monotonicity sanity). Interior rows, when the set
   is non-empty: exactly the 16 pinned (name, hash) pairs — hashes are the
   **16-hex sha256 prefixes** this program has recorded since 13a (GPT
   pre-lock P2.2: they identify against accidents, not adversaries; the pins
   must stay commensurable with 13b's recorded values).
2. **Seeds / anti-contamination:** exactly 32 fresh rows; every basename
   matches `^sae_pythia-1\.4b_L12_(l1|topk)_x8_k(16|64)_s[0-7]\.pt$`; each of
   the 4 cells has seeds {0..7} exactly; results dir wiped before the round
   and calibration artifacts deleted before training.
3. **Arch-matched L0 within cell (gates P3 only):** per L0 target, |mean
   L0(L1) − mean L0(TopK)| ≤ 0.15×target, and both means within
   [0.75, 1.25]×target. A target failing this excludes P3 (the arch
   interaction needs arch-comparable doses); P1 is unaffected — each arch's
   own dose is guarded by MC, and a mis-calibrated L1 cell only *attenuates*
   its span (conservative for P1). The exclusion is reported.
4. **Provenance:** sha256 prefix of every scored weight file recorded, all
   distinct.
5. **Manipulation check** as above.

**Gates 1, 2 and 4 SUPPRESS every verdict** (GPT pre-lock P1.1): if any of
them fails, P1/P2/P3 are reported UNINTERPRETABLE — a conformance violation
cannot coexist with a CONFIRMED headline. (The evaluator still exits 0 so the
driver ships the artifacts; the verdict text is the deliverable.)

**Driver integrity (registered — GPT pre-lock P1.4/P1.2):** the run is bound
to a manifest recording the lock commit; the registered output wipe happens
exactly once per lock, resume is refused across different locks, and the
name gate **aborts** unless exactly 32 files survive. The three input caches
are verified against md5s pinned in the driver at lock time
(`acts_train.pt` D48OnIJPLIGqKgsCMur+JQ==, `acts_eval.pt`
hNwKlaK4hG1H7SMchaglDw==, `words_pythia-1.4b_L12.pt`
VX/fNafoHdj8D37yWgfSrw==; a mismatch aborts).

Evaluator self-test: `analysis/analyze_round16.py --selftest` runs the verdict
logic on synthetic rows covering CONFIRMED / CONFIRMED-BUT-MECHANICAL /
FALSIFIED-DIRECTION / NOT CONFIRMED / D-sensitivity SURVIVES and DOES NOT
SURVIVE / global-gate suppression / per-gate failures / MC (ratio, band,
both-arch) / P3 sign branches — 22 checks, exits nonzero on any mismatch.
It was run before lock (standard since round 15).

## What this cannot do

- Single model (Pythia-1.4B), single layer (L12), single task (first-letter),
  single word sample. No claim generalizes beyond that without further work.
- It does not upgrade round 15's transfer scope (that needs the queued seed
  replication / additional suites), and it does not touch the width axis —
  13b stands.
- The endpoint-construct caveat (13b defense Q2) applies here with extra
  force, because firing probability scales with L0 by construction. The
  D-control constrains but does not settle it; a fragmentation-corrected
  endpoint remains queued (plan item 2).
- k16's worse reconstruction makes P4's confound asymmetric: the design can
  under-read absorption at low L0, not over-read it.

## Ops / budget

- Fresh ephemeral **g2-standard-8 (1×L4)**, created with storage-rw scopes at
  create time (NOT a snapshot rebuild — 13b's scope lesson), spot-first with
  on-demand fallback; `ops/vm_watchdog.sh` self-poweroff armed; the box is
  **deleted** after collection.
- Driver `ops/l4_r16.sh`, staged and resumable: a (cell, seed) whose renamed
  output exists is skipped, so a spot preemption costs at most the in-flight
  SAE.
- Registered budget: λ calibration ≤12 full trainings ≈ 5.3 h; 32 trainings
  ≈ 14 h (26.4 min/SAE at m=16384, measured in 13b); scoring 48 SAEs ≈ 2 h.
  **≈ 21 h L4 ceiling ≈ $7 (spot) / ≤ $21 (on-demand cap).** If L4 capacity
  is unavailable in every tried zone, the round pauses — no design change.
- Inputs pulled from GCS `round12/` (caches) and `round13b/` (interior
  weights); artifacts pushed to `round16/`. Collection falls back to
  `ops/collect_r13b.sh`-style out-of-band pull if uploads fail.
