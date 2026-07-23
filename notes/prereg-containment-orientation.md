# Pre-registration: containment-based pair orientation

**Date:** 2026-07-23. **Status:** locked before generating any interpretation;
this note + `analysis/orient_recompute_e1.py` (S1) + `experiments/
round8b_orientation.py` (S2) are the pre-results commit. Successor to
`RESEARCH_PLAN.md` queue item, flagged in `results/round8/SUMMARY.md`
("orientation is now the pipeline's weakest validated stage ... a
containment-based orientation rule is the natural next pre-registration").

## Question

Given a flagged pair of latents (i, j) from the detector, which one is the
absorbed composite and which is the plain host? The current rule (**rarity**:
composite := whichever of i, j fires less often) scored 0.88 / 0.75 at
m=128/256 under the standard low-prevalence regime (E1, ρ≈eps≪1) but **0.00**
under E3's prevalence inversion (ρ=0.6) — by construction, since rarity
breaks the moment the composite is not the rarer latent. Arm A's gated/leaky
absorption finding explains why this is expected in general, not just at
ρ=0.6: a gated composite fires whenever the host fires (rate ≈ host rate),
so raw firing rate carries no reliable orientation signal once ρ is not
small.

**Hypothesis:** firing-set containment is invariant to ρ where rate is not.
A gated composite's firing set is (near-)nested inside the host's: whenever
the composite fires, the host fires too (composite ⇒ host), but the converse
only holds a ρ-fraction of the time (host ⇏ composite, except at ρ→1). This
asymmetry — which latent's firing "implies" the other's more strongly — does
not depend on which latent is rarer.

## Method (locked)

For a flagged/true pair (i, j), using the same binary fires already computed
by the detector (θ = 0.05):

```
C(i|j) = P(latent i fires | latent j fires)
C(j|i) = P(latent j fires | latent i fires)
composite := i if C(j|i) > C(i|j) + MARGIN
             j if C(i|j) > C(j|i) + MARGIN
             indeterminate otherwise   # MARGIN = 0.02, pre-declared
```

i.e. the composite is the latent whose firing more strongly implies the
other's (not an absolute containment threshold, unlike the natfeat rule —
this is a *relative* call between exactly two candidates, so no 0.80 cutoff
is needed). `rho_hat` and child-residual cosine are then recomputed on the
containment-chosen orientation exactly as they already are on the
rarity-chosen one (`experiments/round8_synthetic.py:142`,
`experiments/prereg_pairid_arm1.py:182`), so this is a drop-in replacement,
not a new pipeline.

Both rules are computed **head-to-head on the same detected pair and the
same fires array** in every run, so the comparison is paired (no
seed-to-seed confound).

## S1 — non-regression check on frozen E1 weights (free, CPU-only)

Reuses `results/round8/weights_r8e1_m{128,256}.pt` (48 absorbed-formed SAEs,
ε=0.002, real GPT-2 layer-6 activations) and the exact regeneration
procedure already used for the amendment-2 recompute
(`analysis/r8_specificity_recompute.py`: same seeds, same QR-random
directions, same injection formula, same θ). No new training, no GPU, no
cost — this is a pure re-analysis of already-collected, already-frozen data.

**Registered prediction (P3, non-regression):** containment-rule accuracy on
this low-ρ regime is ≥ rarity's already-reported 0.88 (m=128) / 0.75
(m=256) — i.e. containment should not be worse than rarity where rarity
already works. This is a sanity check, not the decisive test (ρ is small
here, so the two rules are expected to agree on most runs).

## S2 — confirmatory GPU stress test (new run, weights saved)

Replicates the `round8_synthetic.py` E3 harness (m=32, σ=0.1, λ=0.2,
15k steps) at **four prevalence cells**: ρ ∈ {0.10, 0.50, 0.60, 0.80}
(0.10 = sanity baseline where rarity already works; 0.60 = the registered
round-8 failure point; 0.80 = a stronger inversion, composite clearly
commoner than host). **24 seeds/cell** (standing constraint: confirmatory
cells ≥ 24 seeds, not round 8's 8). **Weights ARE saved this time**
(`results/round8/weights_r8b_orient.pt`) — round 8's Gemini review flagged
E2/E3 weight loss as a gap; this closes it.

**Registered predictions:**
- **P1 (primary):** at ρ=0.60, containment-rule accuracy ≥ 0.90 (vs
  rarity's registered 0.00). Falsified if < 0.70.
- **P2:** containment-rule accuracy at ρ=0.80 ≥ 0.80 — i.e. the fix isn't
  merely trading one blind spot (rare-composite assumption) for a
  symmetric one (common-composite assumption); it should hold across the
  range, not just recover the ρ=0.6 point.
- **P3':** containment-rule accuracy at ρ=0.10 is not worse than rarity's
  (both expected ≈1.0 — agreement expected here, not a discriminating
  cell).
- **Indeterminate rate:** report separately at each ρ, not folded into
  either the pass or fail count. Expected to rise as ρ→1 (both
  conditionals converge to 1, the rule's known degeneracy) — pre-declared,
  not a failure if it appears specifically near ρ=0.8–1.0 rather than at
  0.10–0.6.

## Registered readouts

- **R1:** paired accuracy table (rarity vs containment), all 5 cells (S1's
  E1 regime + S2's 4 ρ cells), with 10k seed-bootstrap CIs where seed count
  allows (S2 only, n=24/cell).
- **R2:** indeterminate-case rate by cell.
- **R3 (descriptive):** child-residual cosine and ρ̂ recomputed on the
  containment orientation, compared to the already-reported rarity numbers
  (0.990±0.001 oracle / 0.948–0.909 auto-rarity) — does containment close
  more of the gap to oracle than rarity did?

**Guardrails:** `reviews/EXTERNAL_REVIEW_GPT-5.6_2026-07-22.md` §5 language
constraints apply. If P1 fails, report the failure and the accuracy actually
observed — do not retune MARGIN post-hoc on the same data (a re-tuned
version would need its own fresh held-out cells). S2 is gated on a
GPU-cost decision (dev-gpu) separate from this lock.

## Amendment 1 (2026-07-23, before any GPU spend on S2)

S1's actual run (n=24/width, full E1 set) came back **1.000 accuracy when
determinate, but indeterminate 54% (m=128) / 79% (m=256) of the time** — far
higher than hypothesized. Two free CPU-only diagnostics on the same frozen
E1 weights (`/tmp/diag_orient.py`, `/tmp/diag_mag.py`, `/tmp/diag_crossmag.py`,
not part of the repo — exploratory pilot probes, disclosed here) explain why
and surface a substantially stronger candidate signal:

- The raw containment gap is real but small: `C(par|comp) − C(comp|par)`
  ≈ 0.02–0.03 across 10 sampled seeds — comparable to MARGIN itself, hence
  the high tie rate. Mechanism: trained absorption here is **leaky/gated**
  (consistent with Arm A) — comp fires almost as often as par overall
  (rates ≈0.35 both), not the idealized "comp fires only on host+child".
- **Ground-truth check** (16 seeds, m=128): comp's mean activation *given
  fired* is ≈6.0 on joint (host+child) events vs ≈0.6 on host-only events;
  par shows the mirror-opposite pattern (≈0.6 joint vs ≈3.9 host-only) — the
  documented anti-rotation mechanism (parent rotates away from the child
  direction under absorption, cutting its own activation when child is
  present; the composite's activation requires the child contribution to
  reach its high mode).
- **Label-free version of the same signal** (no ground-truth jj/pp needed —
  only the pair's own mutual firing): for each latent, split ITS OWN
  activation by whether the PARTNER co-fires:
  `delta_x = mean(act_x | x fires, partner silent) − mean(act_x | x fires, partner fires)`.
  Over 16 seeds, `delta_comp` (1.46–2.18) exceeded `delta_par` (0.49–0.95)
  **every single time**, gap min 0.76 / mean 1.13 — a far cleaner separation
  than the firing-containment gap, and it never tied in the pilot.

**Amended locked rule for S2 (composite := whichever latent shows the
larger cross-conditioned magnitude swing):**

```
delta_i = mean(act_i | i fires, j silent) - mean(act_i | i fires, j fires)
delta_j = mean(act_j | j fires, i silent) - mean(act_j | j fires, i fires)
composite := i if delta_i > delta_j + MARGIN_MAG
             j if delta_j > delta_i + MARGIN_MAG
             indeterminate otherwise   # MARGIN_MAG = 0.3, chosen as < half
                                       # the smallest pilot gap (0.76), locked
                                       # before S2 runs on fresh seeds/cells
Fallback (only if magnitude-delta is indeterminate): firing containment
(original rule, MARGIN = 0.02). Final fallback: rarity.
```

**Disclosed risk:** MARGIN_MAG was sized from an 8–16-seed *pilot* peek at
the same E1 weights S1 already used — not independent data. This amendment
is registered *before* S2 (fresh ρ cells, fresh seeds, m=32 not m=128/256)
runs, so S2 is a genuine held-out test of the rule, but the S1 re-run below
(full 48 seeds, same weights the rule was calibrated on) is **development-
set, not confirmatory** — report it as such, not as independent validation.

**Revised registered predictions:** P1–P3' (original, S2 §above) now apply
to the amended (magnitude-primary, containment-fallback) rule, not the
plain containment rule tested in S1. Add **P4 (development-set, S1
re-run):** amended-rule accuracy on the full 48-run E1 set is ≥ 0.95 with
indeterminate rate < 0.20 (vs plain containment's 0.54–0.79) — a sanity
check that the amendment actually fixes S1's coverage problem, reported
with the "development-set" caveat above, not as a pass/fail confirmatory
claim.
