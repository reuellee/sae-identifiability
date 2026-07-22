# Pre-registration: label-free identification of absorbed (parent, composite) pairs

**Date:** 2026-07-22
**Status:** pre-registered, **not yet run** (confirmatory arm). Successor to
`notes/prereg-bimodality-estimator.md` (Arm A outcome: trained absorption is *gated* —
a parent-aligned latent plus an encoder-gated composite with near-disjoint firing — and,
given the pair, binarized signature counting recovers ρ to ≤ 0.02). This experiment tests
the one remaining gap: finding the pair **without labels**.

## Threshold provenance (disclosed)

Detector thresholds below are calibrated on an 8-run CPU **pilot**
(`experiments/pairid_pilot.py` → `results/prereg_pairid/pilot.csv`; absorbed regime only,
ρ ∈ {0.10, 0.20} × σ ∈ {0, 0.1} × 2 seeds). Pilot runs are calibration data and are
never counted as confirmatory evidence. Thresholds are locked in this note, and committed,
before the confirmatory GPU run starts.

## Hypotheses

- **G1 (detector, confirmatory).** In trained absorbed SAEs, the (parent, composite) pair
  is identifiable label-free from two statistics alone: decoder cosine in a mid band
  (geometric containment) and low binarized co-firing lift (gating). Formally: the detector
  below flags the true pair with high recall at low false-positive cost.
- **G2 (discrimination, confirmatory).** The detector does NOT flag genuinely-correlated
  independent feature pairs (same decoder geometry, independent firing) — i.e. the lift
  term, not the cosine term, carries the discrimination.
- **G3 (child recovery, confirmatory).** For flagged pairs, the residual of the composite
  decoder after projecting out the parent decoder recovers the child direction.
- **G4 (end-to-end, confirmatory).** Signature counting on the *detected* pair recovers ρ
  (closing the label-free frequency-recovery loop of §15b/Arm A, in the toy model).
- **G5 (equivalence class, exploratory).** Mutually-exclusive genuine features 45° apart
  are observationally equivalent to absorption at this level and WILL be flagged — a known,
  disclosed boundary of the method, not a failure.

## Detector (fully pinned)

On an eval set of 200k events from the run's own training distribution; latent j "fires"
iff act > θ = 0.05 (Arm A's binarization); keep latents with fire rate ∈ [5e-4, 0.6].
For every unordered kept pair (i, j):

- `c_ij = |cos(D_i, D_j)|` (unit-norm decoders)
- `lift_ij = P(i ∧ j) / (P(i) · P(j))`

**Flag iff `c_ij ∈ [C_LO, C_HI]` and (`lift_ij ≤ L_LO` or `lift_ij ≥ L_HI`)** —
i.e. co-firing far from independence in EITHER direction.
Orientation: composite = the rarer latent; parent = the commoner.
Child direction estimate: `u = normalize(D_comp − (D_comp·D_par) D_par)`.
Frequency estimate: `ρ̂ = P(comp fires) / P(comp ∨ par fires)`.

**Calibrated values (locked from the pilot, pre-confirmatory-run):
C_LO = 0.45, C_HI = 0.90, L_LO = 0.5, L_HI = 2.0.**

Pilot readout (`results/prereg_pairid/pilot.csv`, 6 absorbed runs of 8) and the
one design revision it forced:

- True-pair decoder cosine: 0.596–0.710 across σ ∈ {0, 0.1} → band [0.45, 0.90]
  gives margin on both sides while excluding duplicate/split latents (cos ≳ 0.95).
- True-pair lift is **bimodal in σ**: ≈ 0.000–0.005 at σ = 0 (clean mutual
  gating: parent fires 0% of joint events, composite 0.03% of host-only) but
  ≈ **2.96–3.07 at σ = 0.1** (noise leaks both gates: composite fires on ~55%
  of host-only, parent on ~60% of joint). The originally drafted one-sided rule
  (`lift < L_MAX`) would therefore miss every noisy absorbed pair. Revised,
  pre-lock, to the two-sided rule above, with the principled reading: **an
  absorbed pair's two latents are driven by the same host event stream —
  exclusively (lift ≪ 1) or jointly (lift ≫ 1), never independently — whereas
  genuinely correlated-but-independent features sit at lift ≈ 1.** G2's
  discrimination claim is unchanged: the lift term still carries it.
- Margins: ≥ 1.48× to L_HI (2.96 vs 2.0), ≥ 100× to L_LO (0.005 vs 0.5).
- Scoping consequence for D4 (locked now): the counting estimator
  `ρ̂ = P(comp)/P(comp ∨ par)` assumes gating, which σ = 0.1 leak inflates
  (pilot: P(comp) ≈ 0.18 at ρ = 0.2 · P_HOST = 0.05 true). **D4's confirmatory
  threshold applies to σ = 0 cells; σ = 0.1 ρ̂ is reported as exploratory**
  (consistent with Arm A's finding that magnitude/count estimators are
  σ-regime-dependent). D1–D3 remain confirmatory over ALL absorbed A cells,
  both σ values.
- Pilot absorption formation: 6/8 (the 2 failures are ρ = 0.10, σ = 0
  child-erasure seeds, matching Arm A's formation rates; disclosed).

## Design (Arm 1, synthetic; 16 seeds/cell; trainer = round-2/Arm A verbatim)

| cond | cells | events | prediction |
|---|---|---|---|
| **A** absorbed | ρ ∈ {0.05, 0.10, 0.20} × σ ∈ {0, 0.1} | Arm A generative model (child-solo = 0) | true pair flagged (G1) |
| **CD** correlated-independent | b-rate ∈ {0.03, 0.10}, σ=0 | second feature v_b = (v_p + e⊥)/√2 at 45° to v_p, firing INDEPENDENTLY | planted pair NOT flagged (G2) |
| **CDX** exclusive-correlated (exploratory) | 1 cell, σ=0 | v_b as CD but fires only when v_p does not | flagged (G5 — the known equivalence class) |
| **F** faithful | ρ=0.10, child-solo = 2.5·ε\*(λ, q_eff) ≈ 0.0152, σ=0 | faithful SAE: orthogonal decoders, co-firing | not flagged (cos ≈ 0) |
| **N** null | 1 cell, σ=0 | background only | false-positive floor |

ρ = 0.02 omitted from A (Arm A: only 4/16 form absorption there — a formation, not
detection, question). SAE weights are SAVED this time (`.pt` per program) so post-hoc
questions don't require retraining (Arm A lesson).

## Metrics & pass/fail (confirmatory)

| # | Metric | Pass |
|---|---|---|
| D1 (G1) | recall: fraction of absorbed-formed A-runs (oracle criterion cos_comp > 0.98, scoring only) whose true pair is flagged | ≥ 0.90 |
| D2 (G2) | (a) planted CD pair flag rate; (b) mean false-positive pairs per SAE across A/CD/F/N (any flagged pair ≠ a planted/true structure) | (a) ≤ 0.10 (b) ≤ 0.10 |
| D3 (G3) | median cos(u, v_c) over correctly flagged pairs | > 0.9 |
| D4 (G4) | mean |ρ̂ − ρ| over correctly flagged pairs | ≤ 0.03 |

**Falsifiers:** D1 < 0.70 → the two-statistic signature is insufficient (absorption
detection needs more structure than geometry + gating). D2(a) > 0.30 → the detector cannot
separate absorption from correlated features; the label-free route is confounded at its
first step. D3 ≤ 0.75 → "child recovery" from pair geometry is illusory.

## Decision rule → next step

- **D1–D4 pass:** promote to **Arm 2 (real data, gated):** GPT-2-small layer 6,
  capacity-limited SAEs (m = 128/256, the §15 true-absorption regime). Detector must flag
  the oracle-scored absorbed pair (scored exactly as §15); additionally run the full scan
  as audit v3 and report flagged pairs descriptively. Pass there = the complete label-free
  pipeline (detect pair → count signatures → ρ̂ → inverse-frequency weighting) becomes the
  paper's constructive answer to §15b.
- **D1 pass, D2 fail:** geometry+gating finds pairs but can't reject confounds → add the
  third statistic (residual-novelty or conditional-signature asymmetry) in a NEW
  pre-registration; do not tune this one post hoc.
- **D1 fail:** report as a genuine negative — gated absorption is detectable by
  construction (cond TV ≈ 1 in Arm A) but not by this cheap statistic; the pair-ID problem
  stays open.

## Analysis plan (fixed)

All runs scored by `analysis/analyze_prereg_pairid.py` (committed with this note).
Absorption filter and exclusion disclosure as in Arm A. Bootstrap CIs by seed resampling
(np seed 0, 10k draws). No threshold changes after the GPU run under any outcome.

---

## Arm 1 outcome (2026-07-22) + v1.1 amendment for Arm 2 (locked pre-Arm-2)

Confirmatory verdicts under the locked v1.0 detector
(`results/prereg_pairid/arm1_runs.csv`, scored by
`analysis/analyze_prereg_pairid.py`; 73/96 A-runs absorbed, disclosed):

- **D1 PASS**: recall 0.9315 (≥ 0.90). The ~7% misses are σ=0 runs whose
  partial gating leak lands the lift in the dead zone (0.5, 2.0).
- **D2(a) PASS**: planted CD pair flag rate 0.0625 (≤ 0.10) — the lift term
  carries the discrimination as predicted (G2).
- **D2(b) INCONCLUSIVE by 0.006**: fp/SAE 0.1062 vs ≤ 0.10, driven entirely
  by the null condition (0.81 fp/SAE). Weights-level diagnosis: every null FP
  is a **feature-splitting doublet** — a spare latent partially duplicating a
  background feature (cos 0.7–0.87) and co-firing with its main latent
  (lift 10–12.6, caught by the lift-HIGH branch).
- **D3 PASS**: median child-recovery cos 0.979, orientation 100% correct.
- **D4 PASS** (σ=0 confirmatory, per the locked scoping): mean |ρ̂−ρ| 0.0134.
  σ=0.1 exploratory ρ̂ is badly inflated (0.58–0.62 across true ρ), exactly as
  the locked scoping anticipated from Arm A's leak analysis.
- **G5 exploratory**: CDX flagged only 37.5% (predicted ~1) — trained encoder
  cross-talk (w_b·v_p ≈ 0.7) makes the b-latent fire on host events despite
  the b feature's exclusivity, pushing lift to ~2.3 near threshold. The
  idealized equivalence class is blurred by training, like §2's idealization
  in Arm A.

### Amendment: detector v1.1 (for Arm 2; calibrated on Arm 1 as training data)

Splits have a signature absorbed pairs lack: **containment** — the rare split
latent fires only within the common latent's events. Add
`overlap_ij = P(i∧j)/min(P(i),P(j))`; **veto pairs with overlap ≥ 0.9**
(splits ≈ 0.95–1.0; true absorbed pairs ≤ 0.81 in pilot + Arm 1).

Re-scored on Arm 1 (training data, disclosed —
`results/prereg_pairid/arm1_v11_rescore.csv`): D1 0.918 (−1 flag), D2(a)
0.0625, D2(b) **0.031** (null 0.81 → 0.25, A-condition FPs → 0). All
registered thresholds met. **v1.1 is hereby locked for Arm 2**, which is the
held-out confirmatory test (real GPT-2 activations, capacity-limited m =
128/256, thresholds transferred UNCHANGED). Arm 1's confirmatory record
remains the v1.0 scoring above; v1.1 claims stand or fall on Arm 2.

Gate decision: D1 ∧ D2(a) ∧ D3 ∧ D4 passed; D2(b) missed by 0.006 with a
diagnosed, vetoed mechanism. Arm 2 proceeds under v1.1 per this amendment.
