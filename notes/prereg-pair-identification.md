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

**Flag iff `c_ij ∈ [C_LO, C_HI]` and `lift_ij < L_MAX`.**
Orientation: composite = the rarer latent; parent = the commoner.
Child direction estimate: `u = normalize(D_comp − (D_comp·D_par) D_par)`.
Frequency estimate: `ρ̂ = P(comp fires) / P(comp ∨ par fires)`.

**Calibrated values: [TO BE FILLED FROM PILOT BEFORE THE CONFIRMATORY RUN — this note
is not final until C_LO / C_HI / L_MAX are concrete numbers and this bracket is replaced
by the pilot readout that justifies them.]**

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
