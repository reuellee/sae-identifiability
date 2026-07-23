# Round 9: gating-corrected ρ estimators — registered scoring

Prereg `notes/prereg-gating-corrected-rho.md`, **lock `b0276cc`** (bars from
D1's error budget; nothing changed after lock). Run: 384 fresh-seed runs
(SC 288 + RC 96), one spot-L4 session (us-central1-a, 36 min wall,
≈ $0.20), weights saved (`weights_r9_*.pt`), scored by the frozen
`analysis/analyze_round9.py`. Formation: RC 96/96 formed (0 exclusions,
including the a-priori-risky Q = 0.04 cell); SC 335/384 total included
(per-cell exclusions 0–6, all cells ≥ 18/24 — every cell scoreable).

## Verdicts (as registered)

| Prediction | Verdict | Detail |
|---|---|---|
| **P1M** (RC mechanism, MAE ≤ 0.03 / falsify > 0.07) | **PASS 4/4** | worst cell 0.0026 |
| **P2M** (SC mechanism) | **PASS 12/12** | worst cell 0.0021 |
| **P1O** (RC operational, ≤ 0.05 / > 0.15) | **INCONCLUSIVE** | ρ = 0.1 cell 0.0695 (inconclusive zone); other 3 pass (0.0116–0.0431); none falsified |
| **P2O** (SC operational) | **INCONCLUSIVE** | ρ = 0.1 σ = 0.1 cell 0.0660 (inconclusive); other 11 pass; none falsified |
| **P3** (beats ρ̂_C by 0.05 in 14 predefined cells) | **FALSIFIED** | 12/14 pass; SC σ = 0 ρ ∈ {0.1, 0.3} fail — see diagnosis |
| **P4** (median δ_J, δ_S ≤ 0.05 / > 0.10) | **PASS 16/16** | worst median 0.0043 |
| **P5** (descriptive) | reported | ρ̂_X biases ≤ 0.05 except SC σ = 0.1 ρ = 0.1 (+0.125, asymmetric leak); weighted δ-mass ≤ 0.002 everywhere |

## Headline numbers

- **The mechanism claim is confirmed decisively** — the estimator beat its
  pre-registered pass criterion by more than 10× in every one of the 16
  cells. Given the pair and orientation, on parent-event tokens, ρ̂_D
  recovers ρ with MAE ≤ 0.0026 in **every** cell — both harnesses, ρ ∈ {0.1 … 0.7}, σ ∈ {0, 0.05, 0.1} —
  with zero tuned constants, while the leak structure it corrects varies
  wildly across cells (a₀: 0.014 → 0.67; g₁: 0.005 → 0.88, strongly
  ρ-dependent on real data). The incumbent-style baseline ρ̂_C errs by
  0.13–0.49 in every leaky cell.
- **The operational (all-token) estimator inherits exactly the predicted
  background bias and nothing else.** Every cell's ρ̂_D-O deviation matches
  the measured mixture w_B·(h_B − ρ) including sign (positive at ρ = 0.1,
  ≈ 0 at ρ ≈ h_B, negative at ρ = 0.7). The two inconclusive cells are the
  ρ = 0.1 cells where |h_B − ρ| is maximal — the RC one was disclosed
  a-priori as at-risk in the lock (predicted ≈ 0.045, observed 0.0695).
  h_B is measurably not a constant (0.0–0.54 across cells): background
  removal or h_B-correction, not a fixed offset, is the way forward —
  queued, not claimed.
- **P3 is falsified as registered, by eligibility misprediction, not by the
  estimator.** The two failing cells are SC σ = 0 at ρ ∈ {0.1, 0.3}, where
  eligibility was predicted from Arm A's σ = 0 leak (a₀ ≈ 0.16). The
  round-8-E3-family harness at σ = 0 in fact gates almost cleanly
  (a₀ = 0.034–0.038, g₁ ≈ 0.006 — its host coefficient is deterministic,
  unlike Arm A's), so ρ̂_C was only 0.025–0.035 wrong and a −0.05 margin
  was arithmetically unclearable. In both cells ρ̂_D was strictly more
  accurate than the baseline (0.0013 vs 0.0350; 0.0021 vs 0.0245). The
  failure is recorded as registered; the lesson is that leak magnitudes do
  not transfer across synthetic harnesses even at σ = 0 — consistent with
  round 8b's constant-transfer failure, now on the eligibility side. This is
  pre-registration doing its job: the margin prediction was locked, it was
  wrong, and the falsification localizes *why* cleanly — a sharper outcome
  than a post-hoc-justified pass would have been.

## Scope (unchanged from the prereg)

Oracle-located, criterion-qualified pairs, oracle orientation, same
activation bank as round 8. Not established: detection-conditioned
performance, automatic orientation (ρ̂_D's swap-equivariance is a candidate
signal, untested), cross-domain transfer, background-corrected operational
estimation (h_B measured but not yet exploited).

## Data

`r9_runs.csv` (384 rows: all estimators at O/M endpoints, θ-sensitivity,
full per-run diagnostics incl. δ_J, δ_S, q_ij-B, π_B, h_B, realized ρ),
`d1_e1_recompute.csv` (D-phase), weights for every run, `r9.log`,
`smoke_r9.log`. Scoring: `analysis/analyze_round9.py` (frozen at lock).
