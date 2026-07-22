# Pre-registration: within-composite bimodality estimator for label-free child frequency

**Date:** 2026-07-22
**Status:** pre-registered, **not yet run**. Companion to
`notes/label-free-frequency-identifiability.md`.
**Cost estimate:** small; CPU-feasible for the synthetic arm (no dev-gpu needed). Real-data
arm reuses existing GPT-2 activation pipeline (`experiments/extract_activations.py`).

This document fixes hypotheses, design, metrics, and pass/fail thresholds **before** running,
so outcomes cannot be rationalized after the fact — matching this project's pre-registration
discipline (README: "all predictions pre-registered").

## Hypotheses

- **H1 (no-go, confirmatory).** For a fully absorbed parent/child pair, the distribution over
  **binarized** co-firing signatures is invariant to the child base rate `ρ`.
- **H2 (estimator, confirmatory).** The **continuous** within-composite bimodality estimator
  recovers `ρ` where H1's binarized route cannot.
- **H3 (SNR boundary, exploratory).** There exists a noise level `σ*` above which the two
  activation modes merge and H2's estimator fails; locate it.
- **H4 (multi-child, exploratory).** With `m>1` children in one composite, the scalar
  estimator recovers only aggregate frequency; a `k`-component fit with `k=m+1` degrades as
  modes overlap.

## Design

### Arm A — synthetic (decisive for H1/H2/H3)

Generative model matching `theory/`: orthonormal `v_p, v_c` in `R^d` (`d=64`), background of
`n_bg` orthonormal distractors firing i.i.d. Bernoulli. Events:
`host-only` w.p. proportional to `(1−ρ)`, `host+child` w.p. proportional to `ρ`, background
otherwise. Additive isotropic Gaussian activation noise, std `σ` (swept).

- Train a vanilla L1 ReLU SAE (same trainer as `experiments/`) to the **confirmed absorbed**
  regime (verify a single composite latent captures both sub-populations via decoder cosine to
  `(v_p+v_c)/√2 > 0.98`). Reject/retrain seeds that do not absorb.
- **Base rates:** `ρ ∈ {0.02, 0.05, 0.10, 0.20}`. **Seeds:** 16 per cell (matches round-2).
- **Noise sweep (H3):** `σ ∈ {0, 0.05, 0.1, 0.2, 0.4}` at fixed `ρ=0.10`.
- **Multi-child (H4):** `m ∈ {1,2,3}` children with distinct base rates at `σ=0.1`.

### Arm B — real data (transfer check, gated on Arm A success)

GPT-2-small layer-6, capacity-limited SAE (`m=128/256`, the regime where §15 found true
absorption). Use an oracle-labeled parent/child pair (e.g. a first-letter feature and a
specific token, per Chanin) **only to score** the label-free estimate — the estimator itself
sees no labels. Run only if Arm A confirms H2.

## Metrics & pass/fail thresholds

| # | Metric | Predicted | Pass condition |
|---|---|---|---|
| M1 (H1) | TV distance between binarized signature distributions at `ρ₁=0.02` vs `ρ₂=0.20`, absorbed SAEs | `≈ 0` | `TV < 0.02` and not significantly `> 0` across seeds (bootstrap 95% CI includes 0) |
| M2 (H2) | `|ρ̂ − ρ|` from the 2-component fit on composite activations, averaged over cells | small | `mean |ρ̂−ρ| < 0.02` **and** Pearson `r(ρ̂, ρ) > 0.9` across the four `ρ` |
| M3 (H2 vs H1) | Δ = M1-route error − M2-route error | M2 ≫ M1-route | estimator route strictly better at every `ρ` |
| M4 (H3) | first `σ` at which `|ρ̂−ρ| > 0.05` | some finite `σ*` | report `σ*`; H3 confirmed if estimator degrades monotonically past it |
| M5 (H4) | per-child recovery error vs `m` | degrades with `m` | report; H4 confirmed if aggregate recovers but per-child error grows with overlap |

## What would falsify each claim

- **H1 falsified** if `TV > 0.05` reliably — i.e. some second-order binarized signal *does*
  leak `ρ` (would contradict §2; investigate background non-independence).
- **H2 falsified** if `mean |ρ̂−ρ| ≥ 0.05` or `r ≤ 0.9` even at `σ=0` — the bimodality does
  not cleanly encode `ρ` (would undercut §3's core claim).
- **Estimator is uninteresting** if M3 shows no advantage over the binarized route — then §3
  adds nothing over §15b.

## Analysis plan (fixed in advance)

- Mixture fit: 2-component 1-D GMM (EM, 10 restarts) on nonzero composite activations; `ρ̂` =
  weight of the higher-mean component. No post-hoc component-count tuning in Arm A confirmatory
  cells (H4 exploratory only).
- Report all seeds; no seed dropping except documented non-absorbing seeds (criterion above),
  disclosed with counts — per the round-3 seed-variance-disclosure precedent.
- Confidence intervals by seed bootstrap.

## Decision rule → next step

- **H1 ∧ H2 confirmed:** the estimator is real; promote to Arm B, and (if Arm B holds) this is
  the label-free remedy that closes §15b. Write up as a result, not a note.
- **H1 confirmed, H2 falsified:** the no-go stands but the escape fails — publish the no-go
  alone (still resolves the binarized route) and fall back to Matryoshka architecturally.
- **H1 falsified:** re-open §2; a binarized signal exists after all.

---

## Outcome (2026-07-22, post-run — original text above unmodified)

Run on one L4 session (256 SAEs, ~$1.4), pre-results commit `cfd3e09`; scored
by `analysis/analyze_prereg_armA.py`; full readout `results/prereg_armA/SUMMARY.md`.

- **H1: falsified in mechanism.** Registered excess TV 0.0215 (CI [0.0210,
  0.0220], excludes 0) — attenuated only by background dilution; in-plane
  excess TV 0.044 ≈ the maximum possible leak, and TV(host-only vs host+child
  signatures) = 0.9999. Decision rule taken: **re-open §2 of the
  identifiability note.**
- **H2: FALSIFIED at σ=0** (mean err 0.483, r 0.234): trained absorption is
  encoder-gated (composite ~never fires on host-only events, act ≈ 0.01 vs
  1.28), so there is no second mode to fit. Works at σ=0.1 (err 0.0075).
- **M3 inverted:** the binarized signature-count route beats the mixture
  estimator at every ρ (err ≤ 0.02), given the oracle-identified pair.
- **H3 premise inverted:** estimator improves with σ up to 0.1; absorption
  itself vanishes at σ ≥ 0.2 (SAEs go faithful — noise-as-remedy, new lead).
- **H4 untestable as configured:** m ≥ 2 children at these rates are erased,
  not absorbed (0/16 mono-composites).
- Disclosed exclusions: absorption-formation is trainability-limited at low ρ
  (4/16 absorbed at ρ=0.02 → 14/16 at ρ=0.20); non-absorbed runs are
  child-erased, not faithful.
