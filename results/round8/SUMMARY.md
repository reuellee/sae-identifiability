# Round 8: v1.2 passes width-specific held-out endpoints; detector scales in the synthetic family; orientation isolated as the weak stage

Pre-registered (`notes/prereg-round8-scaling-robustness.md`, pre-results lock
`a539c76`, pre-collection amendment `69ca642` — width-specific endpoints,
stage separation, corrected S1 matcher — committed while the run was in
flight). One L4 session; E1 64 runs + E2/E3 128 runs; weights saved.

## E1 — v1.2 same-domain resampling stability (confirmatory)

48/48 absorbed-formed (0 exclusions), 24 fresh seeds per width:

| Endpoint | m=128 | m=256 | Verdict |
|---|---|---|---|
| T1a/T1b recall | **24/24 = 1.000** (lift 1.975±0.026) | **24/24 = 1.000** (lift 2.055±0.024) | **PASS at each width** |
| T2 oracle-pair faithful flag rate | 0.000 | 0.000 | **PASS** |

Seed-bootstrap CIs are degenerate at [1.000, 1.000] (every seed succeeded);
the honest uncertainty statement is the exact binomial: 24/24 → 95% lower
bound ≈ 0.86 per width, 48/48 pooled ≈ 0.93. Scope: same GPT-2 domain
(model, layer, corpus, injection protocol) — a resampling-stability result
for v1.2, not cross-domain transfer. Arm 2's L_HI=2.0 outcome stands as the
completed prior result; the lift clusters (1.975/2.055) confirm the
width-dependent calibration reading — L_HI=1.9 clears both, but cutoff
transfer across layers/models/architectures remains untested.

**Stage-separated endpoints (amendment §3), conditional on detection:**

| Stage | m=128 | m=256 |
|---|---|---|
| Orientation accuracy (rarity rule) | 0.88 | 0.75 |
| Child recovery, auto orientation | 0.948 | 0.909 |
| Child recovery, **oracle orientation** | **0.990 ± 0.001** | **0.990 ± 0.001** |
| ρ̂ (counting; true 0.5) | 0.755 | 0.746 |

Residualization is near-perfect given correct orientation; all auto-mode
degradation is orientation error. Orientation is now the pipeline's weakest
validated stage (0.00 under E3's prevalence inversion, by construction of
the rarity rule) — a containment-based orientation rule is the natural next
pre-registration. ρ̂ remains leak-inflated as expected (gating-corrected
estimator queued).

**All-pairs context (amendment §4):** every faithful-control SAE has ≥1
full-scan flag (proportion 1.00 at both widths; 3.8–10.4 flags/SAE,
320–465/M pairs) — candidate counts do NOT separate absorbed from faithful
conditions; these are real-background candidates of unknown status. The
no-injection real-background null remains queued (round 8b).

## E2 — proportional-scale null calibration (pre-registered descriptive)

| d | m | pairs | FP/SAE (null) | FP/M | recall (formed) | formed | wall s | mem MB |
|---|---|---|---|---|---|---|---|---|
| 64 | 32 | 496 | 0.00 | 0 | 1.000 | 6/8 | 33 | 108 |
| 128 | 64 | 2,016 | 0.00 | 0 | 1.000 | 8/8 | 45 | 204 |
| 256 | 128 | 8,128 | 0.00 | 0 | 1.000 | 8/8 | 140 | 407 |
| 512 | 256 | 32,640 | 0.00 | 0 | 1.000 | 4/8 | 325 | 857 |

The registered soft expectation (FP/M falls with width) is exceeded: v1.1
produced **zero** null false positives at every scale in this synthetic
family — the overlap veto fully suppresses the splitting-doublet mode that
dominated Arm 1's FP count. Absorption *formation* declines at the largest
scale (4/8; trainability, disclosed). Scope: proportional scale family
(d, n_bg, m co-scaled); fixed-dimension width sweep and overcomplete m > d
are round 8b.

## E3 — robustness cells (pre-registered descriptive; 8 seeds each)

| Cell | formed | flagged | orient | child res (u\*) | note |
|---|---|---|---|---|---|
| angle cos=0.3 | 7/8 | 7/7 | 1.00 | 0.942 | detector + recovery robust |
| angle cos=0.5 | 6/8 | 3/6 | 1.00 | 0.933 | **detection degrades** (pair cos 0.79 nears band edge; lift 3.5) |
| prevalence ρ=0.6 | 6/8 | 6/6 | **0.00** | 0.649 | registered prediction confirmed: rarity orientation fails when composite is commoner; detection unaffected |
| TopK (k=4, λ=0) | 6/8 | 6/6 | 1.00 | **0.999** | gated absorption + detector signature survive hard-sparsity encoders |

## S1 — real-background candidate stability (exploratory; corrected matcher)

See `s1_stability.log` (bijective pair score + all-seed clustering, run
after the amendment fix and never with the defective matcher).

## Costs

E1 ≈ 11 min, E2+E3 ≈ 13 min on one L4; session ≈ $0.6.
