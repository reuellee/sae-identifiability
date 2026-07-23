# Containment-orientation prereg: S1/S2 outcome — P1 FALSIFIED

Pre-registered in `notes/prereg-containment-orientation.md` (lock `0a6db51`,
Amendment 1 same commit). S1 = free CPU reanalysis of frozen E1 weights
(non-regression + pilot for Amendment 1). S2 = confirmatory GPU run,
`experiments/round8b_orientation.py`, dev-gpu, weights saved
(`results/round8/weights_r8b_orient.pt`), commit `72a3dc7`.

## S1 (development-set, non-regression)

Full 48-run E1 re-analysis (m=128/256, real GPT-2 layer-6 activations,
ε=0.002): plain firing-containment is **1.000 accurate when determinate,
but indeterminate 54% (m=128) / 79% (m=256)** — the raw containment gap
collapses to ~2–3 percentage points under leaky/gated absorption, close to
the registered MARGIN. Amendment 1 (magnitude cross-delta primary rule,
locked before S2) fixes S1's coverage: **1.000 accuracy, zero unresolved,
both widths** — but this is calibration data (the rule was tuned on an
8–16-seed pilot subset of this same weight set), reported as
development-set per the amendment's own disclosure, not confirmatory.

## S2 (confirmatory, held-out cells) — **P1 FALSIFIED**

| ρ | formed/flagged | rarity acc | containment acc (indet) | amended acc |
|---|---|---|---|---|
| 0.10 | 21/24, 21/21 | 1.000 | 1.000 (0/21) | **1.000** |
| 0.50 | 22/24, 22/22 | 0.273 | 0.143 (8/22) | **0.455** |
| 0.60 | 21/24, 21/21 | 0.000 | 0.000 (0/21) | **0.095** |
| 0.80 | 21/24, 21/21 | 0.000 | 0.000 (1/21) | **0.143** |

**P1 (registered: ρ=0.6 amended accuracy ≥ 0.90, falsified < 0.70) is
FALSIFIED at 0.095** — far below the falsification bar, and below chance
(0.5), meaning the rule is not merely uninformative but **systematically
wrong** most of the time. P2 (ρ=0.8 ≥ 0.80) and P3' (ρ=0.10 non-regression)
are also not met at 0.8 (0.143), though P3' at ρ=0.10 passes (1.000).

## Diagnosis (ground-truth + saved-weight recompute, not part of the lock)

The qualitative anti-rotation mechanism replicates cleanly in S2 (par:
mean act ≈0.8 on host-only / ≈0.15 on joint; comp: ≈0.15 on host-only /
≈1.2 on joint — same sign as E1's pilot). The **sign** of the label-free
cross-conditioned delta (`delta_comp > delta_par`) mostly still holds, but
the **magnitude is 5–10× smaller** than in E1's pilot (S2 deltas ≈0.05–0.5
vs E1's pilot ≈0.76–2.2) — this m=32 synthetic harness produces a much
weaker anti-rotation swing than the m=128/256 real-data harness. Nearly
every S2 case falls under MARGIN_MAG=0.3 (calibrated on E1's scale) and
defers to the containment fallback, which resolves confidently but is
**itself systematically wrong** at ρ≥0.5 in this harness (rate ordering
flips: comp's firing rate exceeds par's once ρ>0.5, which also explains
rarity's registered-and-confirmed 0.000).

**Conclusion: MARGIN_MAG (and likely the underlying signal scale) does not
transfer across harnesses/widths.** The S1 "fix" was fit to E1's specific
scale, not a harness-invariant property of absorption. The orientation
problem, expanded scope: solving it requires either (a) a harness-adaptive
threshold (e.g. calibrated per-SAE from its own null distribution, not a
fixed constant) or (b) a genuinely scale-invariant statistic (e.g. a
rank/percentile-based version of the cross-delta, or a likelihood-ratio
test rather than a raw magnitude gap) — both untested here. **Do not reuse
MARGIN_MAG=0.3 or MARGIN=0.02 outside the E1 regime they were calibrated
on.**

## Status

Orientation remains the pipeline's weakest, and now more clearly
**unsolved**, stage outside the specific low-ρ / real-data regime already
validated (round 8: 0.988/0.909 auto, 0.990 oracle). `RESEARCH_PLAN.md`
queue item updated accordingly — a harness-adaptive or scale-invariant
orientation statistic is the queued successor, not a further tuning pass
on this rule.
