# Pre-registration: feature absorption under TopK SAEs (round 10)

*Drafted 2026-07-24. Theory: `theory/topk_absorption.md` (verified,
`theory/verify_topk_absorption.py`). Status: DRAFT — under dual pre-lock
external review (Gemini 2.5 Pro + GPT-5.6); locks after review changes +
the D-phase (D1 non-regression, D2 SMOKE) are committed. After lock, changes
only by dated amendment.*

## Claim under test

Feature absorption is **not** an artifact of L1 shrinkage: it occurs under
hard-budget **TopK** SAEs, where it is governed by **capacity (the budget k)**
rather than a shrinkage coefficient. Specifically (theory §4, §6): the
pure-strategy crossover is **ε\*_TopK = 2q with no λ** at a one-slot budget,
and a **second slot eliminates absorption for every ε > 0** (the capacity
collapse). The SGD experiment tests the *trained* TopK SAE's child-recovery
angle φ against these predictions.

Scope: 2D orthonormal toy, population-style training, the child-recovery
angle φ (0°=parent, 45°=composite/absorbed, 90°=faithful child) as in the L1
experiments (`sae_experiments.py`). Not tested: real-activation TopK SAEs,
large overcomplete m, non-orthogonal features (all queued).

## Measurements (frozen)

Per trained SAE, `phi_child` = the angle (in the v_p,v_c plane) of the
in-plane latent (in-plane fraction ρ = √(cp²+cc²) > 0.5) with the largest
angle in (20°,120°); NaN if no such latent exists. `cos_comp`, `cos_child`
= best |cos| of any decoder column to (v_p+v_c)/√2 and to v_c. Cell = a
(arm, k, q, ε) group of seeds; **cell median φ_child** over defined seeds is
the scored statistic. 10k-seed bootstrap CIs reported alongside.

Bands: **absorbed** = φ ≤ 55°, **faithful** = φ ≥ 75° (the 45°/90° poles with
a 10° guard; 55–75° is the transition band). φ_child is a proxy: if a
configuration systematically produces child-side latents outside the
(20°,120°) window, seeds become undefined and — if a cell drops below the
16/24 scoreability floor — that cell is reported **unscoreable (untested)**,
never a pass; an unscoreable primary cell is disclosed as a gap, not a
confirmation.

## Arms and grids (fresh seeds)

- **Arm M (isolated pair):** d=32, n_bg=6 low-amplitude background
  (bg_rate=0.05, coeff U(0.5,1.0)·0.5), m=16, 15k steps, lam=0. The background
  is deliberately low-amplitude so the unit-amplitude parent/child pair
  reliably wins the top-k slots, closely instantiating the theoretical κ=1
  (k=1) and κ=2 (k=2) subsystem-budget conditions rather than leaving the
  pair's effective budget to fluctuate. Grid: k∈{1,2} × q∈{0.1,0.2}
  (p=q) × ε∈{0,0.05,0.10,0.15,0.20,0.30,0.40} × seeds 0–23. 672 runs.
- **Arm C (realistic background, capacity sweep):** d=64, n_bg=30
  (bg_rate=0.08, unit amplitude), m=32, 15k steps, lam=0, q=p=0.2, ε=0.10
  (fixed, in (0, 2q=0.4)). Grid: k∈{1,2,3,4,6,8,12,16,24} × seeds 0–23.
  216 runs.
- **Arm J (JumpReLU, exploratory, descriptive only):** arm-M setup with a
  fixed threshold gate θ∈{0.1,0.3,0.5,0.7} × ε∈{0.05,0.15} × seeds 0–7.
  64 runs. No pass/fail bar.
- **Arm M-λ (λ-independence control, exploratory):** arm M, k=1, q=0.2,
  ε-grid, lam=0.1 (a small L1 term added to the TopK objective) × seeds 0–7.
  Descriptive; no bar.

Training is batched by (arm, k, jump, dims) so the TopK budget is uniform
within a batch. Weights saved per group (`weights_r10_*.pt`).

Exclusions (registered): seeds with undefined φ_child (no in-plane child-side
latent) are excluded and disclosed; a confirmatory cell is scoreable iff
≥ 16/24 seeds are defined. No exclusion based on the value of φ_child.

## Registered predictions

Let ε_mid(cell-series) = the smallest ε at which the cell-median φ_child
crosses 67.5° (linear interpolation on the ε-grid); right-censored to
">0.40" if it never crosses within the grid.

- **T2 (capacity collapse — PRIMARY):** in arm M, for every (q, ε) with
  0 < ε < 2q — i.e. q=0.1 × ε∈{0.05,0.10,0.15} and q=0.2 ×
  ε∈{0.05,0.10,0.15,0.20,0.30} — the **k=2** cell-median φ_child ≥ **75°**
  (faithful) AND exceeds the matched **k=1** cell-median by ≥ **15°**.
  Falsified if any such cell has k=2 median ≤ **55°** (still absorbed) or the
  k=2−k=1 gap ≤ **5°**. Precondition (reported, not a bar): the matched k=1
  medians should be absorbed (≤ 60°); if k=1 does not absorb in a cell, that
  cell's collapse test is reported as vacuous.
  **SGD-reachability interpretation (registered before results):** the bar is
  unchanged, but its meaning is scoped. M0 already proves the capacity collapse
  for the *global* optimum; T2 tests whether SGD *reaches* it. For very small
  ε the absorbed and faithful optima differ by only ½ε, so the escaping
  gradient is weak and SGD may remain absorbed. A T2 failure **confined to the
  single smallest ε** in a q-row (ε=0.05) will therefore be reported as an
  *SGD-reachability limit at small ε*, not as a refutation of the collapse
  theorem; a failure at any larger ε, or across multiple ε, counts as a
  genuine falsification of the SGD-level capacity-collapse claim. This
  distinction is fixed here, before the run, so it cannot be applied
  selectively after seeing results.
- **T1 (crossover scale + q-scaling, arm M k=1):** (a, robust core)
  ε_mid(q=0.2) > ε_mid(q=0.1) — the transition moves right with q; (b, scale)
  ε_mid(q) ∈ [0.3·2q, 1.0·2q] for both q (i.e. [0.06,0.20] at q=0.1 and
  [0.12,0.40] at q=0.2), consistent with an SGD transition at a fraction of
  2q (the L1 analogue transitioned at 0.58–0.70·ε\*). Falsified if
  ε_mid(q=0.2) ≤ ε_mid(q=0.1), or either ε_mid falls outside
  [0.15·2q, 1.3·2q].
- **T3 (capacity gating, arm C):** cell-median φ_child is absorbed (≤ 60°) at
  k=1 and rises with k, reaching faithful (≥ 75°) at some k ≤ 24, with
  φ(k=16) − φ(k=1) ≥ 20°. Falsified if φ(k=16) − φ(k=1) < 5° (capacity does
  not relieve absorption) or k=1 is already faithful (≥ 75°, no absorption at
  tight budget).

Honest-scoring rule: all three scored as registered from the frozen
`analysis/analyze_round10.py`; failures included; report follows the outcome.
Arms J and M-λ are descriptive readouts (JumpReLU absorption vs θ;
λ-independence of ε_mid).

## Dev phase (before lock)

- **M0** — theory verification (`theory/verify_topk_absorption.py`): per-event
  losses, ε\*_TopK=2q crossover, continuous tilt, capacity collapse.
  **DONE, all checks pass.**
- **D1** — non-regression: re-read the round-8 exploratory TopK cell
  (`results/round8/r8syn_runs.csv`, exp=E3topk: k=4, m=32, d=64, ε=0.10) and
  confirm it sits in the tight-budget absorbed regime the theory predicts
  (κ≈1 at k=4 with ~2.4 active background; ε=0.10 < 2q=0.4).
- **D2** — SMOKE end-to-end on CPU (orchestrator).
- At lock: experiment code, φ measurement, grids, seeds, bands, and bars are
  frozen; the lock commit hash is recorded here by amendment.

## Cost & ops

One L4 session (~1.5–2 h: 672 + 216 + 64 + ~56 small batched runs). Weights
saved; `dev-gpu stop` / spot-instance delete immediately after collection
(`ops/retry_round10.sh`, spot-first). Results CSV + SUMMARY committed from the
GPU box.

## Review provenance

Dual pre-lock external review (Gemini 2.5 Pro via Vertex; GPT-5.6 Sol via
chatgpt.com) — verdicts + applied changes archived in `reviews/`; lock hash
recorded by amendment.
