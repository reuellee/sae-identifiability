# Pre-registration: feature absorption under TopK SAEs (round 10)

*Drafted 2026-07-24; revised same day after two pre-lock external reviews
(Gemini 2.5 Pro: minor; **GPT-5.6: major** — 3-atom counterexample + scoring
loopholes, all addressed). Theory: `theory/topk_absorption.md`; regression
tests `theory/verify_topk_absorption.py` (pass, incl. the counterexample).
Scorer frozen at lock: `analysis/analyze_round10.py`. Status: DRAFT until the
D-phase (D1 non-regression, D2 SMOKE) is committed and the lock commit is
recorded here. After lock, changes only by dated amendment.*

## Claims under test (scoped)

Feature absorption is not an L1-shrinkage artifact, but its TopK form is
**capacity-limited**:
1. In a **two-atom** (capacity-limited) dictionary, TopK has an exact
   crossover **ε\*_TopK = 2q with no λ** (P1) and a two-atom **capacity
   collapse** (P2).
2. An **overcomplete** TopK SAE *escapes* the crossover — a free dedicated
   child atom gives a zero-loss child-recovering solution (P3, SGD behaviour).
3. Consequently **TopK resists the absorption L1 suffers** at matched
   capacity (P4, descriptive head-to-head).

Scope: 2D orthonormal toy, isolated pair (no background), the theory is an
**oracle two-atom proposition** (theory §2a) — the SGD experiment tests trained
SAEs, with encoder/gate diagnostics registered. ε=0 is excluded from recovery
claims (non-uniqueness wall, §7).

## Functional child-recovery metric (frozen)

Per trained SAE, evaluate on labeled events. A run has **child_recovered = 1**
iff some latent j satisfies ALL of (frozen thresholds):
- **signed** cos(D_j, v_c) ≥ 0.6 (positive alignment — not −v_c),
- fires on child-solo events at rate ≥ 0.5 (fires when the child appears
  alone; "fires" = post-TopK code > θ=0.05),
- child-selective: fire|csolo − fire|psolo ≥ 0.3.

This is **binary and always defined** — there is no undefined outcome and
hence no outcome-dependent missingness (closes GPT loophole 4); it is
**activation-aware** (a dead or −v_c-aligned column cannot pass — closes
loophole 3). **absorbed = 1** iff (not child_recovered) AND a composite latent
is present and fires on joint events (cos(D_j, (v_p+v_c)/√2) ≥ 0.9, fire|joint
≥ 0.5) — absorption defined as *absence of a functional child representation*,
not mere presence of a composite (per GPT's theory correction). Geometric
φ_child is recorded as a **secondary diagnostic only**. Cell statistic =
**recovery_rate** = mean child_recovered over the cell's 24 seeds; 10k-seed
bootstrap CI reported. Registered diagnostics: composite presence, conditional
fire rates on joint/parent-solo/child-solo, achieved vs oracle reconstruction.

## Arms and grids (fresh seeds 0–23; isolated pair, n_bg=0)

- **Arm A (exact two-atom):** m=2, d=8, TopK k∈{1,2}, q∈{0.1,0.2} (p=q),
  ε∈{0,0.05,0.10,0.15,0.20,0.30,0.40,0.60}, 15k steps, lam=0. (m=2 gives the
  pair exactly two atoms, instantiating the theorem; ε extends to 0.60 so the
  q=0.2 crossover is falsifiable past 1.3·2q=0.52 — GPT loophole 7.) 768 runs.
- **Arm B (overcomplete TopK):** m=16, d=16, TopK k=1, q=0.2, p=0.2,
  ε∈{0.05,0.10,0.20,0.40}, 15k steps, lam=0. 96 runs.
- **Arm C (overcomplete L1 control):** m=16, d=16, no TopK, lam=0.2, q=0.2,
  p=0.2, ε∈{0.05,0.10,0.20,0.40}, 15k steps. 96 runs.

Batched per group. Weights saved (`weights_r10_*.pt`).

## Registered predictions

Let ε_mid(series) = smallest ε where recovery_rate crosses 0.5 (linear
interpolation on the ε-grid), right-censored to ">0.60" if never.
Outcome names are explicit: **pass / FALSIFIED / inconclusive**; how an
inconclusive cell affects each claim is stated per prediction.

- **P1 (two-atom crossover + q-scaling, arm A k=1) — PRIMARY.** (a) recovery_rate
  is low at small ε and rises to ≥0.75 by ε=0.60; (b) ε_mid(q=0.2) >
  ε_mid(q=0.1) (q-scaling); (c) ε_mid(q) ∈ [0.3·2q, 1.0·2q]. **FALSIFIED** if
  ε_mid(0.2) ≤ ε_mid(0.1), or either ε_mid outside [0.15·2q, 1.3·2q]
  ([0.03,0.26] at q=0.1; [0.06,0.52] at q=0.2); **inconclusive** if a required
  ε_mid is right-censored (transition beyond the grid).
- **P2 (two-atom capacity collapse, arm A) — PRIMARY.** Predesignated
  tight-budget-absorption cells: (q=0.2, ε∈{0.05,0.10}) and (q=0.1, ε=0.05).
  P2 can **confirm only if** these cells absorb at k=1 (recovery_rate ≤ 0.25);
  a cell where k=1 already recovers is **vacuous and blocks confirmation** (not
  a pass — GPT loophole 6). In each predesignated cell that does absorb at
  k=1, k=2 recovery_rate ≥ 0.75 with gap (k2−k1) ≥ 0.5. **FALSIFIED** if any
  such absorbing cell has k=2 recovery_rate ≤ 0.5 or gap ≤ 0.25 — **including
  a failure confined to ε=0.05**, which is scored FALSIFIED (localized) and
  *diagnosed* as an SGD/encoder reachability limit; the diagnosis qualifies
  interpretation but does NOT change the verdict (GPT loophole 5). If none of
  the predesignated cells absorb at k=1, P2 = **inconclusive (tight-budget
  absorption not instantiated)**.
- **P3 (overcomplete escape, arm B, SGD behaviour).** recovery_rate ≥ 0.75 at
  ε ≥ 0.10 (SGD finds the zero-loss child-recovering solution). Reported as
  SGD-behaviour, **not** confirmation of the two-atom global-optimum theory.
  **FALSIFIED** if recovery_rate ≤ 0.25 at ε=0.20.
- **P4 (TopK resists L1 absorption, descriptive).** At ε∈{0.05,0.10}, report
  recovery_rate for arm B (TopK) vs arm C (L1); directional prediction
  TopK > L1. No pass/fail bar (the isolated-setup L1 behaviour is not
  pre-certain); the observed contrast is reported as-is.

Honest-scoring: all scored by the frozen `analysis/analyze_round10.py`;
failures included; report follows the outcome.

## Dev phase (before lock)

- **M0** — `theory/verify_topk_absorption.py`: 2-atom losses, ε\*=2q, tilt,
  AND the 3-atom zero-loss counterexample + wide-cone k=2 scope checks.
  **DONE, all regression tests pass.**
- **D1** — non-regression: the round-8 exploratory TopK cell
  (`results/round8/r8syn_runs.csv`, E3topk) has **ε=0 child-solo** (child only
  co-fires with host in that sampler), so its observed composite formation is
  the ε=0 non-uniqueness wall — consistent with §7, not evidence about the
  ε>0 crossover. **DONE (recorded here).**
- **D2** — SMOKE end-to-end (CPU): pipeline + activation-aware binary metric
  run cleanly (signed cos, child-solo fire rates, recovery flag).
  **DONE 2026-07-24.**
- At lock: experiment, metric thresholds, arms, seeds, bars, and the scorer
  are frozen. **LOCKED 2026-07-24; lock commit hash recorded by the
  immediately-following amendment commit, after which nothing changes except
  by dated amendment.**

## Cost & ops

One L4 session (~1.5 h: 768 + 96 + 96 batched runs). Weights saved; spot box
deleted after collection (`ops/retry_round10.sh`, spot-first). Results CSV +
SUMMARY committed from the GPU box.

## Review provenance

Pre-lock: Gemini 2.5 Pro (minor; algebra verified) and GPT-5.6 (major;
3-atom counterexample, scoring loopholes) — verdicts + point-by-point
responses archived in `reviews/`. All GPT required changes applied (m=2 exact
arm added; m=16 reclassified as SGD-behaviour; activation-aware metric;
missingness loophole removed by a binary metric; ε=0.05 stays FALSIFIED;
vacuous cells block; ε grid extended to 0.60; composite/uniqueness/ε=0
overclaims corrected; counterexample added to the verifier; scorer committed
before lock). Lock hash recorded by amendment.
