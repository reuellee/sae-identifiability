# Feature absorption under TopK sparse autoencoders

*Theory note, 2026-07-24. Numerically verified end to end in
`theory/verify_topk_absorption.py` (per-event losses, pure-strategy
crossover, continuous-optimum tilt, capacity collapse — all checks pass).
Status: pre-experiment. The confirmatory GPU test is pre-registered in
`notes/prereg-topk-absorption.md`.*

## 1. Why this matters

All of the project's exact theory (the ε\* pure-strategy crossover §4, the
non-identifiability wall §3, the coherence-penalty no-go §7.1b) is derived
for **vanilla L1 ReLU SAEs**. But the SAEs practitioners actually train are
**TopK / BatchTopK / JumpReLU** — architectures with a hard sparsity budget
or a firing threshold and *no L1 shrinkage*. If absorption were an artifact
of the L1 penalty, the whole program would be a curiosity. This note shows
it is not: absorption survives TopK, with a **different and in fact cleaner**
governing quantity — capacity, made explicit as the budget k.

## 2. Model

The same solvable 2D toy as the L1 theory (`verify_absorption_theory.py`):
orthonormal parent/child directions v_p, v_c; events **joint**
x = v_p + v_c (prob q), **parent-solo** x = v_p (prob p), **child-solo**
x = v_c (prob ε). The SAE encodes, keeps the **top κ pre-activations**
(ReLU → hard TopK, no L1 term), and decodes with unit-norm atoms and
**nonnegative codes**. κ is the budget available *to this pair's subsystem*;
§9 maps it to a global TopK SAE's k.

Two candidate per-pair dictionaries, as in the L1 analysis:
- **Faithful** {v_p, v_c}.
- **Absorbed** {v_p, d_comp}, d_comp = (v_p + v_c)/√2.

Per-event loss of a decoder under budget κ:
L_κ(U, x) = min over κ-atom subsets S, min over f ≥ 0 of ‖x − U_S f‖².

## 3. Exact per-event losses (verified)

| event | Faithful κ=1 | Absorbed κ=1 | Faithful κ=2 | Absorbed κ=2 |
|---|---|---|---|---|
| joint (q) | **1** | 0 | 0 | 0 |
| parent-solo (p) | 0 | 0 | 0 | 0 |
| child-solo (ε) | 0 | **½** | 0 | **½** |

Two entries carry the whole story:

- **Faithful, joint, κ=1 → loss 1.** With only one slot, faithful must drop
  one of the two equally-projecting features (⟨v_p+v_c, v_p⟩ = ⟨v_p+v_c, v_c⟩
  = 1); it reconstructs one, leaving a unit residual. The composite reconstructs
  both at once (⟨v_p+v_c, d_comp⟩ = √2, exact) — this is why absorption is
  *attractive* under a tight budget.
- **Absorbed, child-solo, κ ≥ 2 → loss ½ (the nonnegativity signature).**
  To extract v_c from the composite you must subtract the parent:
  v_c = √2·d_comp − v_p, coefficient −1 on v_p. ReLU forbids negative codes,
  so the best nonnegative reconstruction of v_c is the projection of v_c onto
  the non-orthogonal cone spanned by {v_p, d_comp}, which is ½(v_p + v_c),
  residual ½(v_c − v_p), loss ½ — **even with a second slot available**. The
  composite cannot be "undone." This is why a second slot does not rescue the
  absorbed dictionary, and why the capacity collapse below is sharp.

## 4. The pure-strategy crossover: ε\*_TopK = 2q (no λ)

Expected losses at κ=1: E[L_faithful] = q, E[L_absorbed] = ½ε. The pure
absorbed strategy beats the pure faithful strategy iff

  **½ε < q  ⟺  ε < ε\*_TopK,  ε\*_TopK = 2q.**

The TopK crossover contains **no shrinkage coefficient**: there is no λ. In
L1, absorption is bought by the soft-threshold tax λ/2 paid per active
coordinate, and ε\*_L1 ≈ 1.17·λ·q → 0 as λ → 0 (absorption is
*shrinkage-driven*). Under TopK there is no such tax; absorption at κ=1 is
bought purely by the **one-slot budget**, and the boundary 2q is a property
of the occurrence statistics alone. The two mechanisms are complementary:
L1 absorption vanishes without shrinkage; TopK absorption persists without
it, gated instead by capacity. (For typical λ the L1 boundary is far smaller
— λ=0.2 gives ε\*_L1 ≈ 0.23·q vs ε\*_TopK = 2q — but the comparison is across
different capacity regimes, so it is a statement about mechanism, not a
horse-race.)

## 5. The continuous optimum tilts (ε\* organizes, does not equal)

Exactly as in L1 §5, the pure candidates are not the global optimum. A scan
over all unit-norm 2-atom dictionaries under the κ=1 budget (verified) shows
the global optimum is a **tilted frame** whose child-side atom moves smoothly
with ε:

| ε | 0 | 0.1 | 0.2 | 0.3 | ≥ 0.4 |
|---|---|---|---|---|---|
| child-side atom | **45°** (composite) | 52° | 58° | 90° | 90° (faithful child) |

At ε = 0 the global κ=1 optimum is exactly the pure absorbed dictionary
(45° composite, loss 0 — the non-identifiability wall: the child never
appears, so it cannot be recovered and absorption is optimal). As ε grows the
child-side atom rotates toward 90° (the true child direction); the parent-side
atom co-rotates to ~32° (under a one-slot budget even the faithful regime
compromises the parent atom to better catch joint events). So ε\*_TopK = 2q
**organizes** the transition — it is the crossover of the two pure strategies
— while the trained SAE traces the smooth tilt through it, and SGD will
transition at some fraction of 2q (the L1 analogue transitioned at
0.58–0.70·ε\*).

## 6. The capacity collapse (the sharp result): κ ≥ 2 ⟹ no absorption for ε > 0

At κ = 2 the global optimum is the **faithful** frame with **loss 0** for
every ε ≥ 0, and any dictionary containing the composite direction is
**strictly worse** (it still pays ½ε on child-solo events, §3). Verified: at
κ=2 the global scan returns loss 0 (achieved by faithful and its wide-cone
neighbours) while the absorbed configuration sits at ½ε. Therefore

  **ε\*_TopK(κ ≥ 2) = 0:** a second slot eliminates absorption for any ε > 0.

The whole absorption region ε ∈ (0, 2q) at κ=1 **collapses to the single
point ε = 0** (the information wall) once the budget reaches 2. Capacity does
not merely shift the boundary — it removes the phenomenon. The mechanism is
the §3 nonnegativity signature: the second slot lets *faithful* reach zero
loss, but does not let *absorbed* separate the child, so faithfulness wins
outright.

## 7. Contrast with L1, sharpened

- **L1, spare capacity (§4 Remark):** for any ε > 0 the unconstrained optimum
  is the *redundant triple* {parent, child, composite} — the λ tax makes a
  dedicated composite atom actively worth having (it reconstructs joint events
  1-sparsely, saving λ/2). L1 *wants* composition when it can afford it.
- **TopK, spare capacity (§6):** the composite atom is worth *nothing* — with
  no λ to save, reconstruction is already perfect with {v_p, v_c}, and the
  composite only adds a direction that cannot separate the child. TopK is
  indifferent-to-averse to composition once capacity is spare.

So the same word "capacity" plays opposite roles: under L1 it converts to the
soft-threshold absorption tradeoff (Theorem 2); under TopK it *is* the control
knob, and relaxing it kills absorption. This is the capacity thesis (§8,
round-8 m≥33 rerun) in its cleanest exact form.

## 8. JumpReLU (predicted, secondary)

JumpReLU keeps activations above a threshold θ unchanged and zeroes the rest —
no shrinkage, no hard count budget, a **magnitude gate**. A dedicated child
atom fires on child-solo events only if the child-solo activation clears θ;
when child-solo events are rare or weak the child atom is pruned, favouring
absorption. We predict JumpReLU sits between L1 and TopK: a threshold-gated
absorption region that shrinks as θ falls (more atoms survive) — analogous to
TopK's capacity relaxation but driven by the gate rather than the count. This
is registered as a secondary exploratory arm, not a confirmatory claim.

## 9. From subsystem budget κ to a global TopK SAE's k

A real TopK SAE has a global budget k (active latents per token out of m).
The pair's *effective* subsystem budget κ is k minus the slots consumed by
the other features firing on that token. With B background features active on
a joint event, κ ≈ k − B. So:
- **Tight global k** (k ≈ B + 1): the pair competes for ~1 slot → κ=1 regime →
  absorption for ε < 2q.
- **Spare global k** (k ≥ B + 2): the pair can claim 2 slots → κ≥2 regime →
  faithful, no absorption for ε > 0.

The experiment sweeps k across this range at a fixed background load to move
the pair through the collapse, and separately isolates the pair (no
background) to test the κ=1 crossover and the κ=2 collapse directly. The
round-8 exploratory TopK cell (k=4, m=32, d=64, 30 background features at
rate 0.08 → ~2.4 active, ε=0.10) sat in the tight regime (κ ≈ 1) with
ε=0.10 < 2q=0.4, and observed absorption — consistent with this theory.

**Background load is stochastic.** B is a per-token random variable
(Binomial(n_bg, bg_rate)), so κ = k − B fluctuates across tokens even at fixed
k, and the trained SAE optimizes the *average* over that distribution. The
idealized sharp κ=2 transition therefore manifests in the Arm C sweep as a
**softer, gradual** rise of φ with k rather than a step. The isolated-pair
Arm M (where the pair reliably wins its slots, §10) is the clean test of the
exact κ=1 vs κ=2 predictions; Arm C tests that capacity gating survives
realistic per-token budget noise.

## 10. Predictions for the SGD experiment

*Caveat (SGD vs. global optimum): the theory characterizes the global optimum
of the population loss; the experiment trains with SGD. For very small ε the
absorbed and faithful optima differ by only ½ε, so the gradient pushing SGD
out of an absorbed basin is weak — SGD may remain absorbed even where the
global optimum is faithful. A capacity-collapse failure confined to the
smallest ε is therefore evidence about SGD reachability, not about the
theorem (which M0 already proves); see the prereg's T2 interpretation clause.*

Measuring the child-recovery angle φ (0°=parent, 45°=composite/absorbed,
90°=faithful child), as in the L1 experiments:
- **T1 (crossover scale, isolated pair, k=1):** absorption (φ near 45°) for
  ε small, tilting toward 90° as ε grows, with the transition on the **2q
  scale** (SGD midpoint predicted below 2q, as in L1; a specific bar is
  registered in the prereg).
- **T2 (capacity collapse, isolated pair):** at k=2 the pair recovers the
  faithful child (φ near 90°) for every ε > 0 tested down to a small ε_min —
  no systematic absorption — in sharp contrast to k=1.
- **T3 (capacity gating, with background):** at fixed ε in (0, 2q), sweeping
  the global k moves φ from absorbed (small k) to faithful (large k); the
  transition k tracks the background load (κ ≈ k − B crossing 2).
- **T4 (λ-independence):** the k=1 crossover scale does not depend on adding a
  small L1 term (there is none in the TopK objective) — a negative control
  distinguishing capacity- from shrinkage-driven absorption.

## 11. Scope and what would falsify this

- **Model scope:** 2D orthonormal toy, population loss, ideal budget. The SGD
  experiment is the empirical test; encoder/gating effects (as seen in the
  round-9 leak analysis) may shift the transition off the exact 2q.
- **Falsifiers:** (i) at k=2 (spare, isolated pair) the trained SAE still
  absorbs the child at small ε > 0 (would break the capacity-collapse
  theorem's SGD relevance); (ii) the k=1 transition scale is independent of q
  (would break ε\* = 2q); (iii) increasing k with background present does *not*
  relieve absorption (would break capacity gating). Each is a registered
  prediction with a falsification bar.
- **Non-identifiability wall is architecture-independent:** at ε = 0 the child
  never appears alone and is information-theoretically unrecoverable under any
  architecture (Theorem 1); TopK with any k cannot recover it. The capacity
  collapse is about ε > 0, not ε = 0.
