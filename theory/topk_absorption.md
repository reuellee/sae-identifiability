# Feature absorption under TopK SAEs: a two-atom oracle crossover, and the overcomplete escape

> **STATUS BANNER (updated 2026-07-24 after round 10 + the whole-repo review).**
> The results below are the **oracle two-atom propositions** (verified). The
> *practical* prediction that "trained overcomplete TopK resists the absorption
> L1 suffers" was **contradicted in round 10's tested isolated regime** (there
> overcomplete L1 recovered the child in 100% of runs vs TopK 0.62–0.83; the
> registered trained-SAE crossover was inconclusive and the capacity-collapse
> prediction was falsified — `results/round10/SUMMARY.md`). Round 11's
> background-rich comparison is **unresolved** (its detector count was
> confounded; the semantic claim is withdrawn — `results/real/SUMMARY.md`). So
> this note is a statement about a *capacity-limited two-atom oracle model and
> an overcomplete escape*, **not** a claim that TopK is more absorption-resistant
> in practice.

*Theory note, 2026-07-24, revised after two pre-lock external reviews
(Gemini 2.5 Pro: minor; **GPT-5.6: major** — it constructed the 3-atom
zero-loss counterexample that scopes the crossover) and the whole-repo review.
Numerical regression tests + scope checks in
`theory/verify_topk_absorption.py` (all pass, incl. the counterexample).*

## 1. Why this matters

All of the project's exact theory (the ε\* crossover §4, the wall §3, the
coherence no-go §7.1b) is for vanilla **L1** ReLU SAEs. Practitioners train
**TopK / BatchTopK / JumpReLU** — a hard per-token budget or a firing
threshold, **no L1 shrinkage**. Does absorption survive? The answer is
two-sided and sharper than a naive port: under a **capacity-limited**
(exactly-two-pair-atom) dictionary, TopK has an exact absorption crossover
**ε\*_TopK = 2q with no λ**; but an **overcomplete** dictionary *escapes* it,
because a free dedicated child atom gives a zero-loss child-recovering
solution. The *conjecture* this motivated — that TopK would therefore resist
L1's absorption in practice — did **not** survive testing (see the status
banner): round 10 found the opposite in its isolated regime, and round 11's
background-rich test is unresolved. This note is the two-atom oracle theory
plus the overcomplete-escape observation, not a practical resistance claim.

## 2. Model and the exact object being analysed

The 2D toy of `verify_absorption_theory.py`: orthonormal parent/child
directions v_p, v_c; events **joint** x = v_p + v_c (prob q), **parent-solo**
x = v_p (prob p), **child-solo** x = v_c (prob ε). Unit-norm decoder atoms,
**nonnegative (ReLU) codes**.

**(2a) The analysed object is a two-atom oracle-coded k-sparse dictionary.**
We compare per-pair dictionaries with *exactly two* atoms — faithful
{v_p, v_c} vs absorbed {v_p, d_comp}, d_comp = (v_p+v_c)/√2 — under an
**oracle** k-sparse nonnegative code
L_κ(U, x) = min over κ-atom subsets S, min_{f ≥ 0} ‖x − U_S f‖². This is an
**oracle lower bound** on what a *learned* TopK encoder achieves: a real
encoder selects atoms by learned pre-activations and uses learned
magnitudes, so the trained loss landscape can differ. The propositions below
are analytic statements about this oracle two-atom model; the SGD experiment
(§10) tests whether trained TopK SAEs behave accordingly, with encoder/gate
diagnostics registered to connect the two.

**(2b) Absorption is the absence of a functional child representation** — not
the mere presence of a composite atom. A dictionary can contain a composite
atom *and* a faithful child atom (the redundant triple, §6); that is not
absorption. Operationally (prereg): the child is *recovered* iff some latent
is positively aligned with v_c, fires on child-solo events, and is
child-selective; absorption is the failure of that.

## 3. Exact per-event losses, two-atom (verified)

| event | Faithful κ=1 | Absorbed κ=1 | Faithful κ=2 | Absorbed κ=2 |
|---|---|---|---|---|
| joint (q) | **1** | 0 | 0 | 0 |
| parent-solo (p) | 0 | 0 | 0 | 0 |
| child-solo (ε) | 0 | **½** | 0 | **½** |

- **Faithful, joint, κ=1 → 1:** one slot must drop one of two
  equally-projecting features. The composite reconstructs both at once
  (⟨v_p+v_c, d_comp⟩ = √2). This is why absorption is attractive under a tight
  budget *with only two atoms*.
- **Absorbed, child-solo → ½ (nonnegativity):** extracting v_c from the
  composite needs a negative parent coefficient (v_c = √2·d_comp − v_p);
  ReLU forbids it, so the best nonnegative reconstruction is the projection of
  v_c onto the non-orthogonal cone spanned by {v_p, d_comp}, i.e. ½(v_p+v_c),
  residual ½(v_c−v_p), loss ½ — for the *two-atom absorbed dictionary*, at
  any κ.

## 4. Pure-strategy crossover: ε\*_TopK = 2q (no λ), two atoms

E[L_faithful, κ=1] = q, E[L_absorbed, κ=1] = ½ε. Among the **two** pure
per-pair strategies, absorbed beats faithful iff **ε < ε\*_TopK = 2q** — no
shrinkage coefficient. In L1, absorption is bought by the λ/2 soft-threshold
tax and ε\*_L1 ≈ 1.17·λ·q → 0 as λ → 0 (shrinkage-driven); under TopK, at a
one-slot budget with a two-atom dictionary, it is bought by the budget alone
(capacity-driven). This is a statement about mechanism in the capacity-limited
regime, not a performance race across regimes.

## 5. The two-atom continuous optimum tilts (ε\* organizes)

Over all unit-norm **two-atom** dictionaries under κ=1, the global optimum is
a tilted frame whose child-side atom moves 45° (ε=0, composite) → 90° (large
ε, child), passing through 52°/58° at ε=0.1/0.2 (verified, exact grid
angles). So ε\*_TopK = 2q *organizes* the two-atom transition; a trained SAE
traces the tilt through it (the L1 analogue transitioned at 0.58–0.70·ε\*).

## 6. Scope: an overcomplete dictionary escapes the crossover (the key correction)

The crossover is a property of dictionaries with **exactly two** pair atoms.
With a **third** atom it vanishes. The dictionary {v_p, v_c, d_comp} achieves
**zero loss at κ=1 for every event and every ε** (verified, GPT-5.6's
counterexample): parent-solo → v_p, child-solo → v_c, joint → √2·d_comp, each
used one-at-a-time. So:

- **There is no ε\*_TopK crossover once the pair can claim ≥ 3 atoms.** An
  overcomplete TopK SAE can hold parent, child *and* composite atoms and use
  them one per token, recovering the child for **every ε > 0** at no
  reconstruction cost.
- The **capacity that gates TopK absorption is the pair's atom allocation**
  (can a dedicated child atom exist?), realized either by a tiny dictionary
  (m = 2) or by a per-token budget so tight the child atom can never fire.
  Dictionary width, not only per-token k, is the binding constraint — the
  original note conflated them; this is corrected.

Two further scope corrections (per GPT-5.6):
- **Faithful is not the unique zero-loss κ=2 solution.** Any two rays whose
  nonnegative cone contains v_p and v_c reconstruct all events exactly (e.g.
  −30° and 120°, verified). The κ=2 statement is *existence of zero-loss
  non-absorbed solutions*, not unique recovery of v_c.
- **"Any dictionary containing the composite is strictly worse" is false.**
  The redundant triple contains d_comp and is zero-loss. Only the *specific
  two-atom* absorbed dictionary pays ½ε. Absorption must be defined
  functionally (2b), not by the presence of a composite atom.

## 7. ε = 0 is non-identifiability, not impossibility (corrected)

At ε = 0 the child never appears alone. It is **not uniquely identified by the
reconstruction objective without additional inductive assumptions** — but it
is not information-theoretically unrecoverable "under any architecture": the
distribution still contains parent-solo (v_p) and joint (v_p+v_c), so
v_c = (v_p+v_c) − v_p is obtainable given the support structure, and the
faithful dictionary remains *a* global optimum at ε = 0. The wall is a
non-uniqueness of the reconstruction objective (matching the project's
Theorem 1 framing), not an impossibility claim.

## 8. The conjecture this motivated — and why it did NOT survive testing

Put §4 and §6 together against the L1 results and one gets a *conjecture*:

- **L1, overcomplete, small ε:** L1 absorbs (the earlier planted-pair
  experiments: φ → 45° at small ε even with m = 32). Mechanism (proposed): the
  child atom, used only on rare child-solo events, does not pay its way against
  the λ/2 tax and the "good-enough" composite.
- **TopK, overcomplete, small ε > 0:** the child atom is **free** (no tax),
  and the zero-loss κ=1 solution keeps it (§6).

This *suggested* that switching L1 → TopK might reduce absorption at fixed
capacity. **That practical prediction did not survive testing** (status
banner): round 10's registered trained-SAE crossover was inconclusive and its
capacity-collapse prediction *falsified*, with overcomplete L1 recovering the
child *more* cleanly than TopK in the isolated regime (P4's direction
inverted); round 11's background-rich comparison is unresolved (confounded
detector count, semantic claim withdrawn). So the honest status is: the
two-atom oracle crossover (§4) and the overcomplete escape (§6) are the
established content; **whether a hard budget helps or hurts absorption in
trained SAEs is an open question**, not a result of this note. The clean test
is the queued causal first-letter comparison with matched seeds and a Pareto
sweep (`results/real/SUMMARY.md`).

## 9. JumpReLU (predicted, exploratory)

JumpReLU gates by a magnitude threshold θ (no shrinkage, no count budget). A
child atom survives only if child-solo activations clear θ; when child-solo
events are weak/rare the atom is pruned, favouring absorption. We predict
JumpReLU sits between L1 and TopK — a threshold-gated absorption that shrinks
as θ falls. Exploratory, no confirmatory bar.

## 10. What the SGD experiment tests

Trained (not oracle) SAEs, isolated pair, with a **functional** child-recovery
metric (a latent positively aligned with v_c that fires on child-solo events
and is child-selective; frozen thresholds in the prereg). Geometric φ is a
secondary diagnostic only.

- **P1 (two-atom crossover, m = 2, k = 1):** child-recovery rate rises with ε
  on the **2q scale**, and its midpoint moves right with q (q-scaling). This
  is the *capacity-limited* theorem test.
- **P2 (two-atom capacity collapse, m = 2):** where k = 1 absorbs, k = 2
  recovers the child — a second slot with a two-atom dictionary. Vacuous cells
  (k = 1 already recovers) *block* confirmation, they do not pass.
- **P3 (overcomplete escape, m = 16 TopK, SGD behaviour):** does SGD find the
  zero-loss child-recovering solution? Predict recovery for ε > 0 down to a
  small ε_min — escaping the two-atom crossover. Framed as an SGD-behaviour
  test, **not** as confirmation of the two-atom global-optimum theory.
- **P4 (TopK resists L1 absorption, descriptive head-to-head):** at matched
  small ε, overcomplete TopK recovers the child more often than overcomplete
  L1. The practical headline; descriptive (the isolated-setup L1 behaviour is
  not pre-certain), reported with the directional prediction.

Registered encoder/gate diagnostics (per GPT-5.6): pair-atom TopK-slot
occupancy; conditional firing on joint / parent-solo / child-solo; achieved
reconstruction vs oracle NNLS for the learned decoder. These let a failed cell
be diagnosed as SGD-reachability vs encoder/gating mismatch.

## 11. Scope and falsifiers

Model: 2D orthonormal toy, oracle two-atom propositions + SGD experiment.
Falsifiers: (i) the m = 2, k = 1 transition scale is independent of q (breaks
ε\* = 2q); (ii) m = 2, k = 2 fails to recover where k = 1 absorbs (breaks the
two-atom collapse at SGD level); (iii) overcomplete TopK absorbs at
moderate ε where the zero-loss child-recovering solution exists (would show
SGD does not reach it). ε = 0 is excluded from recovery claims (non-uniqueness
wall). The 2q law is explicitly scoped to two pair atoms; nothing here claims
it governs an overcomplete SAE.
