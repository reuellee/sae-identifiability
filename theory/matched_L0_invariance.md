# Matched-L0 comparison of L1 vs TopK absorption: support quantization, a conditional agreement theorem, and the round-13b advance predictions

> **STATUS / BLINDING BANNER (2026-07-25).** Written and verified **before
> round-13b unblinding**: the author (this note) has read no file, log, or GCS
> object containing 13b results. The §8 predictions are advance predictions to
> be committed before 13b is scored against them. All quantitative claims are
> machine-checked by `theory/verify_matched_L0.py` (35 checks, all passing,
> exit 0). Scope: the 2D orthonormal toy, oracle (population-objective)
> statements about named candidate dictionaries plus grid scans — **not**
> theorems about trained SAEs; see §9.

## 1. The question

The toy theory gives the two architectures seemingly different absorption
mechanisms and thresholds: vanilla L1 absorbs iff
ε < ε\*_L1(λ,q) = λq(8−4√2−λ)/(2(1−(2−√2)λ)) ≈ 1.17λq (shrinkage-driven,
→0 as λ→0; `PAPER.md` §4), while capacity-limited two-atom TopK at k=1
absorbs iff ε < ε\*_TopK = 2q (capacity-driven, no λ;
`theory/topk_absorption.md` §4). Yet rounds 12/13a found **no** L1-vs-TopK
absorption difference on Pythia-1.4B at matched L0=32 (paired diff +0.0030
CI [−0.0010,+0.0067]; family endpoint −0.0012 CI [−0.0081,+0.0049]). Do the
two mechanisms actually predict different *behaviour* once the comparison is
L0-matched, as the real experiments are? This note answers that inside the
toy, with proofs for the pure-candidate comparisons and scans for the rest.

## 2. Setup (recalled)

2D span of orthonormal v_p, v_c. Events: **joint** x=v_p+v_c (prob q),
**parent-solo** x=v_p (p), **child-solo** x=v_c (ε); other tokens contribute
zero vectors (zero loss, zero L0). Candidate two-atom dictionaries:
faithful F = {v_p, v_c}; absorbed A = {v_p, d_comp}, d_comp=(v_p+v_c)/√2.
L1 SAE: per-event min_{f≥0} ‖x−Df‖² + λ‖f‖₁ (oracle code = exact argmin).
TopK: oracle κ-sparse nonnegative coding (an oracle lower bound on a learned
encoder, exactly as in `topk_absorption.md` §2a). **Achieved L0** of a
configuration := expected number of strictly positive coefficients at the
per-event optimum (ties broken toward the smaller support).

## 3. Achieved L0 of the L1 SAE: support quantization (Proposition 1)

**Proposition 1.** For 0 < λ < √2, the optimal L1 codes have the following
supports and losses (verified: checks A1–A3):

| event | F: loss | F: #active | A: loss | A: #active |
|---|---|---|---|---|
| joint (q) | 2λ−λ²/2 | **2** (v_p, v_c) | √2λ−λ²/4 | **1** (d_comp) |
| parent-solo (p) | λ−λ²/4 | 1 (v_p) | λ−λ²/4 | 1 (v_p) |
| child-solo (ε) | λ−λ²/4 | 1 (v_c) | ½+√2λ/2−λ²/4 | 1 (d_comp) |

Hence **E[L0]_F = 2q+p+ε and E[L0]_A = q+p+ε, for every λ ∈ (0,√2)**:

- **The achieved-L0 gap is exactly q, and it is λ-independent.** The two
  configs differ in active count only on joint events (2 vs 1).
- **λ is not an L0 dial in this toy.** Achieved L0 is *support-quantized*:
  within a fixed configuration it does not vary with λ at all. (This is an
  artifact of unit-magnitude features; in real activations, magnitude spread
  makes E[L0] vary continuously with λ. What the toy isolates is the
  *support* part of the L0 budget, which is the part absorption trades in.)

*Proof.* F is orthonormal, so the code separates per coordinate with
soft-threshold f_i = (⟨d_i,x⟩−λ/2)₊ (`general_no_go.md` Lemma 2): on joint
both coordinates give 1−λ/2 > 0 (λ<2). For A on joint, the two-active KKT
solution has f_{v_p} = λ(1/√2−1) < 0 — infeasible — so the optimum is the
best single atom, d_comp (f=√2−λ/2), beating v_p alone since
√2λ−λ²/4 < 1+λ−λ²/4 always; similarly on child-solo the two-active solution
is infeasible and d_comp alone fires (f=1/√2−λ/2 > 0 needs **λ<√2**, the
validity boundary). Losses are the known closed forms
(`verify_absorption_theory.py`). ∎

For TopK the corresponding facts (checks C1–C4): at κ=1 **both** configs
achieve L0 = q+p+ε (one atom per active event), with the crossover at
ε = 2q; at κ=2, F is zero-loss with supports (2,1,1), achieved
L0 = 2q+p+ε, and beats A for every ε>0.

So the toy has exactly **two L0 operating points**, the same two for both
architectures: **low** (q+p+ε; one active latent per event — TopK κ=1, or L1
in the absorbed configuration) and **high** (2q+p+ε — TopK κ=2 faithful, or
L1 in the faithful configuration).

## 4. What "matched L0" means here — two inequivalent definitions

Because L1's achieved L0 is support-quantized, "choose λ so that L1's
achieved E[L0] equals TopK's" does **not** pin down λ; it pins down which
*operating point* (support structure) L1 must occupy. Two framings, which we
treat separately because they give different answers:

- **(M1) Fixed-point (emergent) matching** — the one that models round 13b.
  L1 minimizes its own objective freely at penalty λ; whatever achieved
  E[L0] results, TopK is given the budget κ ∈ {1,2} whose achieved L0 equals
  it. (In 13b, L0=32 is matched at the whole-SAE level by tuning λ/k; the
  *per-pair* active count is emergent for both architectures — the toy
  analogue is M1.)
- **(M2) Forced-budget matching.** Both architectures are externally held at
  the low operating point E[L0] = q+p+ε and compared there.

## 5. The thresholds are ordered, and λ sweeps L1 through TopK's (Proposition 2)

**Proposition 2** (checks B1–B5). ε\*_L1(λ,q)/q is continuous and strictly
increasing on (0, λ_c], where

**λ_c = 8 − 4√2 − √(92 − 64√2) ≈ 1.12235 (< √2)**

is the unique root of ε\*_L1(λ,q) = 2q in the validity window, i.e. the
smaller root of λ² − (16−8√2)λ + 4 = 0. Consequently:

- **ε\*_L1(λ,q) < ε\*_TopK = 2q for every λ ∈ (0, λ_c)**, with equality
  exactly at λ_c, and ε\*_L1 > 2q on (λ_c, √2) (e.g. 0.510 vs 0.4 at
  λ=1.25, q=0.2).
- **sup_{λ∈(0,λ_c]} ε\*_L1(λ,q) = 2q**: the union over admissible shrinkage
  strengths of L1's absorbed region exactly fills TopK-κ=1's absorbed
  region.
- At practically relevant shrinkage the gap is large: ε\*_L1(0.5,q) ≈
  0.652q, about a third of 2q; ε\*_L1(0.1,q) ≈ 0.12q, about 6% of 2q.
- Both thresholds are **p-independent** (parent-solo events are treated
  identically by F and A) — unlike the anti-rotation competition of
  `general_no_go.md`, where p₀/q ≈ √2 is critical. p plays no role anywhere
  in this note's comparisons.

*Proof sketch.* Numerator λ(8−4√2−λ) is increasing on (0, 4−2√2) ⊇ (0,λ_c];
denominator 2(1−(2−√2)λ) is positive and decreasing on (0, 1/(2−√2)); so the
ratio is strictly increasing there. Setting it equal to 2q gives the
quadratic; its smaller root is λ_c, and λ_c < √2 by direct evaluation
(1.122 < 1.414). Symbolic verification: checks B1–B4. ∎

## 6. Matched-L0 agreement theorem (fixed-point matching, λ ≤ λ_c)

**Theorem 1** (check D1: 559/559 grid cells, exact KKT on both sides). Fix
any q, p, ε > 0 with ε ∉ {2q, ε\*_L1(λ,q)} (excluding exact ties) and any
**λ ∈ (0, λ_c]**. Let C_L1 ∈ {F, A} be the L1-preferred candidate and match
TopK's budget κ so that its achieved L0 equals L1's. Then TopK's preferred
candidate at that budget **equals C_L1**, and both architectures sit at the
same achieved E[L0]. In particular, under fixed-point matching the two
architectures make the **same absorption decision everywhere in parameter
space**, despite their thresholds (2q vs ≈1.17λq) and mechanisms (budget vs
shrinkage) being different.

*Proof.* Case 1: ε < ε\*_L1(λ) ⇒ C_L1 = A, achieved L0 = q+p+ε ⇒ matched
budget κ=1. By Proposition 2, ε < ε\*_L1(λ) ≤ 2q, so TopK-κ=1 also prefers
A. Case 2: ε > ε\*_L1(λ) ⇒ C_L1 = F, achieved L0 = 2q+p+ε ⇒ matched budget
κ=2, where F (zero loss) beats A (ε/2) for every ε>0. ∎

Two honest remarks on what the theorem does and does not say:

- **It is not tautological, but one branch is easy.** Case 2 is easy (κ=2
  with two atoms is an unconstrained TopK; F wins trivially). The entire
  content is Case 1, i.e. the inequality ε\*_L1 ≤ 2q on (0,λ_c]
  (Proposition 2) — and the theorem *fails* exactly where that inequality
  fails (§7). So the agreement is a real property of the constants
  (8−4√2, etc.), not of the matching construction.
- **Mechanism, precisely.** Absorption changes the *support structure*
  (joint: 2 active → 1 active), and the support structure is what achieved
  L0 measures. Under fixed-point matching, whichever support decision L1
  makes, TopK is placed at the budget where the same decision is optimal —
  *provided* shrinkage never buys absorption that the budget alone would
  refuse, which is exactly λ ≤ λ_c. Shrinkage's residual role is to set
  *where* in (0, 2q] the common decision boundary sits, not *which* decision
  is made at matched L0.

## 7. The intuition does NOT survive unconditionally (Proposition 3)

The candidate mechanism to check was: "at a fixed L0 budget the comparison
may be forced identically for both architectures." **That is false as an
unconditional claim**, in two equivalent ways.

**Proposition 3** (checks E1–E4). For λ ∈ (λ_c, √2) there is a nonempty
window ε ∈ (2q, ε\*_L1(λ)) in which, among the pure candidates, L1 prefers
absorbed while TopK at the **same achieved L0** (κ=1, both at q+p+ε)
prefers faithful. Verified instance: λ=1.25, q=p=0.2, ε=0.45 — L1: 0.8943
(A) < 0.9023 (F); TopK κ=1: 0.2000 (F) < 0.2250 (A); both at E[L0]=0.85.
The window is not vacuous against degenerate rivals (A beats the best
single-atom dictionary, 0.9140, and the empty dictionary, 1.05).

Equivalently, under **forced-budget matching (M2)** at the low operating
point: TopK-κ=1 can hold the faithful dictionary and simply *drop* one
feature on joint events (paying q, keeping a clean child atom), whereas an
L1 SAE has no mechanism to withhold a positive-gain latent — the *only* L0
lever L1 owns is the dictionary itself, so it can reach the low operating
point only by absorbing (Proposition 1; check H1: for ε > 2q no λ ≤ λ_c
puts L1 there at all; check H2: for ε < 2q a suitable λ < λ_c does). A hard
budget separates *selection* from *shrinkage*; L1 cannot.

**Scope caveat (important).** Proposition 3 is a pure-strategy (F-vs-A)
statement. At the verified disagreement point, the *global* two-atom L1
optimum found by grid scan is a tilted, functionally child-recovering frame
{≈40°, 90°} with loss 0.8599 < 0.8943 — i.e. at such extreme shrinkage
(λ/2 = 0.625 per unit activation) the F-vs-A dichotomy is no longer the
operative competition. And λ > λ_c ≈ 1.12 is far outside the regime any
experiment in this project has run (machine checks of the closed forms
historically used λ ≤ 0.5). So: **matched-L0 invariance is a theorem only
for λ ≤ λ_c; its failure beyond λ_c is real but lives in an extreme,
practically irrelevant corner, and even there only at pure-strategy level.**

## 8. Overcomplete case: both architectures escape; invariance is trivial (Proposition 4)

**Proposition 4** (checks F1–F4, G1–G2; the TopK half is
`topk_absorption.md` §6, the L1 statement sharpens `PAPER.md` §4's
redundant-triple remark). Give the pair a **free third atom**. Then, at the
oracle population level, for every ε > 0:

- **TopK:** {v_p, v_c, d_comp} is zero-loss at κ=1 (known escape).
- **L1, 0 < λ < √2:** the same triple beats the two-atom absorbed dictionary
  by exactly **ε·(½ − (1−√2/2)λ) > 0** and the two-atom faithful dictionary
  by **q·((2−√2)λ − λ²/4) > 0**. A 3-atom grid scan (λ=0.1, q=p=0.2,
  ε ∈ {0.05, 0.2}) finds the global optimum at exactly {0°, 45°, 90°} — the
  triple — and it is functionally child-recovering (child-solo events served
  by the 90° atom, which is silent on parent-solo events).
- Both architectures then achieve the **same** E[L0] = q+p+ε (each event
  served by one dedicated atom), so the matched-L0 comparison is trivially
  invariant: **neither architecture's oracle objective prefers absorption
  when a dedicated child atom is free.**

Three consequences worth stating explicitly:

1. **The "child atom doesn't pay its way against the λ/2 tax" intuition
   (`topk_absorption.md` §8) is false at the oracle level in this toy**: the
   child atom saves ε·(½ − 0.29λ) > 0, because the composite is *not* good
   enough on child-solo events (cost ½ ≫ tax λ−λ²/4 for moderate λ). Any
   overcomplete L1 absorption observed in trained runs is therefore a
   *dynamics/reachability* phenomenon (e.g. a dead child atom receives no
   gradient), not a population-objective preference. This is consistent with
   round 10, where overcomplete **L1** recovered the child in 100% of
   isolated-regime runs (and TopK only 62–83% — an SGD gap in the *other*
   direction).
2. **Composite routing survives the escape, identically for both
   architectures**: at the escaped optimum, joint events fire *only* the
   composite atom — the child latent is silent on joint tokens even though
   the child feature is present and linearly readable in the reconstruction.
   A token-level absorption metric (round 12/13a family endpoint included)
   can therefore register nonzero "absorption" on joint tokens *even in the
   escaped regime*, and the toy predicts it registers **equally** for L1 and
   TopK. Nonzero absolute absorption at m=16384 (13a: 0.054) is thus
   compatible with this note; the invariance claim is about the
   *architecture contrast*, not the level.
3. The capacity that gates the escape is the **pair's atom allocation**
   (dictionary width / live latents), not the per-token budget — the §6
   correction of `topk_absorption.md`, now seen to apply to L1 as well.
   Empirical anchor: the controlled capacity experiment (`PAPER.md` §6.2)
   moved L1 from 0/96 triples at m=32 (allocation architecturally
   impossible) to 69% at m=34.

## 9. Verdict on the task's candidate mechanism

The intuition survives in a **modified, sharper form**:

- TRUE: absorption is a support-structure decision; achieved L0 differs
  between faithful and absorbed by exactly q, λ-independently
  (Proposition 1); under emergent (fixed-point) matching the architectures'
  decisions coincide everywhere — but only because ε\*_L1 ≤ 2q on
  (0, λ_c], which is a theorem about the constants, not a formality
  (Theorem 1). In the overcomplete regime both escape and invariance is
  trivial (Proposition 4).
- FALSE as stated: "a fixed L0 budget forces the comparison identically."
  Shrinkage is not merely a magnitude effect riding on a fixed support
  decision: for λ > λ_c it buys absorption the budget alone would refuse
  (Proposition 3), and under a *forced* low budget the architectures are
  structurally different — TopK can drop-on-joint, L1 can only absorb.
- The λ-shrinkage does affect *magnitudes* at fixed support (all active
  coefficients shrink by λ/2), which is the natural toy home for the
  **splitting** difference (13a P5: L1 2.61 vs TopK 1.25 latents/letter):
  architecture differences live in magnitude/allocation structure, not in
  the matched-L0 support decision. (Directional reading only; the toy has
  no splitting model — see §10 limitations.)

## 10. Advance predictions for round 13b

Mapping toy → 13b: L0=32 matched at the SAE level with per-pair active
counts emergent ⇒ fixed-point matching (M1); m sweep at fixed L0 ⇒ the
**allocation channel** (§8.3), with "live latents" (dead% recorded per SAE)
as the capacity index; λ in real training corresponds to a moderate
effective shrinkage, far below λ_c. The toy is an oracle theory; SGD may
deviate (both known deviations to date — round 10 P4, 13a P5 splitting —
were architecture-asymmetric *dynamics* effects).

> **BOXED PREDICTION P2 (primary, the interaction [L1−TopK]@m=2048 −
> [L1−TopK]@m=16384): the gap STAYS CLOSED — interaction ≈ 0, CI covering
> 0.** At matched L0 and matched capacity the toy's oracle absorption
> decisions coincide for both architectures in *every* capacity regime:
> abundant (both escape, Proposition 4) and scarce (agreement theorem,
> Theorem 1, for all λ ≤ λ_c ≈ 1.12 — the practically relevant range).
> Scarcity should raise absorption for **both** architectures, not open a
> gap between them.
>
> **BOXED PREDICTION P1 (direction): absorption rises as capacity falls,
> for both architectures — conditional on scarcity actually binding.** The
> escape requires a dedicated child atom per pair-family; falling live
> capacity removes it (toy: escape at ≥3 pair atoms, crossover at 2; real
> anchor: PAPER §6.2's m=32→34 flip). Operationalized: absorption should
> increase with falling *live* latents (m minus dead), not nominal m. If
> dead% at m=2048 remains ≈50% (spare capacity persists), the toy predicts
> a **flat** P1 — that outcome would not contradict this note.
>
> **If P2 comes out nonzero, the toy is silent on the sign.** Two
> out-of-toy channels conflict: (i) harsher effective per-pair budgets
> under a hard global TopK constraint would push TopK absorption up
> (interaction negative, since the unmatched thresholds order as
> ε\*_L1 < 2q); (ii) L1's ~2× larger splitting (13a P5) consumes live
> capacity faster, making L1 effectively more capacity-scarce at equal m
> (interaction positive). A signed outcome would falsify the matched-L0
> invariance mechanism as the operative explanation, not merely miss a
> prediction.
>
> **Exploratory rider:** the 13a pattern — architectures differing sharply
> in *splitting* but not in *absorption* — persists at m=2048 (splitting
> gap L1 > TopK at every m). From §9's reading that architecture
> differences live off the support decision; exploratory, no confirmatory
> bar.

Falsifiers of this note's mechanism: a significant 13b interaction in
either direction (kills matched-L0 invariance as the explanation of the
12/13a null); absorption *failing* to rise at m=2048 despite dead% falling
substantially below ~46% (kills the allocation-channel reading of P1).

## 11. Scope, limitations, and what was NOT established

- **Oracle, not SGD.** Every proposition is about population objectives of
  named candidate dictionaries (plus grid scans). Trained-SAE behaviour has
  already deviated from oracle predictions twice in this project (round 10
  P4; splitting). The 13b predictions are the toy's, offered with that
  record stated.
- **Pure strategies + scans, not global theorems.** Theorem 1 and
  Propositions 2–3 compare the two named candidates; the continuous global
  optimum tilts through the boundary rather than jumping
  (`verify_absorption_theory.py` check 3) and, at the λ>λ_c disagreement
  point, is a tilted child-recovering frame — found by scan, not proved.
  The 3-atom global-optimality of the triple is likewise grid-verified at
  two parameter points only.
- **Support quantization is partly a toy artifact.** Unit-magnitude
  features make achieved L0 exactly discrete; real activations give λ a
  genuine continuous L0 dial. The transferable content is that the
  faithful/absorbed choice moves L0 by one unit *per joint event of the
  pair* in both architectures, so a whole-SAE L0 match leaves the per-pair
  support decision to the same loss comparison in both — not that real λ
  cannot tune L0.
- **No multi-pair/background theorem.** 13b's SAEs share 32 slots and
  m ∈ {2048,…,16384} atoms across thousands of features; the toy treats one
  pair with a private budget/allocation. The mapping in §10 is an argument,
  not a derivation; in particular the toy cannot rule out
  architecture-dependent *allocation* of a shared budget across pairs
  (channel (i) of the sign discussion).
- **Not established:** any claim about JumpReLU/BatchTopK; any magnitude
  prediction for 13b absorption levels; a proof that the fixed-point
  agreement extends to dictionaries beyond {F, A} at every λ (the D1 grid
  covers λ ≤ 1.1, 559 cells, exact KKT, pure candidates); a splitting
  theory (the §9/§10 splitting statements are directional readings, not
  results).

## Verification

`python3 theory/verify_matched_L0.py` — 35 PASS, exit 0 (numpy 1.24.2,
sympy 1.11.1). Checks A (support quantization + closed forms), B (threshold
ordering, λ_c closed form), C (TopK regression), D (agreement theorem,
559-cell exact-KKT grid), E (disagreement window + non-vacuity + global-scan
caveat), F (overcomplete margins, symbolic), G (3-atom global scan,
child-recovery), H (low-L0 feasibility boundary at 2q).
