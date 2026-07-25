# Why L1 SAEs split features more than TopK at matched L0: tie-breaking economics and gating dynamics

*Theory note, 2026-07-25, written to explain round 13a P5 (mean split-family size
2.61 for L1 vs 1.25 for TopK at m=16384, matched L0=32, paired diff +1.36
CI [+0.94, +1.88], 22/24 letters — `results/real/SUMMARY_round13a.md`) and
committed **before** round 13b unblinding (13b P3 prediction boxed in §8).
Every quantitative claim is numerically verified in
`theory/verify_splitting.py` (all checks pass). Propositions 1–3 are exact
oracle statements with proofs; Proposition 4 is an analytic statement about
the training objective's local geometry, with the usual oracle-vs-SGD caveats
made explicit in §7.*

## 0. Summary of the answer

At **matched per-token L0**, the reconstruction objective's preference for
splitting is *not* larger for L1 — it is strictly **smaller** (Prop. 1, the
naive "shrinkage relief" story is backwards). Magnitude-only splitting
(same-direction atoms with different thresholds) is *exactly* loss-neutral for
L1 and cannot drive anything (Prop. 2). What survives analysis is:

- **(a′) An O(λ) opportunity-cost tilt (oracle-level, modest).** When atoms
  are scarce, L1 ranks "refine an already-represented feature into sub-feature
  atoms" *higher relative to* "dedicate the atom to a new feature" than TopK
  does, because the λ/2 shrinkage tax partially cancels in the refinement
  *differential* (relative tax ≈ λ/(1+cos φ) ≈ λ/2) but hits a fresh feature's
  gain in full (relative tax ≈ λ/r). Exact disagreement band in Prop. 3.
- **(b) A qualitative gating-dynamics asymmetry (the primary driver).** L1's
  ReLU gate is **self-gated**: a nascent sub-feature latent fires whenever its
  *own* pre-activation is positive, so it receives learning signal and grows
  whenever the harvestable residual exceeds λ/2 — which the shared-atom
  configuration always leaves behind (residual norm √(sin²φ + λ²/4) > λ/2 for
  every substructure angle φ > 0, Prop. 4i). TopK's gate is **rank-gated**: a
  latent below the top-k cutoff has *exactly zero* gradient and zero effect on
  the loss under any infinitesimal parameter change, so the merged
  (one-shared-atom, spare-dead) configuration is a genuine local minimum of
  the TopK training objective — even though the split is *globally better for
  TopK than for L1* (Prop. 1 + 4ii). TopK's non-splitting is a reachability
  failure, not a preference.

So the sharp statement is: **the split/merge decision is a near-tie broken in
opposite directions by economics (slightly pro-TopK) and by gating dynamics
(strongly pro-L1); the dynamics win.** This is consistent with round 10
(overcomplete TopK failing to reach zero-loss child-recovering solutions that
provably exist — `theory/topk_absorption.md` §6, §8), with the round-11/12
pair-detector recall gap (0.812 L1 vs 0.333 TopK), and with round 12's shared
resampling recipe (§6), while leaving the absorption margin untouched (§7),
matching 13a's arch-invariant absorption.

## 1. Model

Ambient R^d with orthonormal background as in `theory/general_no_go.md`
(Lemma 3 there lets us work in the relevant span; background events project to
0 on the in-plane atoms and identically elsewhere). A **feature with
substructure** is a pair of unit sub-directions v₁, v₂ with angle 2φ between
them (⟨v₁,v₂⟩ = cos 2φ), firing **disjointly**: events x = r₁v₁ w.p. p₁,
x = r₂v₂ w.p. p₂. This matches first-letter features ("A-words" = many
distinct tokens, each firing one sub-direction). Symmetric case: r₁ = r₂ = 1,
p₁ = p₂ = P/2. Optionally a **fresh feature** v₃ ⊥ span(v₁,v₂): x = r·v₃
w.p. p₃.

Architectures, per-event, oracle codes on unit-norm atoms:
- **L1**: min_{f≥0} ‖x − Df‖² + λ‖f‖₁. Single active atom d with projection
  c = ⟨d,x⟩ gives value ‖x‖² − (c − λ/2)₊² — the same gain
  G_L1(c) = (c − λ/2)₊² that powers the project's closed form
  L₀ = E‖x‖² − Σᵢ E(⟨dᵢ,x⟩ − λ/2)₊² (general_no_go.md Lemma 2).
- **TopK** (κ = 1 per family event): gain G_K(c) = c², no shrinkage
  (`theory/topk_absorption.md` §2a oracle).

**Configurations compared** (the split/merge decision for one feature):
- **Shared**: one atom u for the family; symmetric optimum is the bisector,
  projection c₀ = cos φ on each sub-event.
- **Split**: dedicated atoms {v₁, v₂}, projection rᵢ on its own sub-event.

**Lemma S1 (matched per-token L0 — the check the task demanded).** In *both*
configurations and *both* architectures each family event activates exactly
one latent, so the split/merge decision does not change per-token L0.
*Proof.* Shared: one atom exists. Split, TopK: κ = 1 by budget. Split, L1: in
the both-active KKT solve on event v₁ with Gram [[1,c],[c,1]], c = cos 2φ,
the second coefficient is f₂ = −λ/(2(1+c)) < 0, violating f ≥ 0; the active
set is {1} alone. ∎ (Verified: check B2.) The comparison below is therefore a
genuine matched-L0 tie-break, mirroring the empirical design (λ = 4.5 chosen
to match L0 = 32 to TopK's k = 32).

## 2. Proposition 1 — the naive "shrinkage relief" story is backwards

**Prop. 1 (symmetric case).** With a free spare atom, per unit of family
probability P (r₁ = r₂ = 1, all projections ≥ λ/2):

  Δ_L1 = G_L1(1) − G_L1(c₀) = (1 − c₀)(1 + c₀ − λ)
  Δ_K  = G_K(1) − G_K(c₀)  = (1 − c₀)(1 + c₀)
  **Δ_K − Δ_L1 = λ(1 − c₀) > 0  for every φ > 0.**

Both architectures strictly prefer the split for any substructure (φ > 0) —
and **TopK's preference is strictly larger**. The λ‖f‖₁ term actually *favors
merging*: the split's coefficient (1 − λ/2) exceeds the shared coefficient
(c₀ − λ/2), so the split pays *more* L1 penalty; the penalized gain difference
is what Δ_L1 records. *Proof:* expand; G_K(c) − G_L1(c) = λc − λ²/4 is
increasing in c. ∎ (Verified symbolically and by active-set brute force,
checks A1, B1.)

**Prop. 1′ (general version).** For arbitrary sub-directions, magnitudes rᵢ
and probabilities pᵢ (all-firing regime: rᵢ ≥ λ/2 and cᵢ(u*_K) ≥ λ/2, where
u*_K is TopK's optimal shared atom and cᵢ(u) = rᵢ⟨v̂ᵢ, u⟩):

  **Δ_K − Δ_L1 ≥ λ Σᵢ pᵢ (rᵢ − cᵢ(u*_K)) ≥ 0.**

*Proof.* Write V_arch(A) = Σ pᵢ G_arch(cᵢ(A)). The split values use the same
atoms for both architectures, so V_K(split) − V_L1(split) = λΣpᵢrᵢ − (λ²/4)Σpᵢ.
Since V_L1(u*_L1) ≥ V_L1(u*_K),
Δ_K − Δ_L1 = [V_K(split) − V_K(u*_K)] − [V_L1(split) − V_L1(u*_L1)]
≥ [V_K(split) − V_L1(split)] − [V_K(u*_K) − V_L1(u*_K)]
= λ Σ pᵢ (rᵢ − cᵢ(u*_K)) ≥ 0, using cᵢ(u) ≤ rᵢ (unit atoms). ∎
(Verified on random asymmetric instances, check C.)

**Consequence.** Candidate mechanism (a) as commonly told — "shrinkage makes
splitting more attractive to L1" — is **false at matched per-token L0 with a
free atom**. If the objective-with-free-capacity were the whole story, *TopK*
would split at least as much as L1. The observed 2× excess must come from
somewhere else.

## 3. Proposition 2 — pure magnitude splitting is exactly dead (mechanism c)

**Prop. 2.** For data x = r·v (r ≥ 0 random, single direction): (i) any
dictionary of N atoms all equal to v achieves exactly the same optimal L1
objective as one atom — the objective depends on the code only through
s = Σfⱼ (reconstruction ‖x − s v‖², penalty λs) — so duplicate-direction
splitting is a **loss-flat manifold**, never a strict improvement; (ii) the
oracle code s*(r) = (r − λ/2)₊ is *exactly realizable by a single ReLU
latent* f = ReLU(⟨v, x⟩ − λ/2), so there is no encoder-expressivity reason to
split by magnitude band either; (iii) under TopK, duplicates are weakly
harmful (they waste budget slots when k binds elsewhere on the token). ∎
(Verified, check D.)

So "high-magnitude vs low-magnitude latent pairs at the same direction" is
not a mechanism for either architecture. Splitting requires *angular*
substructure (φ > 0) — consistent with 13a's split families being distinct
token-cluster latents. Two side effects worth recording: the flat manifold
means L1 does not *oppose* redundant duplicates (SGD noise can leave them),
and at φ = 0 the nucleation gradient of Prop. 4(i) vanishes exactly — the two
results agree at the boundary.

## 4. Proposition 3 — the surviving oracle mechanism: shrinkage-tilted atom allocation (a′)

Atoms are scarce (width m, and empirically ~50% dead at m = 16384, so "live
capacity" is the binding resource). The real decision is not "split vs don't"
but "spend the marginal atom on splitting feature A vs covering fresh feature
B". Gains per event: split of A (weight P, symmetric, sub-magnitude 1):
S_L1 = (1−c₀)(1+c₀−λ), S_K = (1−c₀)(1+c₀). Cover B (weight p₃, projection r):
F_L1 = (r − λ/2)², F_K = r².

**Prop. 3.** For r > λ/2, there exists a nonempty band of p₃ in which **L1
prefers to split while TopK prefers to cover** iff S_L1·F_K > S_K·F_L1, i.e.

  **H(r, c₀, λ) := λ[ (1+c₀)r − r² − (1+c₀)λ/4 ] > 0,**

i.e. iff r < r₊ = [(1+c₀) + √((1+c₀)² − (1+c₀)λ)]/2 = (1+c₀) − λ/4 + O(λ²).
The band is then p₃·F_K ∈ (P·S_K, P·S_L1·F_K/F_L1). The **reverse** band
(TopK splits while L1 covers) requires r > r₊ ≈ 1 + c₀ — the fresh feature
must project at ≈ 2× the sub-feature scale (≈ 4× the energy). For r ≤ λ/2 the
fresh feature is below L1's firing threshold and L1 trivially prefers the
split. *Proof:* clear the (positive) denominators and expand;
S_L1 F_K − S_K F_L1 = (1−c₀)·H. ∎ (Verified symbolically and by direct
two-configuration loss comparison in the 3D model, checks A2, E.)

**Reading.** The λ/2 tax cancels to leading order in the refinement
*differential* (relative tax λ/(1+c₀) → λ/2 as φ → 0) but hits fresh coverage
in full (relative tax 1 − (1−λ/(2r))² ≈ λ/r): a factor-2 tax asymmetry at
r = 1. Under scarcity both architectures fund uses in order of value; L1's
value ordering promotes refinements. This mechanism **survives**, but it is an
O(λ/2r) *re-ranking* — with round-12's λ = 4.5 against typical activation
projections it plausibly shifts marginal allocations by ~5–20%, not obviously
a 2× family-size effect on its own. It does, however, predict the right
*sign*, and it operates even at the oracle/global-optimum level.

## 5. Proposition 4 — the primary driver: self-gated vs rank-gated nucleation (b)

Work at the **merged** configuration: shared atom u (bisector) live, plus a
spare latent (dead or freshly resampled).

**(i) L1: merged is first-order escapable for every φ > 0.** On a v₂-event
the shared atom leaves residual ρ = v₂ − (c₀ − λ/2)u with

  **‖ρ‖² = sin²φ + λ²/4 > (λ/2)²  for every φ > 0**

(the shrinkage part λ²/4 and the misalignment part sin²φ are orthogonal —
verified symbolically, check A3). Give the spare decoder ρ̂ = ρ/‖ρ‖ and an
encoder that fires t ≥ 0 on v₂-events only (affinely separable since
v₁ ≠ v₂). The population objective along this feasible path has

  dL/dt |_{t=0⁺} = p₂ (λ − 2‖ρ‖) < 0  ⟺  ‖ρ‖ > λ/2  ⟺  φ > 0.

So whenever any spare latent exists, the merged configuration is **not a
local minimum** of the L1 population objective: there is strict first-order
descent toward the split, for arbitrarily small substructure. (At φ = 0,
‖ρ‖ = λ/2 exactly and the derivative is 0 — no duplicate nucleation,
consistent with Prop. 2.) The full re-optimized two-atom loss is strictly
below the one-atom loss for all tested φ, λ (check F). Note the shrinkage
residual *helps* the nascent latent clear its own threshold — this is the
kernel of truth in "shrinkage relief": shrinkage does not make the split
*endpoint* more valuable (Prop. 1 says less), but it fattens the residual
signal that lets SGD *find* the split.

**(ii) TopK: merged-plus-dead-spare is a genuine local minimum.** Let the
TopK encoder be zⱼ = wⱼ·x + bⱼ with top-k selection by z and code
fⱼ = ReLU(zⱼ)·1[j ∈ top-k]. Suppose the spare j satisfies zⱼ(x) < z_(k)(x) − δ
(strictly below the cutoff) on every support event. Then fⱼ ≡ 0 on an open
neighborhood of the spare's parameters (wⱼ, bⱼ, dⱼ): the loss is **locally
constant** in that entire parameter block — zero gradient, zero effect, for
*any* infinitesimal move. Since the bisector is the optimum of the
one-live-atom problem, the merged configuration is a local minimum of the
TopK training objective, **despite the split being globally better for TopK
than for L1 by Prop. 1**. Escape requires an O(1) move in encoder space (the
spare must out-rank a trained incumbent). ∎ (Verified: exact loss invariance
under spare-parameter perturbations, and the global gap, check G.)

**(iii) Both attractors exist for TopK; only one for L1.** Once a split *is*
reached, TopK holds it stably — with encoder rows aligned to their atoms,
each sub-atom wins the ranking on its own sub-event (pre-activation 1 vs
cos 2φ; check G3). So TopK has (at least) two attractors, merged (family
size 1) and split (family size 2), while L1 destabilizes merged. The
empirical family sizes — TopK 1.25 ≈ mostly-merged, L1 2.61 ≈ mostly-split —
are basin statistics consistent with exactly this picture.

**(iv) The round-12 recipe makes the asymmetry concrete.** Both arches used
identical Anthropic-style dead-latent resampling (`experiments/
real_train_sae.py`): dead latents are reseeded to *residual directions* with
encoder scale 0.2. Under L1 the reseeded latent **self-gates**: it fires
immediately wherever its own pre-activation is positive, receives gradient,
and by (i) grows whenever the residual it was seeded on clears λ/2 — which
split residuals do. Under TopK the reseeded latent fires only if scale-0.2
pre-activations beat trained O(1) incumbents in the top-32 — generically they
do not, and it re-dies (dead% stayed high for both arches: ~42–53% TopK,
~52–63% L1, `results/real/stats_summary.txt`). Same seeding, asymmetric
survival: the nucleation channel that the theory identifies is the one the
training recipe actually exercises. Note the direction of the dead% gap: L1
keeps *fewer* latents alive overall yet splits *more* — consistent with the
allocation story (L1 spends live latents on refinements of strong features
rather than on covering additional tail features), and inconsistent with a
crude "L1 simply has more live latents to spend" account.

## 6. Verdicts on the candidate mechanisms

| mechanism | verdict | where |
|---|---|---|
| (a) shrinkage relief: L1's objective rewards splitting more at matched L0 | **DEAD — sign is backwards** (Δ_K − Δ_L1 = λP(1−c₀) > 0) | Prop. 1, 1′ |
| (c) magnitude splitting (same direction, different thresholds) | **DEAD — exactly loss-flat for L1; weakly harmful for TopK** | Prop. 2 |
| (a′) opportunity-cost tilt: λ-tax cancels in refinement differentials, so scarce atoms go to splits under L1 | **SURVIVES** (exact band H > 0 iff r < 1+c₀−λ/4), oracle-level, O(λ) magnitude — right sign, likely too small alone | Prop. 3 |
| (b) gating dynamics: self-gated (L1) nucleation vs rank-gated (TopK) dead-spare trap | **SURVIVES — primary**; L1: first-order escape for all φ > 0; TopK: merged is a local min despite worse global loss | Prop. 4 |

Why absorption stays arch-invariant while splitting differs: both surviving
mechanisms act on the *refinement of already-represented* features (they need
a live shared atom leaving a ‖ρ‖ > λ/2 residual). The absorption margin — is
any selective latent funded for the rare solo signal at all — is a different
decision, whose crossovers (ε*_L1 ≈ 1.17λq; two-atom ε*_TopK = 2q, escaped
when a third atom is free) were already analyzed elsewhere and were not
distinguishable at m = 16384's spare-capacity regime (round 12/13a null). We
do not claim the mechanisms here predict that null; we note they do not
contradict it.

## 7. Scope, limitations, honesty

- **Oracle vs SGD.** Props. 1–3 are exact statements about oracle codes and
  global/optimal configurations. Prop. 4 is about the *population objective's
  local geometry* at specific configurations — first-order escapability vs
  local minimality — which constrains gradient training but is not a proof of
  SGD basin statistics (cf. the round-5 lesson: equilibrium theory translated
  only quantitatively at moderate β). The claim "TopK sits at merged, L1 at
  split" is the theory-consistent reading of 13a, not derived convergence.
- **Toy scope.** Disjoint sub-events, unit norms, orthonormal ambient
  background, one feature's decision analyzed in isolation (plus one fresh
  competitor in Prop. 3). Real first-letter families have >2 sub-clusters,
  overlapping supports, and correlated backgrounds. The transient nucleation
  state can co-fire (family L0 = 2 on a sub-event) before encoders separate;
  the matched-L0 statement (Lemma S1) is about the endpoint configurations.
- **All-firing regime.** Prop. 1′/3 assume projections ≥ λ/2; below it, L1
  truncates (and trivially prefers funded refinements over unfundable fresh
  features — same direction of asymmetry).
- **L1 hard-dead latents.** An L1 latent with a negative-margin bias that
  never fires also has exactly zero gradient — L1 is not immune to dead
  latents (dead% ≈ 52–63% in round 12). The asymmetry of Prop. 4 is about
  *which states are trapped*: L1's firing set is under the latent's own
  parameters (generic inits and the b = 0, scale-0.2 resampled state fire on
  a positive-measure event set, hence receive gradient and grow by 4(i));
  TopK's rank gate zeroes every below-cutoff latent *regardless of its own
  parameters*, so the same reseeded state is flat. First-order escapability
  under L1 holds from any firing state, not from every state.
- **Prop. 3 compares the two designated configurations** (split vs cover);
  we did not certify that no third two-atom configuration (e.g. a blend atom
  mixing the pair plane with v₃) beats both inside the band. The band is a
  statement about the marginal-allocation decision, not a global-dictionary
  optimum.
- **TopK trap vs revival tricks.** Prop. 4(ii) is about below-cutoff latents
  under pure gradient training; aux-k losses or aggressive re-ranking could
  in principle break the trap. Round 12's recipe (residual resampling without
  an aux-k gradient) does not, per §5(iv). Different TopK training stacks
  could shrink the empirical gap — that is a prediction, not a threat.
- We did **not** find a way to make mechanism (a) or (c) work; if 13b shows
  L1's excess *growing* under scarcity, mechanism (a′)+(b) as stated would be
  in trouble (see the box).

## 8. Prediction for round 13b P3 (committed before unblinding)

Mechanism logic: L1's excess splitting is nucleation into **spare capacity**
(dead or marginal latents; ~50% dead at m = 16384). At fixed L0 = 32,
shrinking m raises the opportunity cost of an atom (Prop. 3's bar P·S must
now beat better-funded fresh features) and shrinks the spare pool that
Prop. 4(i) nucleates into; TopK is already near its merged floor.

> **BOXED PREDICTION (13b P3, mean family size |F_L| by width, both arches).**
> 1. **(High confidence)** Families shrink as m falls for **both**
>    architectures (agreeing with the prereg expectation), and the **sign of
>    the gap persists at every width**: L1 ≥ TopK at m = 2048, 4096, 16384,
>    in a clear majority of letters.
> 2. **(Medium confidence)** The paired L1−TopK gap **shrinks monotonically
>    as m falls** — from +1.36 at m = 16384 to a clearly smaller value at
>    m = 2048. L1's 2× excess is a spare-capacity phenomenon; scarcity
>    compresses it.
> 3. **(Lower confidence, quantitative)** m = 4096: L1 ≈ 1.6–2.2, TopK ≈
>    1.05–1.35; m = 2048: L1 ≈ 1.2–1.8, TopK ≈ 1.0–1.25; gap at m = 2048 at
>    most half the m = 16384 gap (≤ +0.7).
> 4. **(Mechanistic exposure, stated now, not post-hoc)** The gap should
>    track **measured spare capacity (dead%/live-latent surplus)** more
>    tightly than m itself. If dead% remains ≳ 40% at m = 2048, expect the
>    gap to persist near +1 there — and that outcome would *support* the
>    spare-capacity mechanism while breaking prediction 3's numbers.
> **Falsifier:** if the L1 excess *grows* as m falls (gap at 2048 > gap at
> 16384) while dead% falls, mechanisms (a′)+(b) as the drivers are wrong.

## 9. Prior art and what is claimed as new

Feature splitting in L1 SAEs is a known qualitative phenomenon: Anthropic's
*Towards Monosemanticity* (Bricken et al., 2023) documented features
splitting into finer variants as dictionary size grows (and its successors
reproduced this at scale); Chanin et al. (*A is for Absorption*, 2024)
introduced the absorption-vs-splitting distinction on exactly the
first-letter task family used here; Chanin & Till's *Broken Latents* work
further documents non-atomic/split latents in trained SAEs. None of that is
claimed here. What we believe is new — stated conservatively, pending the
project's usual novelty adjudication — is the **sharp matched-L0 account of
the L1-vs-TopK asymmetry**: (1) the proof that at matched per-token L0 the
free-capacity objective preference for splitting is *stronger* for TopK
(Prop. 1/1′), so the observed asymmetry cannot be an objective-preference
story; (2) the exact O(λ) opportunity-cost band (Prop. 3); and (3) the
self-gated vs rank-gated nucleation dichotomy (Prop. 4) with its
residual-norm identity ‖ρ‖² = sin²φ + λ²/4 > (λ/2)², which converts
"TopK has dead latents" folklore into a specific local-minimum/escapability
asymmetry at the split/merge decision. If prior work already contains (1)–(3)
in this form we will withdraw the novelty claim; the explanatory value for
rounds 13a/13b stands either way.
