# Finite certificates of ρ-unidentifiability under gated absorption, and the boundary where audits become possible

**Date:** 2026-07-25 (rev 2, same day — after adversarial review, Gemini 2.5 Pro,
`~/finite-certificates/reviews/GEMINI_unidentifiability_review.md`, verdict BLOCKING;
this revision resolves its central objection by **construction** — Certificate C, §3.5 —
and applies its five scoping corrections, each marked "(review)" below.)
**Status:** theory note; every numbered claim is verified sympy-exactly by
`theory/verify_unidentifiability.py` (49/49 PASS, exit 0). Nothing here is empirical.
**Scope declared up front:** all certificates are about **exact equality of
distributions** (population level, infinite data). §6 states what a finite-sample or
approximate version would additionally need.

## 0. What this settles and what it does not

`notes/label-free-frequency-identifiability.md` proved a no-go for one observable class
(binarized co-firing signatures, under the *shared-composite* idealization) and honestly
flagged the broad conjecture — "an absorbed child's rate ρ is unidentifiable from
trained-SAE observables by **any** label-free method" — as **not proven**. Arm A then
showed trained absorption is *gated* (parent latent with an encoder hole + encoder-gated
composite; `results/prereg_armA/SUMMARY.md`), under which signature counting *does*
recover ρ — but only given the oracle labeling of which latent is the composite and,
more deeply, given that the input process really is the parent/child process.

This note replaces the open conjecture with something sharper and two-sided:

1. **Finite certificates (Levels 1, 2, and 2″).** Explicit pairs of generative
   processes with different ρ whose observable distributions are *exactly identical* —
   at the level of all SAE latent activations under a fixed Arm-A-style gated encoder
   (Level 1); at the level of the input distribution itself (Level 2); and — the
   strongest, forced by the review — at the input level with **both decompositions
   support-irreducible, entrywise-nonnegative, and strictly hierarchical**
   (Certificate C, ρ = 3/4 vs 1/2). Certificate C shows the ambiguity survives the
   canonical dictionary-selection rules of the NMF/sparse-coding literature.
2. **A boundary map (§4).** For each natural anchor assumption we either extend the
   certificate to defeat it or give a small proposition restoring identifiability. The
   deliverable is not the counterexamples but the resulting theory of *when label-free
   frequency audits are possible*.

**Precise statement of what is unidentifiable (review, objection (b)).** Given the
input distribution, every *geometric* functional is trivially identifiable — including
"the frequency of atoms with a positive `v_c`-component" for any **fixed, named**
direction `v_c`. What the certificates make unidentifiable is the **code-level /
generative-role** quantity: which events count as *parent-solo* vs *joint*, hence
ρ = r_J/(r_J+r_S). In Certificates A/B the readings agree on the candidate child
direction and disagree on its event attribution; in Certificate C they disagree on
**which direction is the child at all**. So "the geometric child frequency" is
identifiable only *after* a direction has been designated — and that designation is
exactly what the data does not supply. This is the project's dict-vs-code / CDX
distinction (`notes/prereg-pair-identification.md`) in theorem form: the ill-posedness
claim is about code-level quantities, never about functionals of P(x).

The honest restatement of the broad conjecture after this note: **as posed — "ρ from
observables, no assumptions on the input process" — the question is ill-posed, and the
certificates witness this constructively, even under irreducibility and sparsity
selection rules.** Identifiability is purchased by specific side information (§4), and
the certificates say exactly which purchases suffice.

## 1. Setup and definitions

Ambient space ℝ² (ℝ³ only for P8b). Unit directions: parent `v_p = e₁`, child
`v_c = e₂`, composite `u = (v_p+v_c)/√2`. A **generative process** is a finite list of
events, each an atom `x ∈ ℝ^d` with a probability and a *generative* class label:
`S` (parent solo / host-only), `J` (joint: parent and child both active), `C`
(child solo), `U` (composite-sibling solo, no child involved), `B` (background, x = 0).
Following `theory/gating_corrected_rho.md`,

> **ρ = r_J / (r_J + r_S)** — the child-given-parent rate among parent events.

Crucially, the class labels are part of the *generative description*, not of any
observable: that is the entire point of "label-free."

Decompositions have **unit-norm directions and nonnegative magnitudes** (the semi-NMF
class SAEs occupy: codes nonnegative, dictionary arbitrary). Certificate C additionally
lives in the **strict-NMF** class (entrywise-nonnegative dictionaries). A reading is a
**strict activation hierarchy** if the child never fires without its parent (no solo
child, no mixture); we use this exact wording per the review — Certificates A/B's
alternative readings are *not* strict hierarchies (they are mixtures with a solo
composite), Certificate C's both are.

**Fixed gated SAE (Level 1), the stylized Arm A mechanism.** Two latents:

- parent latent: encoder row `w_par = v_p − 2v_c`, bias 0 — the Chanin "hole"
  (parent ∧ ¬child): on host-only, `z_par = 1`; on joint, pre-activation −1 → silent;
- composite latent: encoder `u`, bias −1 — the gate: host-only gives `1/√2 − 1 < 0` →
  silent; joint gives `√2 − 1 > 0`;
- decoders `v_p` and `u`, unit-norm (S1 scale precondition holds).

This reproduces Arm A's measured structure (fire|host_only ≈ 0, fire|joint = 1, parent
silent on joint, ~45° decoder pair). Constants are stylized, mechanism is faithful.

## 2. Level 1 certificate: identical code distributions, ρ ∈ {2/5, 1/4, 0}

Three processes (background 0 with mass 1/2 in each):

| process | events (atom @ prob, class) | ρ | child base rate |
|---|---|---|---|
| **G1** | `v_p @ 3/10 (S)`, `v_p+v_c @ 2/10 (J)` | **2/5** | 1/5 |
| **G2** | `v_p @ 3/10 (S)`, `v_p+v_c @ 1/10 (J)`, `2v_c @ 1/10 (C)` | **1/4** | 1/5 |
| **G3** | `v_p @ 3/10 (S)`, `√2·u @ 2/10 (U)` | **0** | 0 |

Pushing each through the fixed encoder gives **exactly** the same code distribution
(3 atoms): `(1,0) @ 3/10`, `(0, √2−1) @ 2/10`, `(0,0) @ 1/2`. Verified exactly.

Why: the gated encoder is many-to-one — the child-solo event `2v_c` lands on the same
affine fiber of the composite gate as the joint event (`uᵀx = √2`, parent
pre-activation ≤ 0), so G2's transfer of mass from `J` to `C` is invisible; and `√2·u`
*is* the point `v_p+v_c`, so G3's relabeling is invisible a fortiori. Note G1 vs G2
have **different input distributions** (`2v_c ≠ v_p+v_c` as points): Level 1 shows the
gated encoder *itself* destroys ρ-information that was present in x. So even granting
the auditor the full joint distribution of all latent activations (not just binarized
signatures — magnitudes included), ρ is unidentifiable without assumptions on the input
process. This strictly extends the §2 binarized no-go: the old no-go binarized away the
information; here there is nothing to binarize away.

Reconstructions `x̂ = Dz` are functions of z, hence also matched. What is *not*
matched between G1 and G2 is the residual `x − x̂` (their x-distributions differ);
matching that too is exactly what Level 2 does.

## 3. Level 2 certificate: identical input distributions, different ρ

Now the strong form: one distribution over ℝ², two exact feature decompositions. A
**decomposition** assigns to each atom an event = a set of (unit direction, magnitude)
pairs summing exactly to the atom, with generative class labels as before.

**Certificate A (minimal, 3 atoms, d=2): ρ = 2/5 vs ρ = 0.**
Atoms: `0 @ 1/2`, `v_p @ 3/10`, `v_p+v_c @ 1/5`.

- **G1 (strict hierarchy):** dict `{v_p, v_c}`; `v_p+v_c` = parent(1) + child(1),
  class J. ρ = 2/5.
- **G2 (reified composite — a mixture reading, no child):** dict `{v_p, u}`;
  `v_p+v_c` = the single atomic feature `u` firing at magnitude √2, class U. ρ = 0.

Both decompositions reconstruct every atom exactly; the induced x-distributions are
*identical by construction and verified exactly*. This is the **CDX equivalence class**
from `notes/prereg-pair-identification.md` (exclusive-correlated feature firing only
when `v_p` does not) promoted from "known detector confound" to an *exact
distributional identity*. Notably (verified): **both** of Certificate A's dictionaries
are support-irreducible — irreducibility alone cannot reject the no-child reading; only
a "child exists (ρ>0)" assumption can.

**Certificate B (4 atoms, d=2): ρ = 2/5 vs 1/4 — both readings *admit a hierarchical
generative reading*, but (review, objections (a),(b) — conceded) G2's is a mixture
(its `u` fires solo, parent and child silent), not a strict activation hierarchy, and
G2's dictionary `{v_p, v_c, u}` is support-REDUCIBLE (`{v_p, v_c}` reconstructs every
atom; verified).** Atoms: `0 @ 1/2`, `v_p @ 3/10`, `v_p+v_c @ 1/10`, `v_p+2v_c @ 1/10`.

- **G1:** dict `{v_p, v_c}`; child fires at magnitude 1 or 2, always with parent.
  Parent events 1/2, joint 1/5 → **ρ = 2/5**, child base rate 1/5, E[L0] = 7/10.
- **G2:** dict `{v_p, v_c, u}`; `v_p+v_c` is `u` solo (magnitude √2, class U);
  `v_p+2v_c` is parent + child(magnitude 2), class J. Parent events 2/5, joint 1/10 →
  **ρ = 1/4**, child base rate 1/10, E[L0] = 3/5.

The x-distributions are exactly equal and ρ and the child *base rate* both differ;
pushing Certificate B through the §1 encoder also yields identical code distributions
(Level 2 ⇒ Level 1, verified). But the review is right that a dictionary-learning
practitioner enforcing support-irreducibility would canonically reject G2 here.
Certificate B's residual role is expository (the reification move in its simplest
hierarchical-adjacent form); the load-bearing certificate is now:

### 3.5 Certificate C: support-irreducible, entrywise-nonnegative, strictly hierarchical — ρ = 3/4 vs 1/2

**The fork posed by the review, resolved as a construction.** Question: does there
exist a finite distribution with two support-irreducible nonnegative unit-norm
decompositions, both hierarchical, different ρ? Answer: **yes, in d=2 with 3 active
atoms** — but it *cannot* be done by adding a composite to `{v_p, v_c}` (any dictionary
containing both is automatically reducible once atoms lie in their cone; the review's
argument is airtight there). The construction instead uses **interleaved cones**: two
parent–child pairs, neither dictionary containing the other's, in angular order
`c₂ (0°) < p₁ (≈26.6°) < p₂ (45°) < c₁ (90°)`:

- **G1:** parent `p₁ = (2,1)/√5`, child `c₁ = (0,1)`.
- **G2:** parent `p₂ = (1,1)/√2`, child `c₂ = (1,0)`.

Atoms (background `0 @ 3/5`): `z = (2,1) @ 1/10`, `y = (1,1) @ 1/5`, `v = (3,2) @ 1/10`.

| atom | G1 reading | G2 reading |
|---|---|---|
| `z = (2,1)` | parent-solo: `√5·p₁` (S) | joint: `√2·p₂ + 1·c₂` (J) |
| `y = (1,1)` | joint: `(√5/2)·p₁ + (1/2)·c₁` (J) | parent-solo: `√2·p₂` (S) |
| `v = (3,2)` | joint: `(3√5/2)·p₁ + (1/2)·c₁` (J) | joint: `2√2·p₂ + 1·c₂` (J) |

Each atom sits on its solo-reading dictionary's parent ray and strictly inside the
other dictionary's cone. All verified exactly:

- x-distributions identical (same atoms, same probabilities, both decompositions
  reconstruct exactly);
- **both readings are strict activation hierarchies** (child never fires without its
  parent — no mixture reading anywhere);
- **both dictionaries support-irreducible** (every proper subset fails on some atom);
- **both dictionaries entrywise nonnegative** — the certificate lives even in the
  strict-NMF dictionary class the review invoked, which is *stronger* than needed for
  SAEs (semi-NMF);
- dictionary sizes equal (2 each — the minimum possible, since the atoms span ℝ²);
- **ρ = 3/4 (G1) vs 1/2 (G2)**, and the pair is **non-complementary**
  (1/2 ≠ 1 − 3/4), so this is *not* the known {ρ, 1−ρ} orientation ambiguity of
  `theory/gating_corrected_rho.md` §4 — the third atom `v` (joint in both readings)
  breaks the mirror symmetry.

Every canonical tie-breaker the review proposed **ties** on Certificate C:
irreducibility (both pass), dictionary size (equal), entrywise nonnegativity (both),
strict hierarchy (both), unit norms (both). The one criterion that does not tie is
expected L0 — and it obeys an exact **coupling identity** (verified):

> **E[L0]₁ − E[L0]₂ = (ρ₁ − ρ₂) · P(parent events)**

valid whenever every active atom is parent-active with at most one child in both
readings. So minimizing E[L0] *is* minimizing ρ over admissible readings: sparsity is a
**deterministic selection rule biased toward the smaller-ρ reading**, not an
identification principle. (This supersedes rev-1's "L0-vs-dictionary-size
disagreement" framing on Certificate B, retracted per the review since B's G2 is
reducible.) Pleasing resonance: the gating note showed rarity-based counting converges
to min(ρ, 1−ρ); sparsity-based *decomposition selection* has the same min-flavored
bias, one level down.

A last geometric fact (verified): both child directions `c₁, c₂` lie **outside the
conic hull of the data** (rays z..y), and the only 2-feature dictionary *inside* the
data cone, `{ẑ, ŷ}`, admits no strict-hierarchical reading at all (both its rays fire
solo). So a "features must lie in the data's conic hull" restriction — the conic analog
of P8b's span restriction — does not select between G1 and G2; it rejects the hierarchy
hypothesis entirely. Hierarchical readings of this data *require* out-of-cone children.

**Reading.** At Level 2 the phrase "the child's true code-level rate" has no referent
determined by the data — now even under the dictionary-selection principles that
canonically resolve NMF ambiguity. "What is ρ?" is well-posed only relative to a chosen
decomposition, i.e. relative to labels or to structural assumptions strong enough to
select one. Which assumptions suffice is §4.

## 4. The boundary map: which anchors restore identifiability

Each anchor is either **defeated** (certificate extends) or **restores
identifiability** (small proposition). All items verified in the script except where
marked "argument."

| # | Anchor (side information granted) | Verdict | Witness / proof sketch |
|---|---|---|---|
| P3 | **Full dictionary known, linearly independent** (`{v_p, v_c}`, no composite in dict) | **Identifiable** | Coefficients of each atom are the unique solution of a full-rank linear system; the event process, hence ρ, is pointwise determined. Sympy: unique solution (1,1) for the ambiguous atom. This is the oracle regime the project's remedy already lives in — the certificate explains *why* the oracle is load-bearing. |
| P4 | **Dictionary known but overcomplete** (contains `u`) | **Defeated** | `v_p+v_c` has two exact nonnegative sparse decompositions, `(1,1,0)` and `(0,0,√2)`. Knowing all directions is useless if the reified composite is among them — which is precisely what a trained absorbed SAE hands you: its decoder *contains* `u` (cos_comp ≈ 0.99 in Arm A). An SAE dictionary is the *wrong kind* of dictionary knowledge. |
| IRR | **Support-irreducibility** (canonically reject redundant dictionaries; review objection (a)) | **Defeated by Certificate C** (conceded for Certificate B) | Certificate B's G2 is reducible — the objection was valid there. Certificate C: both dictionaries irreducible, entrywise nonnegative, strictly hierarchical, ρ = 3/4 vs 1/2. Irreducibility also fails to reject Certificate A's no-child reading (both A-dicts irreducible). |
| P5 | **Minimality/sparsity of the true process** | **Defeated — precisely, not by "disagreement"** (rev-1 framing retracted per review) | On Certificate C all tie-breakers tie except E[L0], which obeys E[L0]₁−E[L0]₂ = (ρ₁−ρ₂)·P(parent): min-L0 selection ≡ min-ρ selection. Sparsity confidently returns the smaller-ρ reading whether or not it is the truth. Absorption exists *because* SGD found a sparser code; "trust sparsity" re-enacts the pathology. |
| P6 | **Nondegenerate additive observation noise** (x + ε, ε independent of event) | **Defeated** | Same atoms ⇒ same convolution with any noise kernel. Observation noise cannot separate what is already equal. |
| P7 | **Independent per-feature magnitude jitter** (analog features: each active feature's magnitude has a nondegenerate density) | **Breaks reification (proven step); full restoration OPEN** (softened per review objection (d)) | Proven: a single-feature event class lies on a 1-D ray; jittered joint events fill a 2-D patch (rank-2, verified) while jittered solo events stay rank-1 — local support dimension counts co-active features, killing every reification move in Certs A/B/C and L1. NOT proven: full dictionary/event uniqueness. Standard ICA (Comon 1994) assumes mutually independent sources; hierarchical parent/child *indicators* are dependent by construction (child support ⊆ parent support), so ICA theorems do not apply. Dictionary uniqueness under hierarchical supports with independent magnitude jitter is an **open problem**. |
| P8 | **Magnitude grid known** (all magnitudes = 1) | **Restores in d=2; d≥3 defeat is an off-span loophole** (caveat added per review objection (e)) | d=2: sympy-solving `u₁+u₂ = v_p+v_c`, `‖uᵢ‖=1` gives uniquely `{v_p, v_c}`; 1-feature reading impossible (norm √2) → ρ identified. d≥3: the cancellation pair `w₁,₂ = (1/2, 1/2, ±1/√2)` defeats the anchor **only via features outside the data span** (third coordinate ≠ 0, data span = e₁e₂-plane; verified). Restricting features to the data span — standard practice — reduces d≥3 to the d=2 case and **restores identifiability**. Practitioners would close this loophole; we mark it as such. |
| PD | **Parent direction known** | **Breaks Certificate C, not A/B** | C's readings have different parents (p₁ ≠ p₂): naming the parent selects the reading. A/B share parent `v_p` across readings, so the anchor is insufficient there. Whether some certificate survives *irreducibility + strict hierarchy + known parent* jointly is **open** (no construction, no impossibility proof). |
| — | **Labels** (probes/ablation, Chanin-style) | **Restores, trivially** | Labels are exactly a selection of the decomposition. See §5. |

**Boundary summary.** The certificates live on Dirac magnitudes. The strongest
label-free lead remains **P7: analog magnitude dispersion independent across co-active
features** — but post-review its status is: reification-killing proven, full
restoration open. Robustly defeated: irreducibility, sparsity selection, overcomplete
dictionary knowledge, observation noise. Restored: known full-rank dictionary,
unit-magnitude grid within the data span, known parent direction (against C only),
labels. The open frontier is the *conjunction* row (PD) and the P7 uniqueness question.

This retro-dicts the project's own empirical arc: the σ=0 failure of the bimodality
estimator and the σ=0.1 success (Arm A M2/M4), and absorption dissolving at σ≥0.2, are
the empirical shadow of P6 vs P7 — Arm A's σ is *activation* noise entering through
features, a (correlated, imperfect) cousin of per-feature jitter, and it is exactly the
regime where ρ became estimable. "Noise-as-remedy" (Arm A exploratory finding 2) is
P7's constructive face — with the caveat that P7's full claim is open.

## 5. Position against prior art

*(Reframed per review objection (c).)*

- **NMF / sparse-coding uniqueness — the correct home.** The Level-2 certificates are,
  formally, instances of **non-uniqueness of (semi-)nonnegative sparse factorization**:
  a known discrete P(x) with multiple exact dictionary+code factorizations. The right
  comparison points are the NMF uniqueness conditions — Laurberg et al. (2008)
  ("Theorems on positive data: on the uniqueness of NMF"), separability/anchor
  conditions (Donoho–Stodden 2003; Arora–Ge–Moitra 2012), and sufficiently-scattered
  conditions (Huang–Sidiropoulos–Swami 2014; Fu et al. 2019). Our data is deliberately
  *not* sufficiently scattered (three rays in a narrow cone), and Certificate C's
  interleaved-cone geometry is a hierarchical-flavored witness of exactly the regime
  those theorems exclude. The contribution relative to that literature is not the
  existence of NMF non-uniqueness (classical) but: (i) the **hierarchy-preserving,
  irreducibility-surviving** form (Certificate C: both factorizations strict
  hierarchies, both irreducible — the standard non-uniqueness examples are not of this
  shape); (ii) the **ρ-coupling**: the ambiguity is tied to the specific estimand
  (child-given-parent rate) that SAE absorption audits need; (iii) the **E[L0]–ρ
  coupling identity**, which converts "sparsity doesn't guarantee uniqueness" into
  "sparsity-selection is deterministically biased toward small ρ."
- **Allman–Matias–Rhodes (2009) — inherited framing, now scoped** (review objection
  (c) accepted). AMR concerns identifiability of latent-class *mixture models* from
  observed variables; it entered this project as the backbone for the **binarized
  co-firing signature** analysis (§4 of the frequency note), where the observables
  really are coordinatewise codes and classes really are latent. That inherited framing
  remains apt for **Level 1's code-distribution reading** (classes = event types,
  observables = code atoms; absorption makes class signatures collinear — the
  degeneracy AMR-type theorems exclude). It is **not** the right frame for Level 2:
  there P(x) is a finite discrete distribution, trivially identifiable *as a
  distribution*; the ambiguity is factorization-level, not estimation-level, and we no
  longer claim the Level-2 certificates "live on AMR's null set." The
  smoothed-analysis intuition (nondegenerate magnitude densities restore genericity)
  survives as intuition for P7 — with P7's uniqueness claim explicitly open.
- **CDX / dict-vs-code distinction** (`notes/prereg-pair-identification.md`,
  `notes/prereg-natfeat-adjudication.md`). The pair-identification detector's known
  equivalence class — exclusive-correlated features indistinguishable from absorbed
  pairs — is upgraded from an empirical confound to an exact distributional identity
  (Certificate A *is* CDX with matched magnitudes), and Certificate C shows the class
  survives irreducibility and hierarchy constraints with both readings bona fide. The
  dict-vs-code disclosure gets its sharp form in §0: geometric functionals of P(x)
  identifiable; code-level role assignments not.
- **UOT-RFM (arXiv:2509.25713).** Its sidestep — emitting per-sample density ratios,
  never class counts — is shown to be **forced, not modest**: on Certificates B/C the
  x-distributions are equal, so every per-sample functional agrees; no reweighting
  scheme can output different ρ's because no function of the data does.
- **Chanin et al. (arXiv:2409.14507).** Their label-dependence (probes + ablation) is
  usually read as a limitation. The certificates invert the reading: labels are one of
  the few anchors that select a decomposition (§4, last row). Label-free absorption
  *detection* can work (round 8/9 detectors, gated counting under oracle pairing);
  label-free ρ *semantics* cannot, without P3/P8(span)/PD-type structure.

## 6. Scope, and what finite-sample versions would need

- **Exactness.** These are population-level, exact-equality certificates. They are
  therefore *stronger* than any sample-complexity lower bound at their point (no
  amount of data helps), but *silent* off their point. A finite-sample/approximate
  theory would need: (i) a perturbation version — if the observable distributions are
  ε-close in TV to a certificate pair, any estimator suffers error ≳ |ρ₁−ρ₂|/2 with
  constant probability (standard two-point Le Cam argument; the certificate supplies
  the two points, TV = 0, so the bound is immediate *at* the point and degrades
  continuously near it); (ii) a quantitative version of P7 — identifiability strength
  as a function of jitter dispersion, i.e. an SNR curve, which is exactly the SNR
  question the prereg pipeline already measures empirically.
- **Trained-SAE realism.** Level 1 fixes a stylized gated encoder. Trained encoders
  are leaky (F3 in `theory/gating_corrected_rho.md`); a leaky gate is a *noisier*
  channel, and the L1 certificate can be re-run with leak parameters as long as the
  leak responds only to `wᵀx` (it does, for a 2-layer encoder): the fibers that make
  G1/G2 collapse are properties of the encoder's kernel, not of the clean gate.
  Not formalized here for arbitrary leak models.
- **Would SGD find Certificate C's geometry?** Not claimed. Certificate C answers a
  worst-case question (can canonical selection principles fail?), not a
  training-dynamics question (do they fail on the decompositions SGD visits?). The
  practice-relevant reading: absorption shows SGD *does* select non-generating
  readings when they are sparser; C shows no purely structural filter can certify it
  hasn't.
- **ρ definition.** Everything is stated for the directed child-given-parent rate
  r_J/(r_J+r_S); the certificates equally kill the child base rate (Cert B: 1/5 vs
  1/10) and the "fraction of composite firings that are host+child" variant (G3: no
  such fraction exists). Certificate C's readings even disagree on the parent's
  identity while agreeing that one feature fires in every active atom.
- **Not claimed.** No claim that real LM feature magnitudes are Dirac; the empirical
  question "how close to the certificate set does a real model sit" is open and is
  precisely measurable as mode-width/mode-separation — the project's existing SNR
  endpoint.

## 7. Verification

`theory/verify_unidentifiability.py` — sympy-exact (Rational + radical arithmetic
throughout, `simplify`-based atom equality), **49 checks, all PASS, exit 0**:
geometry + gate sanity; L1 equality of code distributions for G1/G2/G3 with
ρ = 2/5, 1/4, 0; L2 Certificates A and B (reconstruction exactness, unit norms,
x-distribution equality, ρ and base-rate inequality, L2⇒L1 pushforward); the
irreducibility block (certB-G2 reducibility **concession**, certA double
irreducibility, and Certificate C in full: exact reconstructions, x-equality, strict
hierarchy in both readings, entrywise nonnegativity, support-irreducibility of both
dictionaries via exhaustive proper-subset testing, equal dictionary sizes,
ρ = 3/4 vs 1/2, non-complementarity, the E[L0]–ρ coupling identity, and the
conic-hull remark); boundary propositions P3, P4, P5 (descriptive), P6, P7(+P7b, with
printed open-problem scope), P8 (d=2 uniqueness by symbolic solve; d≥3 off-span
defeat with printed span-restriction caveat), and the parent-direction anchor.

## References (additions in rev 2)

- Laurberg, Christensen, Plumbley, Hansen, Jensen. *Theorems on Positive Data: On the
  Uniqueness of NMF.* Computational Intelligence and Neuroscience, 2008.
- Donoho, Stodden. *When does NMF give a correct decomposition into parts?* NeurIPS 2003.
- Huang, Sidiropoulos, Swami. *NMF revisited: uniqueness and algorithm.* IEEE TSP 2014.
- Fu, Huang, Sidiropoulos, Ma. *Nonnegative matrix factorization for signal and data
  analytics: identifiability and interpretability.* IEEE SPM 2019.
- (Prior list unchanged: Arora–Ge–Moitra 2012; Anandkumar et al. 2014; AMR 2009 —
  scoped to the Level-1/binarized inheritance; Chanin et al. 2409.14507; UOT-RFM
  2509.25713; Comon 1994 — cited as inapplicable-without-independence; Bhaskara et al.
  2014.)
