# Gating-corrected ρ estimation for leaky absorbed pairs

*Theory note, 2026-07-23, revised same day after two external reviews
(Gemini 2.5 Pro: minor revision; GPT-5.6: major revision — both in
`reviews/`, response doc beside them). Status: development — F4 (tokenwise
dominance) is a HYPOTHESIS until D1's token-level measurements are committed;
the prereg (`notes/prereg-gating-corrected-rho.md`) locks only after D1.*

## 1. Problem

Given an **oracle-located, criterion-qualified absorbed pair** (par, comp) —
round 9 does not test detection; see prereg scope — the pipeline's counting
estimator

  ρ̂_count = P(b_lo = 1) / P(b_lo ∨ b_hi)   (lo = lower-rate latent, binarized at θ)

assumes clean gating: comp fires iff the event is joint (parent ∧ child), par
fires iff parent-solo. Trained absorbed SAEs in leaky regimes violate this:
ρ̂_count ≈ 0.75 vs true 0.5 on semi-synthetic GPT-2 (Arm 2, round-8 E1) and
0.59 vs true 0.1 in Arm A at σ = 0.1. Separately, the rarity rule itself
caps the estimand: under clean gating P(b_comp|parent) = ρ and
P(b_par|parent) = 1 − ρ, so the rarity-based estimator converges to
**min(ρ, 1 − ρ)** — above ρ = 0.5 it is not aimed at ρ at all. The clean
leak-bias comparator is therefore the **oracle-comp count**
ρ̂_C = P(b_comp)/P(b_par ∨ b_comp), which targets ρ under clean gating; the
deployed rarity-based ρ̂_count is reported as the operational incumbent,
valid as a ρ-estimator only for ρ ≤ 0.5.

## 2. Leak model

Token classes: J (joint, rate r_J over tokens), S (parent-solo, r_S),
B (background, r_B = 1 − r_J − r_S). Target: ρ = r_J / (r_J + r_S).
Binarized fires b_par, b_comp with class-conditional rates

  g₀ = P(b_par | S),  g₁ = P(b_par | J),  a₁ = P(b_comp | J),  a₀ = P(b_comp | S),

background *pattern* probabilities q_ij = P(b_par = i, b_comp = j | B)
(marginals b_p = q₁₀ + q₁₁, b_c = q₀₁ + q₁₁ — the marginals do NOT determine
the cells; no independence is assumed on B unless stated), and on co-firing
tokens the dominance quantities

  δ_J = P(act_comp ≤ act_par | J ∧ 11)   (ties counted as inversions),
  δ_S = P(act_comp > act_par | S ∧ 11),
  π_B = P(act_comp > act_par | B ∧ 11).

**Scale-comparability (S1).** Comparing raw activations across two latents is
meaningful only under a common per-latent scale: activation ×c with decoder
column ÷c leaves reconstruction unchanged, so raw ordering is not invariant
in general. In both project harnesses the decoder columns are renormalized
to unit norm at every training step (`BSAE.renorm()`; the real-data loop's
per-step `D.div_(D.norm(dim=1))`), which removes this freedom: activation =
reconstruction-contribution magnitude. The eval code asserts unit norms on
the scored pair's columns; S1 is a checked precondition, not an assumption.

**Structural facts (measured on committed development data; every absorbed
run, both harnesses, all σ):**

- **F1: a₁ ≈ 1** (E1: min 0.9978 over 48 runs; Arm A `fire|joint` = 1.000).
- **F2: g₀ ≈ 1** (E1: min 0.9977).
- **F3: the leaks are large and harness-dependent.** a₀ ∈ [0.54, 0.65] (E1),
  0.16 / 0.60 / 0.71 (Arm A σ = 0 / 0.05 / 0.1); g₁ ∈ [0.48, 0.62] (E1),
  ≈ 0 at σ = 0 (Arm A tv_cond = 0.9999). No fixed constant can absorb them.
- **F4 (HYPOTHESIS, pending D1): tokenwise dominance** — δ_J ≈ 0 and
  δ_S ≈ 0. Current evidence is *aggregate*: class-conditional mean
  activations separate by ~5–8× (J: comp ≈ 1.2 vs par ≈ 0.15; S: par ≈ 0.8
  vs comp ≈ 0.15, consistent sign in every diagnostic recompute). A mean gap
  does not preclude a minority of severe inversions, so F4 is a hypothesis
  motivated by aggregates; D1 measures δ_J, δ_S token-by-token on the frozen
  E1 weights before anything locks.

All estimators inherit the detector's binarization threshold θ = 0.05 — no
*new* harness-tuned constant is introduced, but θ itself is an inherited
absolute constant; F1–F3 are properties of this θ, and a descriptive
θ-sensitivity readout (θ ∈ {0.02, 0.10}) accompanies the confirmatory run.

**Naive bias, explained.** With F1–F2, per parent event
P(b_comp) = ρ·a₁ + (1−ρ)·a₀ → 0.8 at ρ = 0.5, a₀ ≈ 0.6, and the observed
0.75 is consistent with this once the measured background marginal excess
(≈ 6% of B-tokens firing each latent) is added to numerator and
union-denominator ([0.32 + 0.036]/[0.4 + 0.070] ≈ 0.757). The attribution is
illustrative, not identified from marginals alone — D1/round-9 diagnostics
measure the background pattern cells q_ij directly. Arm A σ = 0.1's larger
gap (0.59 observed vs 0.74 leak-only) is *consistent with* a larger
background/noise-fire denominator; alternative contributions (correlated
fires, threshold interactions, F1–F2 imperfection) are not excluded by the
CSV-level data.

## 3. Identifiability

Under exact F1–F2 the pattern-cell equations are (no independence needed for
the J/S terms — par fires on every S-token and comp on every J-token, so
P(11|S) = a₀ and P(11|J) = g₁ hold exactly regardless of within-class
dependence):

  P₁₀ = r_S(1−a₀) + r_B·q₁₀
  P₀₁ = r_J(1−g₁) + r_B·q₀₁
  P₁₁ = r_J·g₁ + r_S·a₀ + r_B·q₁₁
  P₀₀ = r_B·q₀₀

With F1–F2 only approximate (≥ 0.998), second-order corrections of size
≤ (1−a₁) + (1−g₀) enter each class term; negligible at measured values.

**Scope of the identifiability deficit.** In the background-free (or
known-q_ij) idealization, P₀₀ pins r_B, and the two free active-pattern
proportions constrain the three unknowns (ρ, a₀, g₁): binarized patterns
alone are **one scalar constraint short** of identifying ρ. With unknown
background cells the system is more severely underidentified (q₁₀, q₀₁, q₁₁
are three further unknowns). Two closure routes:

- **(C-sym) leak symmetry a₀ = g₁** — one extra *equation* in the same
  binary observation model. Empirically close on E1 (odds factor
  (1−g₁)/(1−a₀) = 1.08–1.10) but badly violated at σ = 0 (a₀ = 0.16,
  g₁ ≈ 0); regime-dependent.
- **(C-dom) tokenwise dominance (F4)** — not merely one more equation: it
  changes the observation model, importing token-level activation
  information under the S1 scale precondition and a separability
  assumption whose failure rates are exactly δ_J, δ_S.

## 4. Estimators

Throughout: "pair-active" = tokens with b_par ∨ b_comp; undefined-denominator
cases are reported as undefined, never smoothed (no pseudocounts).

**ρ̂_C (oracle-comp count; leak-bias baseline).**
ρ̂_C = P(b_comp)/P(b_par ∨ b_comp). Under clean gating (a₀ = 0, g₁ = 0,
no background) it equals ρ for all ρ — the correct baseline for measuring
leak inflation. Its leak bias per parent event is
[ρ + (1−ρ)a₀]/[1] − ρ = (1−ρ)a₀ ≥ 0 (background shifts both terms).

**ρ̂_X (exclusive-ratio; needs C-sym).** Under exact F1–F2, a 10-token can
only be S or B and a 01-token only J or B; ignoring background,

  ρ̂_X = n₀₁ / (n₀₁ + n₁₀),
  odds identity: ρ̂_X/(1−ρ̂_X) = [ρ/(1−ρ)] · (1−g₁)/(1−a₀),

exact iff a₀ = g₁, given 0 < ρ < 1 and n₀₁ + n₁₀ > 0 (at a₀ = g₁ = 1 both
exclusive cells vanish and ρ̂_X is undefined). With background,

  ρ̂_X → [r_J(1−g₁) + r_B·q₀₁] / [r_J(1−g₁) + r_S(1−a₀) + r_B(q₀₁ + q₁₀)].

E1 predicts ρ̂_X ≈ 0.52 (true 0.5); Arm A σ = 0 predicts ≈ 0.117 (true 0.1).

**ρ̂_D (dominance-partition; needs C-dom + S1; primary).** Classify every
pair-active token: 10 → S; 01 → J; 11 → J iff act_comp > act_par (ties → S,
an arbitrary rule; tie rate reported). ρ̂_D = n̂_J/(n̂_J + n̂_S).

*Error decomposition (exact F1–F2).* Background-free:

  ρ̂_D → ρ(1 − g₁δ_J) + (1−ρ)·a₀·δ_S,
  bias = (1−ρ)·a₀·δ_S − ρ·g₁·δ_J.

So: (i) **consistency is a background-free theorem**: δ_J = δ_S = 0 ⇒ exact,
regardless of leak symmetry; (ii) what matters is the *inversion rate within
each oracle class's 11-cell*, not aggregate magnitude gaps; (iii) J- and
S-errors can cancel — an unbiased ρ̂_D does not by itself prove C-dom, which
is why δ_J and δ_S are measured directly and scored separately. With
background, ρ̂_D mixes in the background assignment share

  h_B = (q₀₁ + q₁₁·π_B) / (1 − q₀₀):
  ρ̂_D → [r_J(1−g₁δ_J) + r_S·a₀·δ_S + r_B(q₀₁ + q₁₁·π_B)]
         / [r_J + r_S + r_B(1−q₀₀)],

so the operational all-token estimator is **potentially biased toward h_B**
— consistent only when background-active mass vanishes, is removed, or
balances. The confirmatory design therefore carries two endpoints: a
*mechanism* endpoint on oracle parent-event (J ∪ S) tokens, and an
*operational* endpoint on all tokens, with h_B reported per run.

Uses every parent event rather than the exclusive minority (lower variance
than ρ̂_X). Orientation: swapping (par, comp) maps ρ̂_D → 1 − ρ̂_D exactly
when the tie rate is zero (ties break exact equivariance — a tied 11-token
goes to S under either orientation). Equivariance is a structural property,
not robustness: for the directed child-given-parent estimand, 1 − ρ is still
wrong; it only helps if the output is treated as the unordered pair
{ρ̂, 1 − ρ̂} (noted for the queued orientation problem; round 9 scores under
oracle orientation, with auto-orientation descriptive).

## 5. Background contamination, per estimator

For ρ̂_X with symmetric exclusive cells (q₀₁ = q₁₀ = q — implied by
b_p = b_c regardless of independence, since q₁₀ = b_p − q₁₁ and
q₀₁ = b_c − q₁₁): writing A = r_J(1−g₁), C = r_S(1−a₀),

  ρ̂_X → (1−λ)·A/(A+C) + λ·(1/2),  λ = 2·r_B·q / (A + C + 2·r_B·q)

— a pull toward 1/2 with mixture weight λ. For ρ̂_D the pull is toward
**h_B**, which equals 1/2 only under additional exchangeability of the B∧11
activation ordering (π_B = 1/2) plus q₀₁ = q₁₀; it must be measured, not
assumed. For ρ̂_C the background limit is yet another quantity. B∧11 mass is
second-order under near-independent small marginals (q₁₁ ~ b_p·b_c ≈ 0.004
vs q ≈ 0.056 at b ≈ 0.06) but SAE latents can co-fire on background through
shared structure, so q₁₁ is measured directly (D1 + round-9 diagnostics).
Contamination scales with r_B/(r_J + r_S) and is invisible at ρ = 0.5 for
symmetric pulls — cells at ρ ≠ 0.5 are therefore mandatory, and the ρ = 0.1
cells are the designed stress test.

## 6. What would count against this note's picture

- A leaky absorbed run with a₁ or g₀ materially < 1 (breaks exclusive-cell
  class purity — both corrected estimators lose their anchor).
- δ_J or δ_S materially > 0 in D1 or the confirmatory cells (C-dom fails;
  ρ̂_D inherits the decomposition's bias terms).
- Systematic ρ̂_D bias at ρ = 0.5 — evidence of an *omitted asymmetry*
  (q₀₁ ≠ q₁₀, π_B ≠ 1/2, systematic ties, unequal inversion masses, F1/F2
  or S1 violation), i.e. a mechanism the current model does not carry, though
  not one it categorically forbids.

## 7. Relation to prior art

The counting estimator and its gating basis are §7 (Arm A). The mechanism's
prior art: Chanin et al. (arXiv:2409.14507 App. A.3) document the encoder
"hole" (parent latent ≈ parent ∧ ¬child); Feature Hedging (arXiv:2505.11756)
documents *partial* absorption — weak parent firing on joint tokens, the
leak regime here; Chanin & Till's Broken Latents proposes histogram-based
detection. A verified deep-research sweep (2026-07-23,
`notes/deep-research-2026-07-23-ortsae-and-round9-novelty.md`) found **no
existing statistical estimator correction on binarized co-activation
counts** in this line — the pattern-cell class-purity argument and the
rank-based split of the co-fire cell appear unclaimed, to our current
knowledge. That sweep's novelty finding gates the claim language, not the
experiment.
