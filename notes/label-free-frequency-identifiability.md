# Label-free frequency identifiability under absorption: a no-go for co-firing signatures and a within-composite estimator

**Date:** 2026-07-22
**Status:** SUPERSEDED IN PART by Arm A results (same day; `results/prereg_armA/SUMMARY.md`).
P1's invariance premise **does not transfer to trained SAEs** (§2 re-opened per the
registered decision rule): trained absorption is *gated* — a parent-aligned latent plus an
encoder-gated composite — so the binarized code separates the sub-populations nearly
perfectly (cond. TV ≈ 1.0) and simple signature counting recovers ρ to ≤ 0.02. P2's
mixture estimator is FALSIFIED at σ=0 (the composite has no host-only mode to fit) and
works only in a noise window (err 0.0075 at σ=0.1). The §2 algebra stands for its stated
single-shared-latent model; that model is not what training produces, even capacity-forced.
Remaining open problem: label-free **pair identification** (the ~45° decoder geometry +
near-disjoint firing is the obvious signature to scan for).
Original pre-run status: theory + derivation in the idealized model; GPU validation
pre-registered (`notes/prereg-bimodality-estimator.md`). Follow-up to §15b (the open problem).
**Provenance:** derivation with Gemini 2.5 Pro (adversarial theory pass); identifiability
backbone and novelty check from an adversarially-verified literature sweep (16+ sources,
claims verified 3-vote). Claims here are analytic in the toy model and a literature
synthesis — they are *not* empirical results.

## Summary

The validated inverse-density remedy for feature absorption requires an **oracle** for each
concept's base rate. §15b refuted two label-free substitutes because novelty **magnitude**
measures distance-from-background, not class identity. This note settles the natural next
idea — recover per-class frequency from the **combinatorial identity** of the code
(co-firing signatures / topic-model base rates) — and reaches a sharper, two-sided result:

1. **No-go (binarized route).** For a fully absorbed parent/child pair, the *binarized*
   co-firing signature is provably **independent of the child base rate**. Signature-counting
   cannot recover the frequency. (§2)
2. **Possibility (continuous route).** The information is not destroyed — it moves from the
   binary code to the **continuous activation magnitude** of the composite latent, which is
   **bimodal**. A one-dimensional two-component mixture fit recovers the child's relative
   frequency. This is a *different* signal from the two refuted attempts: a within-class mode
   separation, not a global novelty score, and it reads the estimate **off the very composite
   latent absorption creates** — dissolving the circularity of §15b. (§3)
3. **Why.** Both facts are the toy-model shadow of a known identifiability boundary: every
   label-free mixing-weight recovery theorem depends on one condition — linear independence /
   "sufficiently scattered" / Kruskal-rank of the per-class signature vectors — and full
   absorption is exactly the degeneracy that violates it. (§4)

To our knowledge no published method recovers label-free per-class frequency for this
target (§5).

## 1. Setup

Idealized model (as elsewhere in this project): orthonormal features; a host/parent latent
`p` and a child feature `c` with orthonormal directions `v_p ⟂ v_c`. A vanilla L1 ReLU SAE
in the absorbed regime learns a single **composite** latent with encoder/decoder direction
`w = (v_p + v_c)/√2`. Two sub-populations activate it:

- **host-only** events `x = v_p`;
- **host+child** events `x = v_p + v_c`.

Let `ρ` be the fraction of composite firings that are host+child (the quantity proportional
to the child base rate the remedy needs). The target is to recover `ρ` **label-free**.

## 2. No-go: binarized co-firing signatures are independent of ρ

Project the two event types onto the composite direction:

- host-only: `wᵀx = (v_p+v_c)/√2 · v_p = 1/√2 > 0`
- host+child: `wᵀx = (v_p+v_c)/√2 · (v_p+v_c) = 2/√2 = √2 > 0`

Both are strictly positive, so **both binarize to the identical signature** `S = {w}`. Every
other latent fires only on background and is (in the idealized model) independent of which
sub-population produced the composite firing. Hence the probability of every signature is a
function of the background and the *total* composite firing rate only:

> `P(S = {w}) = P(host-only) + P(host+child)` — the ρ-dependence cancels.

The distribution over binarized signatures is therefore **invariant to ρ**: the map
`ρ ↦ P(signature)` is constant, so `ρ` is unidentifiable from signature counts. Any
topic-model / NMF / signature-frequency estimator run on the **binarized** code inherits this
degeneracy. **The combinatorial-identity route, in its natural binary form, is a dead end
under absorption.**

## 3. Possibility: a within-composite bimodality estimator

Binarization is exactly what discards the signal. The composite latent's **continuous**
activation takes two distinct values, `1/√2` (host-only) and `√2` (host+child), so over the
firing sub-population its distribution is a two-component mixture with weights `(1−ρ)` and `ρ`:

```
act(w) | fires  ~  (1−ρ)·δ(1/√2)  +  ρ·δ(√2)          (idealized; broadened by noise in practice)
```

The mode weight **is** ρ. Estimator:

1. Collect the nonzero activations of the composite latent.
2. Fit a 2-component 1-D mixture (GMM/k-means on the scalar).
3. `ρ̂` = mass of the higher-magnitude mode; child base rate `= ρ̂ ×` (composite firing rate).

Two properties make this more than a repackaging of the refuted attempts:

- **It is not a novelty score.** §15b measured how far a point sits from the *whole
  background* (a global outlier magnitude). This is a **within-class** test: it separates
  sub-modes *inside* the set already assigned to the composite latent. The failed estimators
  could not see it because they never conditioned on the composite.
- **It dissolves the circularity.** §15b's estimators needed to undo absorption to see the
  child. Here absorption is *required*: the estimate is read directly off the composite latent
  absorption produces. Consistent with Chanin et al. (arXiv:2409.14507), the absorbed
  concept is not erased — the composite decoder is a *mixture* of host and child directions —
  so the child leaves a recoverable trace, just in the magnitude, not the bits.

A 1-D two-component mixture with separated means is classically identifiable, so the toy-model
recovery is exact; the real-data question is one of **signal-to-noise** (see §6).

## 4. Why: the identifiability backbone

The literature sweep places both results inside a single, well-understood boundary. Four
independent label-free routes recover latent-class structure from co-occurrence/count
statistics:

| Route | Recovers | Pivotal condition |
|---|---|---|
| Anchor-word / separable NMF — Arora–Ge–Moitra (FOCS 2012) | topic-word matrix `A` (not the mixing weights in general) | separability / anchor word per class |
| Anchor-free correlated topics — Fu–Huang–Sidiropoulos (NeurIPS 2016) | topics **and** topic-topic correlation from 2nd-order co-occurrence | **sufficiently scattered** `C`, `rank(P)=F` |
| Method-of-moments / tensor decomposition — Anandkumar et al. (JMLR 2014) | per-class **mixing weights** `w_i` from 2nd/3rd co-firing moments | **non-degeneracy**: `μ_i` linearly independent, `w_i>0` |
| Bernoulli-product mixtures — Allman–Matias–Rhodes (Ann. Stat. 2009) | parameters + proportions, **generically** (non-id set has measure zero) | Kruskal-rank sum `≥ 2r+2` |

The Anandkumar route is the direct instantiation of "recover base rates from co-firing
moments" — and every route reduces to the **same** pivotal quantity: linear independence /
sufficiently-scattered / Kruskal-rank of the per-class **signature vectors**. A fully absorbed
rare child has a signature that is collinear with (subsumed by) the host composite — effective
`w→0`, `μ_child` not independent of `μ_host` — which is **exactly** the degeneracy each theorem
excludes: it drives Fu–Huang–Sidiropoulos to the trivial `|det E|=0` optimum and violates
Anandkumar's Condition 4.1. So §2's no-go is not a toy accident; it is the local picture of a
general boundary. The continuous estimator (§3) escapes it by projecting to one dimension,
where a 2-component mixture is identifiable regardless of the high-dimensional collinearity.

**Honesty on the no-go's strength.** The broader claim "an absorbed child is fundamentally
unidentifiable by *any* label-free method" is **not** proven and should not be asserted.
Non-degeneracy failure is only *conjectured* hard (Mossel–Roch computational; Moitra–Valiant
information-theoretic), and there is a tensor-product / smoothed-analysis escape hatch
(Bhaskara et al. 2014; Anderson et al. 2014) that higher-order co-firing moments may exploit.
The clean, defensible no-go is the specific one in §2 — about the **binarized signature
observable** — and §3 is itself a label-free method that succeeds, so the correct headline is:
**solvable label-free, but not through the binary code.**

## 5. Novelty

No published method directly solves label-free per-class frequency for the absorbed-SAE
target:

- **UOT-RFM** (arXiv:2509.25713) emits only a per-sample continuous density ratio ("majority
  score") and explicitly sidesteps class-count recovery; it would transfer to SAE activations
  only as a per-sample reweighter, not a per-concept frequency estimator (and its own
  evaluation uses oracle proportions).
- **"A is for Absorption"** (Chanin et al., arXiv:2409.14507) detects absorption only with
  ground-truth probes + causal ablation — fully label-dependent.

The within-composite bimodality estimator and the matched no-go/possibility framing are, to
our knowledge, unclaimed. (Caveat: a null over a searched space, not a proof of absence; the
2026 marginal-independence preprint arXiv:2606.07914 is a fresh, un-refereed adjacent lead.)

## 6. Limitations

1. **SNR is the whole ballgame on real data.** The toy modes are Dirac spikes; real
   activations broaden them with background noise, encoder bias, and non-orthogonality.
   Viability = mode separation ÷ mode width. Untested.
2. **Multi-child conflation.** With several children in one composite, scalar magnitude
   encodes *aggregate* child frequency, not per-child; the multi-child case (relevant to
   matryoshka) needs more than one magnitude dimension.
3. **Two-phase pipeline**, not a one-shot fix: train to absorption → estimate ρ → reweight /
   finetune. An online variant (estimate as the composite forms) is untested.
4. **Idealized model.** Orthonormal features, tied weights, noiseless. The Bernoulli-mixture
   backbone additionally assumes conditional independence of coordinates given class, which
   superposition deliberately violates.

## 7. Predictions (pre-registered)

- **P1 (no-go).** Train SAEs to absorption at two base rates `ρ₁≠ρ₂`, all else equal. The
  total-variation distance between their **binarized** signature distributions is `≈ 0`.
- **P2 (estimator).** The **continuous** within-composite bimodality estimator recovers the
  true `ρ` (target within a pre-set CI) where P1 fails.
- **P3 (SNR boundary).** There is a noise level beyond which the modes merge and P2 fails —
  locating it is the real-world deliverable.

Full spec, metrics, thresholds, and falsifiers: `notes/prereg-bimodality-estimator.md`.

## References

- Arora, Ge, Moitra. *Learning Topic Models — Going Beyond SVD.* FOCS 2012. arXiv:1204.1956
- Fu, Huang, Sidiropoulos. *Anchor-Free Correlated Topic Modeling: Identifiability and
  Algorithm.* NeurIPS 2016.
- Anandkumar, Ge, Hsu, Kakade, Telgarsky. *Tensor Decompositions for Learning Latent Variable
  Models.* JMLR 15 (2014). arXiv:1210.7559
- Allman, Matias, Rhodes. *Identifiability of parameters in latent structure models with many
  observed variables.* Annals of Statistics 37 (2009). arXiv:0809.5032
- Chanin et al. *A is for Absorption.* arXiv:2409.14507 (NeurIPS 2025)
- UOT-RFM. arXiv:2509.25713
- Kanamori, Hirose, Yamamoto. (marginal-independence identifiability). arXiv:2606.07914 (2026, preprint)
- Mossel, Roch 2006; Moitra, Valiant 2010 (hardness); Bhaskara et al. 2014; Anderson et al. 2014 (escape hatch)
