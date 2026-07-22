# Oracle-dependence review: is label-free per-class frequency estimation possible?

**Date:** 2026-07-22
**Method:** Claude deep-research harness (16 sources fetched, 70 claims extracted,
25 adversarially verified with 3-vote refutation) + Gemini 2.5 Pro adversarial review
(reasoning-only, Vertex path — its web grounding is unavailable on this box).
**Status:** literature/theory review of an open thread; no new GPU runs.

## The open problem (recap)

Feature absorption is fixable by inverse-density (event-frequency) loss weighting, but
that requires an **oracle** for each concept's frequency. Two label-free substitutes
(residual novelty; background-relative novelty, §15/§15b) were refuted for the *same*
reason: novelty/residual **magnitude** measures how far a point sits from the
background, not **which rare class** it belongs to. Open question: can per-concept
frequency (or a rare-concept reweighting) be recovered label-free from SAE activations
or the residual novelty subspace?

## Verdict from the literature sweep

**No published method demonstrably solves it.** Every label-free density/importance/
novelty technique surfaced reduces to a **per-point outlyingness magnitude** — the
exact failure mode already refuted here:

- Importance-weighting / density-ratio estimation (Kimura & Hino 2024, arXiv:2403.10175):
  every variant is a per-point weight `r(x)=p_te(x)/p_tr(x)` between two *given*
  distributions; no class-identity recovery. (verified 3-0)
- Direct density-ratio outlier scoring (Hido et al., KAIS 2011): per-test-point inlier
  score; structurally needs a clean inliers-only reference set. (3-0)
- COLT "tailness" (arXiv:2306.04934): per-point neighborhood-sparsity proxy; never names
  or counts a class. (3-0)

**The project's own leading hypothesis is threatened at the root.** "Cluster the novelty
subspace, inverse-weight by cluster size" fails because cluster *sizes* are
objective-induced artifacts, not concept frequencies (Nie et al., TPAMI 2019,
arXiv:1705.05950, verified 3-0):

- small-bandwidth kernel K-means / average-association isolate dense modes (Breiman's bias);
- cut-based spectral criteria (Normalized/ratio cut) have the opposite sparsest-subset bias;
- basic K-means is biased toward **equal-cardinality** clusters (2-1) — so inverse-size
  weighting measures the objective, not the world.

Long-tail SSL methods that advertise "category-level" operation do not deliver true
per-class frequency: FASSL (arXiv:2309.04723) and COLT fall back to per-point atypicality;
Geometric Harmonization (NeurIPS 2023, arXiv:2310.17622) imposes an *assumed*-uniform
simplex prior via Sinkhorn/OT rather than measuring frequency; TASE (arXiv:2410.22883)
and MiniClustering (K-means + `1/sqrt(|cluster|)`) inherit the clustering biases above.

**Best positive lead — architectural, not estimation:** Matryoshka SAEs (ICML 2025,
arXiv:2503.17547). Nested dictionaries; smaller prefixes capture common concepts, larger
ones capture rarer/specific concepts. Label-free, empirically *reduces* absorption on
Gemma-2-2B / TinyStories. Caveat: it **sidesteps** frequency estimation; reduction is
not elimination.

**Novelty check:** the "nothing solves it" result is a *search-bounded null*, not a
proof. Newer label-free frequency estimators (e.g. UOT-RFM, arXiv:2509.25713;
unsupervised class-distribution estimation) appeared in search but were not in the
verified set — candidates for a next round.

## The pivot: magnitude → combinatorial identity

Gemini's adversarial review argued the "nothing solves it" conclusion is **too strong**
because the sweep was overly geometric and structurally missed technique classes that
reason about **combinatorial structure**, not distance:

1. **Co-occurrence / combinatorial firing statistics** of the SAE latents themselves —
   not naive per-feature firing rates, but frequencies of feature *combinations*.
2. **Topic models / NMF / LDA** over activation vectors: model each activation as a
   mixture of latent "topics"; the base rate of inferred topics *is* a per-concept
   frequency estimate.
3. **Minimum Description Length**: carving out a rare concept into a dedicated dictionary
   element can shorten total description length vs. absorbing it — a principled,
   non-magnitude route to rare structure.

**Key argument (the "Rarity" discrete-signature idea, bioinformatics, btad750):** a rare
"lemur" and a rare "volcano" have *similar large* reconstruction errors if both are
absorbed by a general parent — magnitude cannot separate them — but they fire
*qualitatively different combinations* of SAE features. Defining a concept by its
**discrete co-firing signature** (which latents are on) replaces *distance in continuous
space* (the trap) with *counting occurrences of discrete identities*. This is a genuinely
different bet from the two refuted attempts, and both the sweep and Gemini converge on it.

Gemini's calibrated verdict: **likely solvable label-free; the obstruction is an artifact
of using geometric distance instead of combinatorial identity — not fundamental.**

## Caveats (Claude's adversarial check on the above)

The direction is right; the naive instantiation is not. Before pre-registering:

- **Exact-signature grouping fragments to noise.** In a high-dimensional sparse code
  almost every input has a *unique* binary signature, so exact-match counts ≈ 1 everywhere
  ("everything is rare" collapse). A similarity relaxation is required — group by signature
  overlap (Jaccard) or, better, model co-occurrence directly (i.e. NMF / topic model over
  the binary code, which is Gemini's own #1/#2).
- **Mild circularity.** Absorption *corrupts the signatures themselves* (the child fires
  the parent's composite latent), so any signature-based estimator partially depends on
  the SAE whose quality is in question. Controllable on synthetic data with known
  frequencies; a genuine caveat for the real-data version.

## Recommended direction

1. **Abandon** inverse-cluster-size-of-the-novelty-subspace as the estimator (dead end
   per the clustering-bias theorems).
2. **Pursue combinatorial identity**: co-firing signatures / topic-model (NMF/LDA) base
   rates over the *binarized* SAE code, with a Jaccard/overlap relaxation — evaluated on
   synthetic data with a common parent + mutually-exclusive rare children of known
   frequency. Success criterion: label-free per-child frequency estimates correlate
   highly (target Pearson r > 0.9) with ground truth **and** the child signature-sets are
   largely disjoint (distinguishes cat vs dog, the precise magnitude-trap failure).
   (Formal pre-registration spec: TBD — deferred.)
3. **Fallback**: adopt Matryoshka nested architecture, which reduces absorption without
   estimating frequency at all.

## Sources (primary unless noted)

- Matryoshka SAEs — arXiv:2503.17547 (ICML 2025)
- Clustering density biases — arXiv:1705.05950 (TPAMI 2019)
- Importance-weighting survey — arXiv:2403.10175 (Kimura & Hino 2024)
- Density-ratio outlier detection — Hido et al., KAIS 2011 (doi:10.1007/s10115-010-0283-2)
- FASSL — arXiv:2309.04723; Geometric Harmonization — arXiv:2310.17622; TASE — arXiv:2410.22883
- COLT — arXiv:2306.04934
- "Rarity" rare-population discovery — Bioinformatics btad750
- Not-yet-verified leads: UOT-RFM arXiv:2509.25713; unsupervised class-distribution estimation
