# Novelty sweep: the detector arc (Arm A → round 8)

*2026-07-23. Method: supervised gemx (Vertex, grounded google_web_search)
single-shot queries — 2/7 succeeded (raw: `results/novelty_round8/Q2.md`;
Q6 output was corrupted) — completed by Claude WebSearch/WebFetch for the
remaining coverage, with load-bearing items fetched and read. Complements
the §10 novelty assessment (ε\*, β\*, no-go, weighting — unchanged).*

## Per-claim verdicts

**1. Gated absorption mechanism (§16): PRIOR ART EXISTS — recalibrate.**
Chanin et al.'s "Toy Models of Feature Absorption in SAEs" (LessWrong,
2024–25) documents the encoder/decoder asymmetry explicitly: absorption
creates an encoder *hole* (a latent fires on feature₀ ∧ ¬feature₁) while the
decoder direction is unchanged, and suggests this asymmetry "can be used as
a signal to detect when this phenomenon is occurring." Our contribution is
therefore NOT the discovery of encoder gating but: (i) its quantification in
trained untied SAEs (mutual gating both directions, fire-rate/magnitude
tables, conditional TV 0.9999); (ii) the **identifiability consequences** —
signature counting recovers ρ to ≤0.02 given the pair, and the formal
**dictionary-vs-code identifiability distinction** as stated properties
(which we did not find claimed anywhere); (iii) the pre-registered
refutation of the shared-composite idealization in trained SAEs.

**2. Bimodality/histogram detection (Arm A H2): ANTICIPATED — and our
result directly qualifies it.** "Broken Latents: Studying SAEs and Feature
Co-occurrence in Toy Models" (chanind & Demian Till, LessWrong, 2024-12-30)
observed that broken latents "exhibit multiple peaks in activation
histograms," suggested label-free detection from them, and proposed
penalizing low-magnitude activations as a fix. Arm A is, in retrospect, a
pre-registered test of that suggestion in trained untied SAEs: **it fails at
σ = 0 because encoder gating suppresses the low mode entirely** (the
composite never fires on host-only events), and works only in a leak window.
Our identifiability note's formal apparatus (binarized-signature no-go,
mixture-identifiability framing) appears unclaimed. Cite Broken Latents
prominently; frame Arm A as testing-and-qualifying it.

**3. The pair detector as a method (cos band + two-sided lift + overlap
veto, locked thresholds, held-out validation, shuffled null): appears
UNCLAIMED.** Nearest neighbors, all distinct: Chanin et al. use NPMI /
co-occurrence to *measure* absorption given probe-based ground truth
(label-dependent); O'Neill et al. (arXiv:2408.02622 — **WRONG ID, corrected
below**) build feature-family
hierarchies from co-activation; Michaud et al. (arXiv:2410.19750) cluster
co-occurrence for functional lobes and note co-occurring features align
geometrically; Tree SAE (arXiv:2605.07922) and HSAE (arXiv:2602.11881) use
co-activation constraints *during training* as remedies. None detect
absorbed parent/composite *pairs* label-free post hoc, none use the
two-sided-lift dependence signature or a containment veto, none run
matched-negative/shuffled-null validation.

**4. Two-sided lift ("shared event stream: dependence far from independence
in either direction") as the discriminative invariant: appears unclaimed.**

**5. Cross-seed pair/structure stability (S1): feature-LEVEL seed
(in)stability is documented** — Paulo & Belrose (EleutherAI blog, "SAEs
trained on the same data don't learn the same features") and "Unstable
Features, Reproducible Subspaces" (arXiv:2606.12138, 2026-06) — but
matching *pairs/structures* across seeds with a bijective pair score
appears unclaimed. Cite both as context: S1's seed-stable pair clusters are
noteworthy precisely because individual features are seed-unstable.

**6. Shuffled-firing dependence null: standard permutation technique**; its
application to SAE latent-pair pathology detection appears unclaimed but is
methodologically routine — claim as good practice, not novelty.

**7. Benchmarks/metrics context:** SynthSAEBench (arXiv:2602.14687,
2026-02; abstract fetched) provides ground-truth synthetic SAE evaluation
with correlation/hierarchy/superposition but no absorption metrics or
pair-detection — complementary, not overlapping; a natural future home for
the detector as a benchmark task. Absorption measurement in the literature
remains probe-based (first-letter tasks, k-sparse probing) — i.e.,
label-dependent, supporting claim 3's framing.

## Actions taken

- PAPER.md: §7 reframed to credit encoder-hole prior art and state our
  delta; §8 adds the Broken Latents connection for Arm A; Related Work
  gains O'Neill, Michaud, HSAE, SynthSAEBench, Broken Latents, Toy Models
  post, Paulo & Belrose, and arXiv:2606.12138.
- report.md §16: pointer note added.

## Sources

- Toy Models of Feature Absorption in SAEs — lesswrong.com/posts/kcg58WhRxFA9hv9vN
- Broken Latents — lesswrong.com/posts/XHpta8X85TzugNNn2 (chanind, D. Till, 2024-12-30)
- A is for Absorption — arXiv:2409.14507 (NeurIPS 2025 oral)
- O'Neill et al. — arXiv:2408.02622 · Michaud et al. — arXiv:2410.19750
- Tree SAE — arXiv:2605.07922 · HSAE — arXiv:2602.11881
- SynthSAEBench — arXiv:2602.14687 · Unstable Features — arXiv:2606.12138
- Paulo & Belrose — blog.eleuther.ai/sae_seed_similarity/

---

## Correction (2026-07-25): misattributed arXiv id

The O'Neill citation above is **wrong**, found by `ops/check_citations.sh`:

- **arXiv:2408.02622** is *"Language Model Can Listen While Speaking"* (Ma et al.),
  a full-duplex speech-language-model paper with no connection to SAEs.
- The intended work is **arXiv:2408.00657**, *"Disentangling Dense Embeddings with
  Sparse Autoencoders"* (O'Neill, Ye, Iyer, Wu), which does introduce
  "feature families" representing related concepts at varying levels of abstraction.

The wrong id came from this novelty sweep's raw output (`results/novelty_round8/Q2.md`,
a grounded-search answer). It is left in place above, flagged, so the provenance of
the error stays visible; `PAPER.md` §9 is corrected.

Note what the failure mode was: the id **resolved to a real arXiv paper**, just the
wrong one — digits corrupted into a valid neighbour. A "does this id exist" check
passes cleanly. That is why `ops/check_citations.sh` prints the resolved title
beside the citing text instead of only checking resolution.

The **mechanism** attribution ("from co-activation") could not be verified from the
abstract of 2408.00657, and the full text was not machine-readable at check time, so
PAPER.md now says only that the work groups features into families spanning levels of
abstraction — which the abstract does support. Do not restore the stronger claim
without reading the method section.
