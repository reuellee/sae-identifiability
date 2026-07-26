# Novelty sweep: is "the absorption metric measures absence, not absorption" claimed?

Date: 2026-07-26. Method: `gemx` grounded search (single-shot, `ops/`-style driver)
for the framing question, plus direct arXiv API queries from the orchestrator for
verification. **Every arXiv id below was resolved against the live arXiv API and the
title/author list checked** — not taken from model recall. That check earned its
keep twice in one sitting (see "Corrections" at the end).

## Question

Rounds 12–14 converged on a *construct-validity* claim: the first-letter absorption
endpoint fires when "the letter is present and no sufficiently selective latent
fired", which is an **absence**, and round 14 finds no carrier picking up the child's
mass. Has anyone already published that the Chanin/SAEBench absorption metric
conflates absorption with representational loss, threshold suppression, or splitting?

## Verdict: the specific claim appears UNCLAIMED

The grounded search returned an explicit NONE FOUND for a critique of the absorption
metric on these grounds, and the three nearest papers — all located and verified
independently — each attack a *different* axis. This is a live and crowded research
area, so the boundary matters and is drawn below.

## The three nearest works, and why each is complementary rather than prior

| work | what it audits | axis | overlap with §8b.7 |
|---|---|---|---|
| Chanin, *Are Sparse Autoencoder Benchmarks Reliable?* (arXiv:2605.18229, 2026-05-18) | SAEBench quality metrics via reseed noise, ground-truth correlation, discriminability | **reliability** | none on construct validity — asks whether metrics are *stable and discriminating*, not whether the quantity means what its name says. Finds TPP and SCR unfit; absorption is not singled out. |
| Bal, *From Geometric Recovery to Causal Validation* (arXiv:2607.12166, 2026-07-13) | cosine-recovery metrics for SAE features | **construct validity, but of the recovery metric** | closest in *spirit* and the key citation: same move (a standard metric conflates two claims; settle it causally). Different target — decoder geometry vs encoder activation, finding up to 77% of "recovered" features causally inert. Says nothing about absorption. |
| Leask et al., *SAEs Do Not Find Canonical Units of Analysis* (arXiv:2502.04878, 2025-02-07) | whether SAE latents are atomic units at all | **ontology** | argues features are non-atomic (splitting/composition). Motivates the family endpoint of round 13a; does not analyse the absorption *metric*. |

Also relevant, not competing: SAEBench itself (arXiv:2503.09532) as the source of the
standardised metric; hierarchical-architecture remedies (arXiv:2506.01197,
arXiv:2602.11881, arXiv:2605.07922) which attack the pathology at training time.

## What that leaves as this project's contribution

1. The absorption endpoint infers a merge from an **absence** and, tested directly for
   a carrier, the absence looks like loss/threshold suppression (round 14).
2. The single-latent form of the endpoint additionally **inflates absorption 23–33%**
   by counting feature splitting (rounds 13a/13b), with a splitting-corrected family
   endpoint given.
3. The endpoint is **monotone in dictionary width** in the direction opposite to the
   capacity story (round 13b), i.e. it co-moves with fragmentation.

Positioning: Chanin asks whether SAEBench metrics are *reliable*; Bal asks whether
*recovery* metrics measure what they claim; this work asks whether the *absorption*
metric measures what it claims. The three are siblings and should be cited as such —
the framing "metric criticism is now an active line and this is a specific,
complementary entry" is stronger and more honest than claiming an isolated novelty.

## Corrections this sweep forced

- **PAPER.md misattributed a citation.** "Seed-level feature instability (Paulo &
  Belrose; arXiv:2606.12138)" — that id is Gerasimov et al., *Unstable Features,
  Reproducible Subspaces*. Paulo & Belrose is **arXiv:2501.16615**, *Sparse
  Autoencoders Trained on the Same Data Learn Different Features*. Both are real, both
  are relevant, and the id resolved to a real paper — so an existence check passes and
  only an author check catches it. Fixed by citing both. This is the **second** time
  this exact failure mode has appeared (the first: O'Neill cited as arXiv:2408.02622,
  a speech paper; correct id 2408.00657). `ops/check_citations.sh` prints resolved
  titles beside citing text precisely for this.
- **The search oracle misattributed authorship.** It reported arXiv:2607.12166 as by
  "Leask, Bussmann, Tigges, Nanda et al." It is a **single-author** paper by Mohamed
  Abdessalem Bal, and the oracle could not produce the id. The paper is real; the
  attribution was invented. Treat grounded-search author lists as leads to verify, not
  as facts — the API lookup is cheap and decisive.

## Consequence for the north star (read this before planning round 15)

`RESEARCH_PLAN.md`'s stated pivot is **causal validity**: "whether a *recovered* child
code actually mediates the child feature's effect (ablation/intervention)". That is,
almost exactly, what arXiv:2607.12166 did thirteen days ago — subject every feature
passing a cosine-recovery bar to ablation and steering, and report what fraction is
causally inert (up to 77% in a degraded SAE, 9% in a good one, 14% in a production
SAE). It also ships the method as a reusable instrument, `sae-causal-audit`.

This does not scoop rounds 12–14, which are about the *absorption* endpoint. It does
partly occupy the ground the plan intended to move onto next, so the next round should
be chosen with that in mind:

- **Do not rebuild the general causal-audit apparatus.** If an intervention stage is
  wanted, start from the published instrument and cite it.
- **The absorption-specific question is still open and is the better target**: on
  trials the metric calls absorbed, does *ablating the putative carrier* remove the
  child's effect? Bal asks whether recovered features are causally real in general;
  nobody has asked whether the thing absorption metrics implicitly posit — a parent
  latent carrying a child — is causally real. Round 14 is the correlational half of
  that question; the interventional half is unclaimed.
- Related distinction to preserve: Bal's inertness is about *encoder activation* given
  decoder geometry. Round 14's finding is about *where the mass goes* when the
  selective latent is silent. Different failure, same methodological moral.
