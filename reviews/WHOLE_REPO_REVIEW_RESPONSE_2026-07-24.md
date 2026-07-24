# Whole-repo review: response (2026-07-24)

Review: `WHOLE_REPO_REVIEW_GPT-5.6_2026-07-24.md` (GPT-5.6 Sol, High; fetched
the public repo at HEAD `6248c01`, independently reran the theory checks and
the round-9/10 frozen scorers — reproduced; verdict **MAJOR REVISION**, with
the affirmation that *no failed results were hidden or reclassified into
successes*). Every finding was correct. Dispositions below; all applied unless
marked *queued* (needs a compute run).

## Required changes (review numbering)

1. **Round 11 semantic headline — WITHDRAWN.** `results/real/SUMMARY.md` now
   states only the raw fact (25,041 vs 936 flagged *under that script*) and
   itemizes why the "~27× redundancy / supports the hypothesis" reading is not
   defensible: θ=0.0≠registered 0.05 (architecture-asymmetric), unseeded
   subsample, abs cosine, raw counts (opportunity-normalized ≈ **15.3×** not
   27×; cluster over-count), 99.5% vs 87.3% band-flagged ⇒ geometry not
   co-firing, top-15 only. `real_analyze.py` fixed (θ=0.05, seeded shared
   subsample, signed cosine, opportunity + connected-component stats, top-40).
2. **SAE quality not "held out."** `real_train_sae.py` comment corrected
   (in-cache Monte Carlo FVU, not held-out); SUMMARY reframed (legitimate real
   SAEs in the minimal sense, **not** benchmark-quality); proper eval
   (doc-separated test, loss-recovered/KL, Pareto) folded into the confirmatory
   run. *(script comment applied; the eval itself is queued)*
3. **Coherence no-go narrowed everywhere.** `theory/general_no_go.md` + `PAPER`
   now scope "cannot remove absorption / no penalty of any form / bites in
   full" to the restricted O₂ orthonormal-frame class (m≤d, stated column
   conditions; overcomplete + mixed-plane open); √2 flagged as asymptotic, not
   the exact finite-λ ratio. *(Complete piecewise proof / interval bounds for
   the omitted global cases: queued for the arXiv version.)*
4. **TopK narrative matched to round 10.** `theory/topk_absorption.md`
   retitled around the two-atom oracle law + overcomplete escape; STATUS
   BANNER added (practical resistance prediction contradicted in round 10,
   round 11 unresolved); §8 rewritten as "the conjecture this motivated — and
   why it did NOT survive testing"; P4 described as "direction inverted."
5. **L1 crossover active-set domain added** (λ<√2 for the absorbed child-solo
   formula, PAPER §4); `verify_absorption_theory.py` docstring corrected (ε\*
   is the pure-candidate crossover; the global optimum tilts continuously, no
   jump); "uniform scaling collapse" softened to consistency on a modest grid
   with the proper regression test queued.
6. **Non-identifiability ≠ "absorption is the truth."** `report.md` now states
   the three claims separately (data non-identifiability / objective preference
   / code identifiability); absorption is the objective's preferred coding of a
   non-identifiable distribution.
7. **Gating estimator scoped.** round-9 SUMMARY "decisive in all 16 cells"
   → "P1M/P2M pass 16/16; P4 16/16 but sparse σ=0 contributor counts"; the
   oracle-scoping was already registered and is retained.
8. **Semi-synthetic detector downgraded to what it shows** — planted-pair PoC,
   natural-feature adjudication null (0/15); PAPER/results already say
   "planted"/"synthetic proof of concept"; round-8 amendment described as
   in-flight (disclosed), not a clean pre-run lock, in the claim ledger.
9. **Audit trail + consistency.** Round counts/status unified (PAPER "through
   round 11"; README "Eleven rounds"; plan "post round-11"); **`CLAIM_LEDGER.md`
   added** (claim → lock → scorer → verdict → location); "external review"
   relabeled **LLM-assisted adversarial review** in README/PAPER/plan.
10. **Novelty + bibliography.** "Minder et al." → Chanin/Dulka/Garriga-Alonso;
    C2R → "Cross-sample Consistency Regularization"; "exact mechanism for
    feature hedging" → "solvable toy analogue / candidate mechanism". Prior art
    (Chanin absorption, Matryoshka, OrtSAE, C2R, stitching/meta-SAE) already
    cited; novelty framed as the exact solvable-model layer.
11. **PAPER restructure around one contribution** (theory-first arXiv note;
    round 11 out of the abstract). *Queued* — a focused rewrite; the current
    doc is honest and suits a LW/AF technical report now.
12. **Round-11 reproducibility.** `results/real/ARTIFACT_MANIFEST.md` added
    (GCS object sizes + md5/crc32c hashes, model/layer/corpus/SAE
    hyperparameters, environment); remaining gaps (persisted eval indices,
    full pair outputs, lockfile, revision hashes) are part of the confirmatory
    run.

## Highest-value next experiment (adopted)

Combined into **one pre-registered experiment** in `RESEARCH_PLAN.md`
(real-model track): matched-seed L1-vs-TopK with **causal first-letter
absorption on a held-out SAEBench-style set as the primary endpoint**, the
detector scored blind as a secondary predictor, λ/k Pareto sweep,
opportunity/cluster normalization, and a modern control if compute permits.

## Net

The review confirmed the theory as the strongest, publishable-with-scoping
contribution and the pre-registration discipline as sound; the fixes remove
the interpretive overshoot (chiefly round 11) and the stale/inconsistent
packaging. Deferred items (arXiv restructure, the confirmatory causal run,
certified proofs of the no-go's open cases) are the explicit path to an arXiv
note; the current repo is an honest LW/AF-grade technical report.
