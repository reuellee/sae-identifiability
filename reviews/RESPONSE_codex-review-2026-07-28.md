# Response to the Codex whole-repo review of 2026-07-28

Review: `CODEX_REVIEW_2026-07-28.md` (at commit `a4a89c3`). Point-by-point;
**[A]ccepted / [Q]ueued / [N]oted** — applied same day unless queued.

1. **Round 15 does not establish model-level transfer. [A]** Correct: the CI
   is over letters conditional on one released SAE per width; the supported
   statement is "the positive within-suite width association reproduced in one
   independently trained JumpReLU suite." `SUMMARY_round15.md` retitled
   ("reproduces on", not "transfers to") and its scope bullet rewritten to the
   reviewer's formulation; multi-seed (`262k_l0_medium_seed_1`) and multi-suite
   (270m/4b) replications queued in the plan as the path to a transfer claim.

2. **Round 14 carrier conclusion not identified by its statistic. [A]** The
   per-trial share is concentration of the positive projection conditional on
   positive mass; it bounds out "single recurring parent" but does not
   establish carriage (magnitude vs probe margin, negative cancellation,
   causality unmeasured — Chanin established carriage by ablation). PAPER §8b.7
   and the abstract rewritten to the supported reading ("consistent with",
   causal half = registered successor); the successor spec in the plan now
   names signed/absolute contribution vs raw probe margin + held-out selection
   + ablation.

3. **Round 13b P1 clustering unit. [A, independently verified]** The prereg
   registered bootstrap over seeds; the evaluator pooled 16 arch-diffs (same 8
   seeds twice). Recomputed from committed `round13b_results.json` with seeds
   as clusters: mean −0.0445, t-CI [−0.0515, −0.0374] (bootstrap
   [−0.0498, −0.0389]) — reproducing the reviewer's numbers exactly.
   FALSIFIED-DIRECTION stands. Correction-of-record appended to
   `SUMMARY_round13b.md`; frozen evaluator left as-run; the seed-clustered
   interval is the citable one. (PAPER's abstract retains the as-run CI with
   the correction note in §8b — updated at next full pass.)

4. **Word-domain shift conditionality. [A/Q]** Conditionality now stated
   explicitly in `SUMMARY_round15.md` (with the reviewer's own mitigating
   observation that near-constant FVU across widths disfavors a trivial
   reconstruction account of P1). Sequence-context replication queued.

5. **Novelty framing. [Q]** The four references are acknowledged; the
   theorem-by-theorem comparison table (vs 2409.14507, 2505.11756, 2506.15963,
   2606.30609) is queued as a paper task extending
   `docs_novelty_adjudication.md`. No novelty claim is strengthened meanwhile.

6. **Artifact sync. [A, partial]** README updated (15 rounds, 2026-07-28,
   CLAIM_LEDGER declared canonical, both whole-repo reviews cited); PAPER
   banner updated (through round 15, abstract-level; ledger canonical).
   Full single-source regeneration + one reproducibility entry point remains a
   pre-submission task (queued with #5). ARTIFACT_MANIFEST extension to rounds
   12–15 queued with it.

7. **Crossover domain in abstract. [A]** "(valid on the active branch
   λ < √2)" added to the abstract's formula display.

**On the review's positive verification:** the reviewer independently reran
the symbolic checker, the frozen 13b and 15 evaluators (bit-reproduced), and
the round-14 self-test — this is the first external reproduction of the
round-15 pipeline and is noted with thanks.
