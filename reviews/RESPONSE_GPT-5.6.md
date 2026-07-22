# Response to the external GPT-5.6 review (2026-07-22)

Review archived verbatim at `EXTERNAL_REVIEW_GPT-5.6_2026-07-22.md`. Verdict
accepted: **major revision, applied same day.** Point-by-point:

## Accepted and fixed

**R1. "ε\* is a candidate-dictionary loss crossover, not the global transition."**
Accepted for the *headline language*, with one factual correction to the review
(below). All uses of "exact phase boundary" replaced by "exact pure-strategy
crossover" (title, executive summary item 2, §4 heading, README); §4 gains an
explicit framing note (global 2-latent optimum tilts continuously; its 67.5°
crossing sits at ≈ 0.88·ε\*; ε\* upper-bounds and organizes the transition).
*Factual note:* the intermediate angles the review cites (69°/80°/85°) are
quoted from this report's own §5 (Result 3), which has framed the continuous
tilt as a headline result — "ε\* marking the exact point where the pure
absorbed strategy loses to the pure faithful one" — since round 1, and §8 has
carried the 0.88·ε\* global-optimum crossing figure. The defect was branding
("exact phase boundary" in title/headings), not a missing distinction. Both
are now consistent.

**R2. Theorem 1b uniqueness over-claim.** Accepted, exactly right: unused atoms
attain the bound. Statement corrected in §3 and the executive summary to
"active dictionary unique up to permutation and unused atoms," with an inline
correction note.

**R5. The m = 32 capacity confound.** Accepted — the sharpest catch in the
review. 30 background features + parent + child + composite = 33 > 32, so the
round-1 "dynamics gap" claim (SGD avoids the triple *even without scarcity*)
was untestable in that architecture and is **withdrawn** (§8, struck and
corrected in place). Note the report's own §14 (m = 1536) already showed
triples forming in every spare-capacity condition, corroborating the review.
Rerun at m ∈ {32, 34, 40} implemented (`experiments/capacity_m33_rerun.py`,
prediction pre-registered in the script header: triples appear with headroom);
results will replace the withdrawn claim either way.

**R6. "Semi-synthetic" labeling.** Accepted. §14 retitled "Semi-synthetic
results: synthetic-pair injection into real GPT-2 activations" with a scoping
note; README updated; the audits' null result (no natural-absorption positive
observation yet) stated explicitly.

**R-framing. "Machine-checked" → "computationally verified."** Accepted
throughout (title line, §4, README): sympy symbolic enumeration + exact scans,
explicitly "not a proof assistant." The referee review is now labeled
"fresh-context LLM referee (adversarial, not external human peer review)."
"Research program complete (single day)" replaced by "actively developed
technical report."

**R7. Reproducibility metadata.** Partially applied (environment note +
provenance-by-commit in README). A pinned lockfile and per-table commit hashes
remain TODO.

## Accepted in substance, already present in the report

**R3. No-go scope.** The proven scope (m ≤ d, orthonormal frames, L1,
non-pair latents off the pair plane; mixed configurations excluded by search
not proof; overcomplete open) has been stated in §7.1b and
`theory/general_no_go.md` since the general derivation landed, including the
p₀\* domain boundary where penalties *do* work. We agree the executive-summary
sentence read stronger than the body; it already carries the domain-boundary
caveat, and the body scope governs. TopK/JumpReLU and the overcomplete case
are explicitly open problems (§10, memory of the program); a TopK extension is
on the queue.

**R4. Oracle-dependence of the weighting remedy.** Agreed and long-standing:
§15's own language is "the oracle result stands as the theory validation; the
practical estimator remains future work," after §15b refuted two label-free
substitutes with mechanisms. "Diagnostic existence result" is a fair summary
and §12/§15 now read that way. The successor line of work (Arm A → the
pair-identification pre-registration in `notes/prereg-pair-identification.md`)
is precisely the attempt to close this gap label-free.

## Net effect

The review's bottom-line reframe — *"a solvable model of feature absorption
with an exact pure-strategy crossover, a rotation blind spot for Gram
penalties, and an oracle-level repair"* — is adopted as the paper's identity
(new title). The two empirical actions it forced (m ≥ 33 rerun; natural-pair
benchmark) are now queued experiments rather than framing debt.

---

## Round 2 (same day): reviewer's follow-up on the revision

Verdict received: upgrade to "credible, workshop-ready technical report."
All requested theoretical and framing revisions have been applied. The
controlled capacity rerun and complete GPU environment pinning remain open
implementation items, tracked in §"Pending" below. Item-by-item:

1. **Theorem 1b duplicate-atom counterexample — accepted, fixed.** The reviewer
   is right that ‖f‖₁ is invariant to splitting a coefficient across duplicate
   collinear columns, so "extra atoms stay unused" was still too strong.
   Restated (§3 + executive summary): the *set of active directions*
   {a_p, a_m} is unique; dictionary columns and codes are non-unique under
   permutation, duplication with code-splitting, and unused atoms. Parameter
   assumptions now explicit (p₀, q > 0; 0 < λ < 2 so r ≥ λ/2 for both events).
2. **Practitioner rule — accepted, reworded** (report + README): 1.17·λq is
   the characteristic *scale* of the transition (exact only as the
   pure-strategy crossover; global two-latent midpoint ≈ 0.88·ε\*, SGD
   0.58–0.70·ε\*), and "more compute cannot help" is scoped to the population
   optimum under the two-latent capacity constraint.
3. **Evidence tiers — accepted.** §7.1b headline now labels
   (proved) rotation blindness / (analytic reduction) in-plane frame
   competition / (numerical) finite-β boundaries and mixed-configuration
   search explicitly.
4. **m = 34/40 rerun + reproducibility.** The capacity rerun is bundled into
   the in-flight GPU session (results will land in `results/capacity_m33/`).
   README now pins the CPU verification environment exactly and carries a
   results-CSV → commit provenance table. Full lockfile still TODO (the GPU
   image's torch version is recorded in run logs).

---

## Round 3: final revision brief (docx), applied

Completed in this pass: README uniqueness sentence replaced with the
reviewer's exact formulation; Theorem 1b equality condition tightened to
"iff the set of directions used with positive probability is exactly
{a_p, a_m}" (both directions must occur); response-status language made
exact; repo-wide consistency pass (4 stale "machine-checked/verified"
phrasings, "Theorem 1b's unique global optimum" → "optimal active-direction
set", §11's "closed-form phase boundary" → "pure-strategy crossover", figure
alt-text); `ENVIRONMENT.md` added with the pinned CPU verification stack,
exact run commands, and the list of results independently rerun from a clean
environment (theory_merged, symbolic_verify — all checks pass —, analyze_ab
on committed CSVs, Arm A CPU smoke).

## Pending (tracked, not claimed done)

- **m ∈ {32, 34, 40} capacity rerun:** pre-registered (K1–K3 in
  `experiments/capacity_m33_rerun.py`), bundled into the in-flight GPU
  session; results will land in `results/capacity_m33/` and be reported
  whichever way they fall.
- **Pair-ID Arm 1 confirmatory run:** locked prereg, same GPU session;
  `results/prereg_pairid/`.
- **Full GPU lockfile:** pip freeze capture from the session box, to be
  committed with the session results.
- **Natural-feature absorption benchmark** (review R6 second half): open;
  the audit-v3 scan inside pair-ID Arm 2 is the next step toward it.

## Declined / archival-scope note

Historical artifacts are not retro-edited: `results/round*/SUMMARY*.md`,
`reviews/*` (including the archived referee report) and
`docs_novelty_adjudication.md` are point-in-time records of what was believed
when they were written; the live claims of the project are those in
`report.md`, `README.md`, `theory/`, and `notes/`. Editing dated records to
match later understanding would falsify the provenance trail this repository
is explicitly organized around (pre-registration → result → correction).
