# Whole-repo review — GPT-5.6 Sol (High), chatgpt.com, 2026-07-24

*Comprehensive adversarial review of the ENTIRE repo at HEAD 6248c01. GPT fetched
the public repo, independently reran the theory checks + round-9/10 frozen scorers
(reproduced), and compared against the SAE literature. Verdict: MAJOR REVISION —
"scientifically promising, unusually honest, but not yet publication-ready";
explicitly: "no evidence that failed results were hidden or reclassified into
successes." Verbatim below; responses in ROUND_ALL_REVIEW_RESPONSE_2026-07-24.md.*

# Overall verdict

**MAJOR REVISION - scientifically promising, unusually honest, but not yet publication-ready as an arXiv paper.**

The repository contains a credible and potentially publishable **narrow theoretical contribution**: an exactly solvable nonnegative-(L_1) model of absorption, a genuine observational non-identifiability result, and a useful construction showing why decoder-only coherence regularization can be blind to semantically bad rotations. The preregistration discipline-especially in Rounds 9 and 10-is substantially better than normal exploratory ML practice. I found no evidence that failed results were hidden or reclassified into successes.

However:

* Several headline statements are broader than the mathematics supports.
* `theory/topk_absorption.md` retains a practical conclusion that Round 10 directly contradicted.
* The Round 11 "~27x more redundant/split pairs" interpretation is not currently scientifically defensible because the detector is architecture-asymmetric and the comparison is not reproducibly sampled or opportunity-normalized.
* The formal deliverables are stale and internally inconsistent.
* The real-model evidence does not yet cross the gap from "an SAE was trained on real activations" to "absorption was measured on real SAE features."

I audited the claim-bearing repository at the pinned commit: the deliverables, theory notes and machine checks, preregistrations, result summaries, relevant scorers and experiment scripts, prior review trail, and commit provenance. I independently reran the main theory checks and the frozen Round 9/10 scorers; their reported numerical outputs reproduce. I could not independently rerun Round 11 because its approximately 5.7 GB activation cache and SAE weights are stored outside GitHub in GCS, without a public artifact manifest or checksums. ([GitHub][1])

## Headline-claim audit

| Claim                                            | Verdict                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ( \varepsilon^*(\lambda,q)\approx1.17\lambda q ) | **Supported with important scope restrictions.** Exact for the two specified pure candidate dictionaries on the relevant active-set branch; (1.17\lambda q) is a small-(\lambda) approximation, not the trained transition.                                                                   |
| Non-identifiability wall at (\varepsilon=0)      | **Supported.** The observational-equivalence result is exact. The active-direction optimum characterization is sound under the stated unit-column, nonnegative-code and (\lambda) assumptions.                                                                                                |
| Coherence-penalty no-go and (\sqrt2) ratio       | **Partially supported and overstated.** The Gram-penalty rotation blind spot is exact. The global no-go is proved only in a restricted orthonormal-frame class; the overcomplete and mixed-plane cases remain open. (\sqrt2) is an asymptotic ratio, not the exact finite-(\lambda) boundary. |
| ( \varepsilon^*_{\rm TopK}=2q )                  | **Supported only for the oracle-coded, exactly-two-atom, (k=1) model.** It is not a theorem about trained overcomplete TopK SAEs.                                                                                                                                                             |
| Gating-corrected (\hat\rho_D)                    | **Strongly supported as an oracle-pair mechanism estimator.** Not yet supported as an end-to-end label-free or all-token operational estimator.                                                                                                                                               |
| Round 11 L1-vs-TopK "~27x redundancy"            | **Raw detector-count fact reproduced; semantic interpretation unsupported.** It should not presently be called a robust redundancy, splitting, or absorption ratio.                                                                                                                           |

The first two conclusions follow directly from the population-loss calculation and wall proof; the manuscript correctly distinguishes the pure-candidate crossover from the continuously tilted optimum, though it omits an important active-set domain condition discussed below. ([GitHub][2])

---

# Required changes, prioritized

## 1. Withdraw the current Round 11 semantic headline pending a corrected analysis

`results/real/SUMMARY.md` may accurately say:

> "Under this particular analysis script, 25,041 L1 pairs and 936 TopK pairs were flagged."

It cannot yet say that L1 has "~27x more redundant/split pairs," that the ratio is "robust," or that Round 11 supports the practical hypothesis that TopK resists L1 absorption.

The main problems are in `experiments/real_analyze.py`:

* It says the constants are inherited from Rounds 8/9 but changes the firing threshold from the preregistered `THETA=0.05` to `THETA=0.0`.
* That is architecture-asymmetric. In the L1 SAE, every arbitrarily small positive ReLU output counts as firing; TopK produces exact zeros outside its selected set.
* `torch.randperm(N)` has no persisted or explicit seed. Separate architecture runs are not guaranteed to use the same 50,000 tokens.
* It uses `abs(D.T @ D)`, so negatively aligned decoder pairs enter a detector later described as finding "aligned" or redundant features.
* Only the 15 highest-absolute-cosine pairs are saved and inspected. That is an extreme-value sample, not evidence that all 25,041 pairs-or even most of them-are splitting.
* Pair counts are not independent units: a redundant feature cluster of size (r) creates (r(r-1)/2) flagged pairs. Comparing pair counts without cluster adjustment can quadratically amplify modest differences in feature fragmentation. ([GitHub][3])

There are also unequal counting opportunities:

* TopK: 9,518 eligible latents, or 45,291,403 possible pairs.
* L1: 12,573 eligible latents, or 79,033,878 possible pairs.

The raw flag ratio is (25{,}041/936=26.75). After merely normalizing by eligible-pair opportunities, it is approximately **15.33x**, not 27x. More importantly, 99.5% of L1 cosine-band pairs and 87.3% of TopK cosine-band pairs are flagged, so most of the difference is already present in the decoder-cosine geometry; the co-firing test contributes little additional discrimination after band selection. ([GitHub][4])

**Required fix:** rerun on the same persisted token indices, with explicit seeds, a preregistered architecture-neutral firing definition, signed-cosine reporting, repeated evaluation subsamples, pair-opportunity normalization, and cluster-level as well as pair-level statistics. Save either every flagged pair or sufficient aggregate distributions-not just the top 15.

---

## 2. Stop calling the Round 11 SAE evaluation "held out" or sufficient evidence of high quality

`experiments/real_train_sae.py` calls its final evaluation a "held-out random slab," but it samples that slab from the same activation cache repeatedly used for training. With 30,000 steps and batch size 4,096, training consumes roughly 123 million sampled rows from a cache containing about 1.4 million rows-approximately 88 cache-equivalents, with replacement. The final FVU is therefore an in-cache Monte Carlo evaluation, not held-out generalization. ([GitHub][5])

The activation corpus is also unusually narrow: up to four Project Gutenberg novels, with no document-level train/test separation. ([GitHub][6])

Thus:

* **Yes**, these are legitimate real SAEs in the minimal sense that wide SAEs were trained on real Pythia-1.4B residual-stream activations.
* **No**, FVU (0.043/0.056) and (L_0=32) do not establish benchmark-quality or generally useful SAEs.
* The two architectures are matched on (L_0), but not on reconstruction: TopK has materially lower FVU.
* FVU alone does not measure model behavior preservation, feature disentanglement, absorption, or causal validity. Modern SAE evaluation explicitly uses multiple metrics because reconstruction proxies can miss practically important differences. ([arXiv][7])

**Required fix:** create document- or corpus-separated train/validation/test activations; report test FVU, loss recovered or KL/CE degradation after SAE insertion, dead-feature rate, sparsity, and training curves. Compare architectures along matched sparsity-reconstruction Pareto frontiers rather than one hand-selected (\lambda) and one (k).

---

## 3. Narrow the coherence-penalty "no-go theorem" everywhere it appears

The strongest exact result is valuable:

> Any penalty depending only on pairwise decoder inner products is constant across orthonormal frames with the same Gram matrix, so it cannot distinguish faithful from anti-rotated semantic orientations.

That is a real and useful blind-spot result.

But the stronger theorem in `theory/general_no_go.md` is restricted to:

* (m\le d),
* globally orthonormal dictionaries,
* two columns exactly spanning the pair plane,
* remaining columns orthogonal to that plane,
* background directions orthogonal to the pair,
* nonnegative-(L_1) coding,
* penalties from the specified pairwise-angle class.

The note explicitly concedes that partially in-plane/background-mixed columns were checked only by random rotation searches, not proved, and that the general overcomplete case (m>d) is open. ([GitHub][8])

Therefore phrases such as:

* "coherence penalties cannot remove absorption,"
* "no penalty of any form or strength rescues children,"
* "the no-go bites in full,"

must be rewritten as:

> "Within the (O_2) orthonormal-frame class, no decoder-angle-only pairwise penalty distinguishes the faithful and anti-rotated frames; in the co-occurrence-dominated region, the population objective prefers the anti-rotated frame."

The (\sqrt2) statement also needs consistent wording. The exact finite-(\lambda) boundary is

[
\frac{p_0^*(\lambda,q)}q
========================

\frac{(2-\sqrt2)-\lambda/4}
{(\sqrt2-1)-\lambda/4},
]

and only tends to (\sqrt2) as (\lambda\to0). It is not an exact universal critical ratio. ([GitHub][8])

For an arXiv theorem, replace random search with either a complete piecewise analytic optimization of the one-dimensional frame objective or certified interval bounds. Machine checks are excellent regression tests, but they are not a proof of the omitted global cases.

---

## 4. Correct the TopK narrative to match Round 10

The exact proposition in `theory/topk_absorption.md` is sound:

* With exactly two atoms and oracle nonnegative (k=1) coding, the faithful joint-event loss is (q).
* The absorbed child-solo loss is (\varepsilon/2).
* Hence the pure-strategy crossover is (\varepsilon=2q).
* Adding a third parent/child/composite atom permits zero reconstruction loss for all three event classes.

The note itself acknowledges the oracle and exact-two-atom restrictions. ([GitHub][9])

Nevertheless, its title and practical headline still say that "overcomplete TopK resists" absorption and that "TopK resists the feature absorption that L1 suffers." Round 10 found the opposite in the tested isolated regime: overcomplete L1 recovered the child in 100% of runs, while TopK recovery was 0.62-0.83; the registered trained-SAE crossover was inconclusive and capacity-collapse prediction was falsified. ([GitHub][10])

Round 11 cannot currently rescue that practical headline because of Finding 1.

**Required fix:**

* Retitle the note around the **two-atom oracle law and overcomplete escape**, not "why TopK resists."
* Add a prominent status banner: "Practical trained-SAE prediction contradicted in Round 10; background-rich comparison unresolved."
* Describe P4 as a "descriptive comparison whose direction inverted," since P4 had no registered confirm/falsify bar. "Refuted" is rhetorically understandable but stronger than its preregistered status. ([GitHub][11])

---

## 5. Add the missing active-set domain to the (L_1) crossover theorem

The reported event losses and (\varepsilon^*) formula reproduce in the tested regime. But the manuscript gives them without specifying the branch on which the nonnegative-Lasso coefficients remain positive.

In particular, the absorbed child-solo formula

[
\frac12+\frac{\sqrt2\lambda}{2}-\frac{\lambda^2}{4}
]

uses the positive composite coefficient (1/\sqrt2-\lambda/2), so that branch requires (\lambda<\sqrt2). Above that point the optimum changes active set and the displayed formula is no longer globally valid as written. The machine script only tests (\lambda\le0.5), safely inside this range. ([GitHub][2])

This does not invalidate the experiments, which use small (\lambda), but it does invalidate presenting the formula without a domain or piecewise extension.

There is also an internal script error: the `verify_absorption_theory.py` docstring says the global-optimum switchover matches (\varepsilon^*), while the script's own scan-and the paper-show a continuously tilting optimum rather than a global jump at (\varepsilon^*). Correct the docstring. ([GitHub][12])

The empirical (0.58)-(0.70,\varepsilon^*) result is reproducible, but "uniform scaling collapse" is stronger than the evidence. It is based on a modest parameter grid with substantial seed-level spread. Report a regression of the transition estimate on (\lambda q), uncertainty on its exponent and intercept, and a comparison against plausible alternatives such as separate (\lambda) and (q) effects.

---

## 6. Keep "non-identifiability" distinct from "the absorbed ontology is true"

The wall theorem is one of the repository's strongest results. At (\varepsilon=0), the hierarchical and flat/composite generative descriptions induce exactly the same observed activation distribution. The lower-bound argument also correctly establishes the parent and composite event directions as the positively used optimum directions under the stated assumptions. ([GitHub][2])

The conceptual conclusion, however, is:

> The ontology is not identifiable from the observations.

It is **not**:

> Absorption is the information-theoretic truth.

The data do not privilege either ontology. The SAE objective and sparsity regularizer privilege one coding of that distribution. These are three separate claims:

1. **Data identifiability:** whether the latent ontology is recoverable from the activation distribution.
2. **Objective preference:** which dictionary minimizes the SAE objective.
3. **Code identifiability:** whether the encoder's actual codes preserve or separate the distinctions.

The paper handles this better than parts of `report.md`; standardize the distinction everywhere.

---

## 7. Present the gating-corrected estimator as an oracle-scoped identity, not an operational estimator

Round 9 is methodologically strong. The preregistration explicitly excludes detection, automatic pair location, validated orientation and end-to-end label-free operation. Given a qualified and correctly oriented pair, the dominance-partition estimator recovered the parent-event prevalence with MAE at most 0.0026 across all mechanism cells. The operational versions were inconclusive overall, and P3 was retained as falsified. ([GitHub][13])

That supports:

> "Given an oracle-located and oriented pair, tokenwise dominance corrects the gating/counting bias in these harnesses."

It does not yet support:

> "We can estimate absorbed-feature prevalence from a real SAE."

Additional qualifications:

* The near-zero error against realized prevalence is partly a validation of the supervised token-partition rule under low inversion rates.
* The background-mixture explanation is a same-run diagnostic, not a preregistered successful prediction.
* Some clean-gating P4 cells have only 1-10 seeds with adequate class-11 denominators; the summary discloses this, but "decisive in all 16 cells" should be avoided.
* Automatic orientation remains an unresolved bottleneck. ([GitHub][14])

The Round 9 handling of P3 is honest: the post-hoc eligibility diagnosis explains why the margin failed but does not change the falsified verdict.

---

## 8. Downgrade the semi-synthetic detector evidence to exactly what it demonstrates

Rounds 7-8 establish that the detector can identify **planted pairs** in matched synthetic and semi-synthetic settings. They do not establish detection of natural absorption.

The limitations are unusually clear in the result files:

* Round 8 is same-model, same-layer, same-corpus and same-injection-family resampling, not cross-domain transfer.
* Every faithful-control SAE has at least one full-scan flag; total candidate counts do not distinguish absorbed and faithful conditions.
* Detection degrades at parent-child cosine 0.5.
* Rarity-based orientation fails completely under prevalence inversion.
* The later natural-feature adjudication found that **0 of 15** stable candidates met the asymmetric-nesting criterion; they were correlated or anti-correlated feature families, not validated absorption. ([GitHub][15])

This makes the detector a good synthetic mechanism diagnostic and a potentially interesting research lead. It is not yet evidence for natural absorption prevalence.

Round 8's amendment was made before result collection but while the run was in flight. That is not evidence of manipulation, but it is weaker than a clean pre-run immutable lock and should be described as such rather than simply "pre-results." ([GitHub][16])

---

## 9. Preserve the preregistration record, but improve the repository-wide audit trail

The overall honesty discipline is real.

* Round 9 names lock `b0276cc`, uses a frozen scorer, reports P1O/P2O as inconclusive and P3 as falsified, and retains later numerical corrections without changing verdicts.
* Round 10 names lock `f2e92fc`, reports a largely negative round, explicitly separates preregistered verdicts from post-hoc diagnosis, and does not claim that the trained experiment verified the oracle theorem.
* Round 11 is explicitly marked exploratory rather than confirmatory. ([GitHub][14])

The main honesty problem is not suppression; it is **interpretive overshoot after an exploratory result**, especially Round 11's "robust ratio" and "supports the original hypothesis."

Repository-wide provenance is also incomplete:

* `README.md` says there have been ten rounds while the plan contains Round 11.
* Its provenance summary does not cleanly extend through Rounds 9-11.
* `PAPER.md` claims to be "current through round 8" while containing Round 9 material and Round 10 discussion.
* `RESEARCH_PLAN.md` says "updated post round-10" while including Round 11. ([GitHub][17])

Create one immutable claim ledger with columns for claim, prereg commit, amendment commit, result commit, scorer hash, verdict, post-hoc analyses and manuscript location.

The Gemini/GPT reviews should consistently be described as **LLM-assisted adversarial review**, not "external review" without qualification. They improved the work, but they are not independent human peer review.

---

## 10. Tighten the novelty claims and correct the bibliography

### Genuinely new or plausibly new

The strongest novel contributions appear to be:

1. The finite-(\lambda), child-solo-dependent pure-candidate crossover in this specific nonnegative-(L_1) hierarchy.
2. The explicit wall proof tying observational equivalence to the SAE objective's active-direction optimum.
3. The anti-rotated orthonormal-frame construction showing the blind spot of decoder-angle-only penalties.
4. The finite-(\lambda) occurrence boundary and asymptotic (\sqrt2) ratio, within the restricted class.
5. The dominance-partition prevalence identity for oracle-located gated pairs.
6. Possibly the geometry-plus-two-sided-cofiring pair detector as a label-free post-hoc diagnostic-although it presently lacks natural-feature validation.

These are enough for a focused technical note if precisely scoped.

### Not new

* Feature absorption itself, its first-letter manifestation and causal mediation were introduced and quantified by Chanin et al. ([arXiv][18])
* Feature splitting, absorption's dependence on width/sparsity, and the inadequacy of naive SAE tuning were already documented.
* Matryoshka SAEs already report large reductions in first-letter absorption and splitting relative to BatchTopK-for example, 0.05 versus 0.49 absorption at (L_0=40) in their reported setting. ([arXiv][19])
* OrtSAE already proposes decoder orthogonality as a remedy and reports a 65% absorption reduction. Your theorem can show why such a remedy lacks a universal guarantee without implying that its empirical benefits are impossible. ([arXiv][20])
* C2R directly studies splitting and absorption, argues that per-sample TopK as well as (L_1) can prefer pathological representations, supplies a conditional theoretical guarantee, and evaluates TopK, BatchTopK, Matryoshka and OrtSAE using real-model and causal metrics. That is the most important direct contemporary comparison. ([arXiv][21])
* The broader proposition that SAEs do not identify canonical atomic units is already well established by stitching and meta-SAE results. ([arXiv][22])
* Feature Hedging already studies correlated-feature merging in narrow SAEs. Your continuous tilt is best described as a **solvable toy analogue or possible mechanism**, not "the exact mechanism" of all empirically reported hedging. ([arXiv][23])

There are concrete citation errors in `PAPER.md`:

* "Minder et al., Feature Hedging" should be **Chanin, Dulka and Garriga-Alonso**.
* C2R's title is **Cross-sample Consistency Regularization**, not "Consistency-Contrast Regularization."
* The manuscript's "exact mechanism for feature hedging" language is too strong given the different model and phenomenon. ([GitHub][2])

---

## 11. Rebuild `PAPER.md` around one contribution instead of eleven rounds

`PAPER.md` is currently closer to a compressed repository summary than a publishable paper, while `report.md` is a useful but sprawling research log.

For an arXiv note, the cleanest structure is:

1. Model and assumptions.
2. Observational non-identifiability wall.
3. Exact pure-candidate (L_1) crossover and continuous optimum.
4. Capacity/overcompleteness distinction.
5. Restricted decoder-coherence blind-spot theorem.
6. Carefully selected synthetic evidence.
7. Relation to Chanin, Matryoshka, OrtSAE, C2R and canonical-unit work.
8. Limitations and preregistered negative results in an appendix.

Round 11 should not be in the abstract until corrected. The detector and (\hat\rho) program may deserve a separate empirical paper after real validation; including every branch now dilutes the strongest theoretical result.

Before publication:

* State all parameter domains and active-set assumptions.
* Separate theorem, proposition, numerical verification, SGD observation, planted semi-synthetic evidence and exploratory real evidence typographically.
* Provide complete proofs or proof appendices rather than relying on machine checks.
* Include uncertainty estimates rather than primarily counts and cellwise pass labels.
* Correct the bibliography.
* Make all status markers and round counts consistent. ([GitHub][2])

As it stands, a revised version could make a strong **LessWrong/Alignment Forum technical report**. An arXiv note needs the theory-first restructuring and removal of the Round 11 overclaim.

---

## 12. Make Round 11 independently reproducible

The real-model binaries live in GCS, while GitHub contains only two small JSON files with 15 examples each. There are no committed:

* cryptographic hashes,
* exact artifact sizes,
* persistent evaluation-index files,
* complete pair-level outputs,
* training logs or curves,
* package lockfile/container,
* corpus snapshot hashes,
* model revision hashes.

`ENVIRONMENT.md` is useful for the earlier experiments but does not fully pin the Round 11 transformers/model/data pipeline. ([GitHub][1])

Add an artifact manifest with SHA-256 hashes, model and tokenizer revisions, corpus hashes, training and evaluation indices, full aggregate outputs, and a one-command reproduction environment. The weights need not be in GitHub, but they need stable public access or enough information for another group to reproduce them.

---

# Round-by-round standing

* **Rounds 1-2:** The core toy phenomenon and (\lambda q) scaling are credible. Treat the empirical scaling coefficient as an estimate, not an exact universal law.
* **Rounds 3-6 and capacity reruns:** Support the claim that spare dictionary width can replace harmful absorption with redundant composition in this particular generative family. They do not establish that real SAEs are generally capacity-starved in the same way.
* **Arm A:** The decoder/code distinction and leaky gating result are meaningful. They weaken any account of absorption based solely on decoder geometry.
* **Pair-ID and Round 8:** A technically interesting planted-pair detector proof of concept. Natural-feature adjudication is currently negative.
* **Round 9:** Strong oracle-scoped mechanism result; operational deployment remains unresolved. P3 was honestly falsified.
* **Round 10:** A valuable negative result. It materially revises the practical TopK story and should be treated as such throughout the repo.
* **Round 11:** Useful infrastructure milestone and exploratory anomaly. Not yet evidence of a 27x architecture effect in semantic redundancy or absorption. ([GitHub][24])

# Highest-value next experiment

The queued direction is broadly right, but **detector recalibration, first-letter validation and the confirmatory architecture comparison should be combined into one experiment rather than run sequentially as separate claims.**

The single highest-value experiment is:

> **A preregistered, matched-seed L1-vs-TopK real-SAE comparison whose primary endpoint is causal first-letter absorption on a held-out SAEBench-style dataset, with the label-free detector evaluated only as a secondary predictor of those ground-truth outcomes.**

Minimum design:

1. Train at least five seeds per architecture on the same model/layer, activation corpus, token order, initialization scheme, width and training-token budget.
2. Sweep (\lambda) and (k) and compare matched points on the (L_0)-loss-recovered Pareto frontier, rather than one L1 and one TopK model.
3. Use fixed, persisted held-out evaluation tokens shared by every SAE.
4. Make the primary endpoint the Chanin/SAEBench first-letter absorption metric, including the causal ablation component-not raw pair counts.
5. Measure splitting separately using sparse-probe or cluster-level criteria.
6. Evaluate the current detector blind to labels and report precision, recall and calibration against the first-letter absorption labels.
7. Report pair-level and connected-component-level redundancy.
8. Include at least one strong modern control such as BatchTopK, Matryoshka, OrtSAE or C2R if compute permits. Chanin's task provides ground-truth labels and causal adjudication; SAEBench supplies the broader standard needed to prevent FVU alone from standing in for SAE quality. ([arXiv][18])

That one experiment would answer the repository's most important unresolved question: **does the toy geometry predict a reproducible difference in causally validated feature absorption between real SAE objectives?** It would also validate-or falsify-the label-free detector on exactly the real phenomenon it is intended to measure.

[1]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/results/real/SUMMARY.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/results/real/SUMMARY.md"
[2]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/PAPER.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/PAPER.md"
[3]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/experiments/real_analyze.py "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/experiments/real_analyze.py"
[4]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01085d2695cca530e99d036791ff56511db/results/real/sae_pythia-1.4b_L12_topk_x8_pairs.json?download=text "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01085d2695cca530e99d036791ff56511db/results/real/sae_pythia-1.4b_L12_topk_x8_pairs.json?download=text"
[5]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/experiments/real_train_sae.py "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/experiments/real_train_sae.py"
[6]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/experiments/real_extract.py "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/experiments/real_extract.py"
[7]: https://arxiv.org/html/2503.09532v3 "https://arxiv.org/html/2503.09532v3"
[8]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/theory/general_no_go.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/theory/general_no_go.md"
[9]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/theory/topk_absorption.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/theory/topk_absorption.md"
[10]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/results/round10/SUMMARY.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/results/round10/SUMMARY.md"
[11]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/notes/prereg-topk-absorption.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/notes/prereg-topk-absorption.md"
[12]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/theory/verify_absorption_theory.py "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/theory/verify_absorption_theory.py"
[13]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/notes/prereg-gating-corrected-rho.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/notes/prereg-gating-corrected-rho.md"
[14]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/results/round9/SUMMARY.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/results/round9/SUMMARY.md"
[15]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01085d2695cca530e99d036791ff56511db/results/round8/SUMMARY.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01085d2695cca530e99d036791ff56511db/results/round8/SUMMARY.md"
[16]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01085d2695cca530e99d036791ff56511db/notes/prereg-round8-scaling-robustness.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01085d2695cca530e99d036791ff56511db/notes/prereg-round8-scaling-robustness.md"
[17]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/README.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/README.md"
[18]: https://arxiv.org/html/2409.14507v2 "https://arxiv.org/html/2409.14507v2"
[19]: https://arxiv.org/html/2503.17547v1 "https://arxiv.org/html/2503.17547v1"
[20]: https://arxiv.org/abs/2509.22033 "https://arxiv.org/abs/2509.22033"
[21]: https://arxiv.org/html/2606.30609v1 "https://arxiv.org/html/2606.30609v1"
[22]: https://arxiv.org/abs/2502.04878 "https://arxiv.org/abs/2502.04878"
[23]: https://arxiv.org/abs/2505.11756 "https://arxiv.org/abs/2505.11756"
[24]: https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/RESEARCH_PLAN.md "https://raw.githubusercontent.com/reuellee/sae-identifiability/6248c01/RESEARCH_PLAN.md"

