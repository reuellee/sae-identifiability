# External round-8 review (GPT, via user, verbatim extract)



Review and Revision Brief
Latest SAE Identifiability Research and Round 8 Preregistration
A self-contained prompt-ready review for repository revision
Overall verdictThe core theoretical contribution remains strong. The completed detector experiment provides meaningful but limited semi-synthetic transfer evidence. Round 8 is largely the right next step, but the preregistration should be corrected before its eventual results are treated as confirmatory—especially the candidate-stability matcher, width-specific success criteria, and all-pairs real-background specificity.
Repository reviewed: reuellee/sae-identifiability
Purpose and requested action
Claude: Use this document as an external review of the latest repository state. Revise the report, README, result summaries, preregistration, and analysis code where the recommendations below apply. Preserve the distinction between completed results and preregistered future tests. Do not retroactively redefine completed endpoints as confirmatory successes.
Primary objective: Make the repository’s empirical claims exactly match what the completed Arm 2 evidence establishes, and make Round 8 a valid test of the detector rather than a pooled or implementation-sensitive confirmation.
Current contribution map
Evidence level
Current status
Defensible description
Strong
Analytical toy model and capacity mechanism
A well-supported theoretical and synthetic contribution within its stated assumptions.
Encouraging
Frozen detector on an injected absorbed pair in GPT-2 activations
Partial semi-synthetic transfer of pair detection, with strong width dependence.
Exploratory
Unrestricted candidate discovery, automatic orientation, child recovery, and frequency estimation
Promising components that are not yet validated as a practical label-free identifiability pipeline.
1. Completed Arm 2: what the evidence establishes
Positive result. The frozen detector distinguishes the planted absorbed oracle pair from the planted faithful oracle pair in the semi-synthetic GPT-2 setting. At width m = 256, the absorbed oracle pair was detected in 8/8 runs and the faithful oracle pair was flagged in 0/8 controls. At m = 128, absorbed-pair recall was only 1/8 because the lift statistic clustered just below the registered threshold of 2.0.
Defensible claim: On the planted oracle pair, the detector distinguishes absorbed and faithful configurations in this semi-synthetic setup. Precision and specificity over unrestricted background-pair scans remain unmeasured.
Keep the threshold-2.0 result as the completed preregistered outcome.
Treat a 1.9 cutoff as a new detector version tested on fresh runs, not as a retrospective correction of Arm 2.
Describe Round 8’s fresh 1.9 experiment primarily as a resampling-stability test within the same GPT-2 domain, not broad cross-domain transfer.
2. Do not overstate specificity or natural-candidate validation
Key limitation. The faithful-control result concerns the known oracle pair. It does not show that the detector has good specificity when scanning every pair in an SAE. The committed runs contain similar numbers of full-scan flags in absorbed and faithful conditions, so total candidate count does not itself separate the conditions.
Required wording change: Replace broad language such as “specificity transfers perfectly” with the narrower oracle-pair statement above.
Required naming change: Call the output a “real-background candidate list” or “candidates found in SAEs trained on real activations,” not a validated “natural-absorption candidate list.”
Why this matters: A candidate found against a real activation background is not automatically a naturally occurring absorbed concept. It still requires semantic, causal, cross-seed, or synthetic-ground-truth adjudication.
3. Treat the width effect as calibration dependence
The m = 128 and m = 256 lift distributions move systematically. This is more than a random near-threshold miss. The detector statistic is configuration-dependent, and a single global cutoff has not yet been shown to transfer across width, layer, model, sparsity regime, or architecture.
Recommended language: “The failure is a width-dependent calibration failure. A cutoff of 1.9 would recover these Arm 2 examples, but whether one cutoff transfers across widths, layers, models, sparsity regimes, and SAE architectures remains open.”
4. Separate pair detection from the downstream recovery pipeline
The current end-to-end pipeline has multiple logically distinct stages:
Detect a suspicious pair.
Determine which member is the parent and which is the composite.
Recover a residual child direction.
Estimate the relevant event frequencies.
Use the inferred structure in an intervention that improves SAE recovery.
Current status. Only the first stage has encouraging transfer evidence. The “rarer latent is composite” rule was correct for roughly five of nine detected planted pairs. Child-direction cosine was near 0.99 when orientation was correct and around 0.66 when it was wrong. The frequency estimator returned roughly 0.75 for a true injected value of 0.5.
Report orientation accuracy conditional on detection as a separate endpoint.
Report child-residual cosine under both automatic orientation and oracle orientation.
Do not present frequency estimation as a practical replacement for oracle weighting yet.
5. Statistical interpretation
Small-sample caution. Eight successes in eight trials is encouraging, but it does not tightly establish a very high population recall. Round 8’s larger sample is justified, and the report should avoid equating a perfect observed proportion with a precise population estimate.
Provide width-specific confidence intervals for recall and faithful-control flag rates.
Keep pooled summaries secondary when a strong width interaction is present.
Distinguish an observed endpoint pass from evidence that the underlying rate exceeds the target with high confidence.
6. Round 8 preregistration: required changes before interpretation
Highest priorityFix the cross-seed candidate-stability matcher before generating or interpreting stability results. The current implementation can miss stable candidates absent from seed 0 and can match both members of one pair to the same direction in another pair.
6.1 Require width-specific success
Problem: A pooled criterion across m = 128 and m = 256 can conceal continued failure at one width, even though Arm 2 showed a nearly perfect width stratification.
Require recall_128 >= r0 AND recall_256 >= r0,or define both widths as separate confirmatory endpoints.
Do not let a pooled pass override a failed width-specific endpoint.
6.2 Add operational all-pairs specificity
The oracle faithful-pair control is necessary but insufficient. Round 8 should quantify what happens when the detector scans the entire SAE.
Flags per million candidate pairs.
Proportion of faithful or null SAEs with at least one flag.
Precision after synthetic or manual adjudication.
A real-background null condition with no injected hierarchical pair.
Candidate-generation cost and false positives as SAE width grows.
6.3 Separate width from scale and include overcompleteness
Problem: The current scale sweep changes ambient dimension, background-feature count, and SAE width together. It therefore tests an overall scale family but cannot identify which variable drives any change. The proposed settings also remain undercomplete relative to production-style SAEs.
Add a fixed-dimension, fixed-background sweep over SAE width.
Retain a proportional scale sweep as a separate experiment.
Include at least one overcomplete regime with m > d.
6.4 Fix the candidate-stability matcher
Current defects:
Seed-0 anchoring: a candidate present in seeds 1–7 but absent from seed 0 can never be declared stable.
Non-bijective matching: both directions of the reference pair may be matched to the same direction in another pair.
Minimum pair score: compare the two valid one-to-one assignments and use the better assignment:
score = max(    min(|d_i · e_k|, |d_j · e_l|),    min(|d_i · e_l|, |d_j · e_k|))
Preferred approach: Build a graph or clustering procedure over all candidates from all seeds, with at most one candidate per seed in a cluster, rather than choosing a single seed as the universal reference.
6.5 Make orientation a confirmatory endpoint
Orientation should not be relegated only to an exploratory prevalence stress test. It is essential to the claimed child-recovery pipeline and already failed frequently in Arm 2.
Pair-detection recall.
Orientation accuracy conditional on detection.
Child-residual cosine under automatic orientation.
Child-residual cosine under oracle orientation.
Frequency-estimation error conditional on correct and incorrect orientation.
7. Documentation and implementation consistency
Update the Round 8 real-data script docstring: it still describes eight seeds and 32 runs, while the implementation uses the expanded design.
Ensure the README, main report, result summary, preregistration, and executable scripts all state the same seed counts, widths, thresholds, and endpoint hierarchy.
Mark completed Arm 2 results, detector v1.2 changes, and Round 8 tests with explicit pre-result commit hashes.
Label any change derived from Arm 2 observations as development, not confirmatory evidence.
8. Suggested replacement language
For the Arm 2 result:
“A detector frozen on synthetic development data identified the planted absorbed oracle pair in a semi-synthetic GPT-2 activation setting, with strong width dependence: 1/8 detections at m = 128 and 8/8 at m = 256 under the preregistered lift threshold. The planted faithful oracle pair was not flagged. This supports partial transfer of the pair-detection signal, but does not establish all-pairs specificity, automatic orientation, or natural-feature absorption.”
For candidate discovery:
“The full scan yields real-background candidates requiring further validation. Their status as natural absorbed pairs is unknown.”
For Round 8:
“Round 8 tests fresh-run stability of the revised detector within the same GPT-2 domain, width-specific calibration, scaling behavior, robustness to altered geometry and encoder choice, and cross-seed candidate recurrence. It is not by itself a test of transfer across models, layers, corpora, or SAE families.”
9. Prioritized revision checklist
P0  Fix the cross-seed matcher before running or interpreting stability analysis.
P0  Make width-specific recall and faithful-control behavior confirmatory endpoints.
P0  Add full-scan real-background specificity metrics and a no-injection real-background null.
P1  Separate detection, orientation, child recovery, and frequency estimation in all reporting.
P1  Add fixed-dimension width sweeps and at least one overcomplete setting.
P1  Narrow claims about specificity, transfer, and natural candidates.
P2  Synchronize docstrings, README, report, preregistration, and provenance metadata.
10. Final assessment
The latest research is a meaningful advance. It shows that a detector developed on synthetic data can recognize an injected absorbed pair in real-model activations under some widths, and it exposes calibration, orientation, and specificity failures rather than hiding them. That is good scientific progress.
Final verdict: Keep the analytical and capacity contributions as the repository’s strongest claims. Present Arm 2 as a frozen-threshold semi-synthetic transfer test. Treat unrestricted candidate discovery and the downstream recovery pipeline as exploratory. After correcting the Round 8 matcher and endpoint design, Round 8 can become a credible test of whether the detector is evolving from a matched synthetic diagnostic into a practical empirical method.
Repository materials referenced in this review
results/prereg_pairid/SUMMARY.md
results/prereg_pairid/arm2_runs.csv
notes/prereg-round8-scaling-robustness.md
analysis/s1_candidate_stability.py
experiments/round8_realdata.py
report.md
README.md
Repository: github.com/reuellee/sae-identifiability