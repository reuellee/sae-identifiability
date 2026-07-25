# Defense round13b — Q5: statistical-inference-seeds-and-multiplicity

_Examiner: Gemini 2.5 Pro via gemx. Generated 2026-07-25T16:39:57Z. Single-shot._

**FINDING 1: Seed Bootstrap Generalization vs. Data Variance**
(a) The claim: "child recovery median $\cos = 0.979$ $[0.976, 0.983]$ (the only CI-established endpoint)" and Round 13b's reliance on a "10k bootstrap over seeds" to establish generalization.
(b) Why it fails: Bootstrapping exclusively over SGD initialization seeds while holding the activation cache (e.g., `acts_eval.pt`) fixed calculates the variance of the training algorithm *conditional on this exact finite dataset*. It provides exactly zero coverage for the variance of the data distribution. Presenting these as general statistical confidence intervals implies you have bounded the effect across the true underlying distribution. You have not; you have only bounded the effect across different random initializations on one finite text sample.
(c) Evidence/Experiment to settle: A two-way cluster bootstrap that resamples both the random initialization seeds *and* the underlying activation sequences to estimate the true total variance.
(d) Severity: MAJOR.
Classification: [POST-HOC-ONLY]

**FINDING 2: Multiplicity and Uncorrected Family-Wise Error Rate**
(a) The claim: "every confirmatory experiment in this project was pre-registered... failures are reported as first-class results."
(b) Why it fails: Pre-registration fixes the endpoints and prevents post-hoc threshold moving, but it does not suspend the laws of probability. Across this program (Rounds 8, 9, 10, 12, 13a, 13b), you have tested dozens of "primary" and "secondary" hypotheses (P1–P5 per round). Without a Family-Wise Error Rate (FWER) or False Discovery Rate (FDR) correction, finding "CI excludes 0" is a statistical guarantee over enough tests, not a scientific discovery. Your nominal 95% coverage is a fiction at the program level. 
(c) Evidence/Experiment to settle: Group all primary pre-registered endpoints in the project into a single hypothesis family and apply a standard multiplicity correction (e.g., Holm-Bonferroni or Benjamini-Hochberg) to the final reported confidence intervals and pass/fail thresholds.
(d) Severity: MAJOR.
Classification: [PRE-RESULTS-OK]

**FINDING 3: Degenerate 8-Seed Difference-in-Differences Bootstrap**
(a) The claim: "Interaction: `[L1−TopK]@m=2048 − [L1−TopK]@m=16384`, paired by seed, 10k bootstrap. CONFIRMED if CI lower > 0." (Round 13b P2).
(b) Why it fails: A non-parametric bootstrap on $n=8$ independent units is mathematically degenerate. The empirical CDF only has 8 discrete steps, meaning the 2.5th and 97.5th percentiles are entirely determined by the sample extrema. You cannot extract a valid 95% confidence interval from the order statistics of 8 points. Worse, this is a difference-of-differences estimator, which aggressively compounds the variance across four experimental cells. Resampling 8 data points 10,000 times does not synthesize statistical power; it merely draws a high-resolution picture of sampling noise. A "CONFIRMED" verdict based on the lower bound of an $n=8$ diff-in-diff bootstrap is statistically meaningless.
(c) Evidence/Experiment to settle: Specify a parametric hierarchical model (e.g., repeated-measures ANOVA) justified by residual diagnostics, or increase the seed count to $n \ge 30$ where bootstrap asymptotics actually begin to hold. 
(d) Severity: BLOCKING.
Classification: [PREREG-VIOLATION]
