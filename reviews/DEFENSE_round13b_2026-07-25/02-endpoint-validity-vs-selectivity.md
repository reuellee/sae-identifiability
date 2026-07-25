# Defense round13b — Q2: endpoint-validity-vs-selectivity

_Examiner: Gemini 2.5 Pro via gemx. Generated 2026-07-25T16:36:25Z. Single-shot._

# Thesis Defense Interrogation — Endpoint Validity

Candidate, your reliance on the first-letter "absorption rate" as a valid scientific proxy for mechanistic feature absorption is mathematically and conceptually compromised. Under close scrutiny, your metric does not isolate feature absorption; it is primarily a re-expression of encoder-level threshold suppression and feature splitting.

---

### PART I: ADJUDICATION OF THE ENDPOINT

#### Finding 1: The First-Letter Absorption Endpoint is a Selectivity and Threshold-Suppression Artifact
*   **The Claim:** 
    > "Scoring each letter against its whole split family instead of one designated latent removes 25.2% of the measured absorption. That quarter was splitting. The remaining ~75% is not: it is instances where the letter is still linearly present in the reconstruction and no sufficiently-selective latent for that letter fires at all." (from `results/real/SUMMARY_round13a.md` and integrated into `PAPER.md`)
*   **Why it fails / is unsupported:** 
    The assertion that the remaining 75% is "genuine absorption" is an empirical leap. The family-corrected endpoint (`rate_fam`) still shares $R^2 = 0.381$ (38.1% of its variance) with the maximum selectivity of the family. A letter trial is flagged as "absorbed" whenever *no* latent in the split family $F_L$ fires, despite the letter's linear presence in the reconstruction. 
    However, this failure to fire is not diagnostic of mechanistic absorption into a parent latent (the $a_m = (a_p + a_c)/\sqrt{2}$ composite optimal state). It can be trivially caused by:
    1. **Sparsity Shrinkage:** The $L_1$ penalty or TopK selection suppresses weak activation signals below the encoder threshold (especially under real-world model noise).
    2. **Reconstruction Degradation:** Standard reconstruction loss at low capacity ($m = 2048$), where features are simply *lost* or blurred, not absorbed (the "retention confound" you register in P5).
    Your metric lacks any decoder-level or activation-level verification showing that the letter's reconstruction on these trials is actually being carried by a broader parent/composite latent. You are calling "feature loss due to thresholding" by the more interesting name of "absorption."
*   **Specific Evidence / Experiment to settle it:** 
    To prove this is mechanistic absorption, you must perform a residual projection check on the "absorbed" trials: project the reconstruction residual onto the parent-latent decoder columns. True absorption requires that the parent latent's activation increases to reconstruct the child's missing mass, whereas simple threshold suppression/loss will show no such compensation.
*   **Severity:** **MAJOR**
*   **Classification:** `[POST-HOC-ONLY]` (Resolving this requires exploratory post-hoc analysis of the existing activation and weight artifacts from Rounds 12 and 13a/b).

---

### PART II: THE CAUSAL & ALGEBRAIC COUPLING (COLLIDER BIAS)

Your decision to **discard** the regression of absorption rate on architecture controlling for selectivity was **100% mathematically correct**. Resurrecting it would be a severe error of statistical inference.

Let us prove that this regression is an algebraic trap and a textbook case of **collider conditioning**.

#### The Algebraic Proof of Collider Bias
Let:
*   $A$ be the treatment variable (Architecture: L1 vs. TopK).
*   $T = P(\text{fire}_j \mid L)$ be the True Positive Rate of the main latent on letter trials.
*   $F = P(\text{fire}_j \mid \neg L)$ be the False Positive Rate of the main latent on non-letter trials.
*   $Y = 1 - T$ be the outcome (the single-latent absorption rate).
*   $S = T - F$ be the covariate controlled for (selectivity).

The linear regression of $Y$ on $A$ controlling for $S$ is:
$$Y = \beta_0 + \beta_1 A + \beta_2 S + \epsilon$$

By definition, we have the deterministic algebraic identity:
$$S = T - F \implies S = (1 - Y) - F \implies Y + S = 1 - F$$

Rearranging for $Y$:
$$Y = 1 - S - F$$

When you control for selectivity ($S$ is held constant), the partial derivative of $Y$ with respect to the treatment $A$ is restricted to:
$$\left. \frac{\partial Y}{\partial A} \right|_{S} = -\frac{\partial F}{\partial A}$$

#### Causal DAG Interpretation
The causal pathways are:
$$A \to T \to Y$$
$$A \to F \to S \leftarrow T$$

```
    [A]
   /   \
  v     v
 [T]   [F]
  |  \ /
  v   v
 [Y] [S]
```

In this system:
1.  **$S$ is a collider** between $T$ (true positive rate) and $F$ (false positive rate).
2.  Conditioning on $S$ opens a non-causal backdoor path: $A \to F \to [S] \leftarrow T \to Y$.
3.  Because of the algebraic constraint $Y = 1 - S - F$, any variation in $Y$ at a fixed $S$ is purely driven by $-F$ (the negative false positive rate).

Thus, the coefficient $\beta_1$ does **not** measure the causal effect of architecture on feature absorption ($1-T$). It measures the effect of architecture on the **negative false positive rate** ($-\text{FPR}$). Because L1 and TopK differ systematically in feature splitting (L1 has a higher FPR due to broader, split latents), controlling for selectivity forces a sign flip that is a pure mathematical artifact of the FPR difference.

Your defense stands on this point: **resurrecting that regression would be scientific collider conditioning.**

---

### PART III: WHAT THE REAL-DATA ARC ESTABLISHES

If we strip away the conflation with feature splitting (25.2% of the signal) and threshold suppression ($R^2 = 0.381$ with selectivity), your real-data arc establishes exactly three empirical facts:
1.  **Metric Disproof:** The standard, single-latent absorption metrics in current literature (such as those in SAEBench) are severely confounded by feature splitting. 
2.  **Splitting Divergence:** Under an $L_1$ penalty, Pythia-1.4B splits the first-letter feature across twice as many latents as TopK ($|F_L| = 2.61$ vs. $1.25$).
3.  **Null Causal Architecture Effect:** When you correct for feature splitting using your family-based metric, L1 and TopK absorb at the **exact same rate** (paired difference of $-0.0012$, CI $[-0.0081, +0.0049]$).

Your real-data arc does **not** establish that first letters are mechanistically absorbed into parent concepts in Pythia-1.4B. It is a highly valuable, rigorous **methodological negative result** proving that previous reports of architectural differences in absorption were likely illusions of feature splitting, and that at large capacity ($m=16384$), both architectures perform identically on this axis.
