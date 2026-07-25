# Defense round13b — Q1: central-null-informative-or-underpowered

_Examiner: Gemini 2.5 Pro via gemx. Generated 2026-07-25T16:34:32Z. Single-shot._

# Defense Interrogation — The Central Null (Rounds 12 & 13a)

As the Hostile External Examiner, I must inform the candidate that their "central null" result is not an informative, high-resolution scientific finding. It is a textbook example of an underpowered study masquerading as a precise null due to a fundamental misunderstanding of the units of statistical replication.

By treating optimization seeds as the primary unit of replication and bootstrapping over a sample size of $N=8$, you have constructed a "fixed-effects" confidence interval with respect to the data. This interval measures only the variance of the optimization process (initialization noise) while completely ignoring the variance over the linguistic features (letters) and the data. When the variance is correctly calculated over the actual features of interest, the statistical power collapses, and the null is revealed to be entirely uninformative.

---

### PART I: DISSECTING THE LOAD-BEARING CLAIMS

#### Finding 1: The "Informative Null" Claim is a Statistical Illusion Built on the Wrong Unit of Replication
*   **The Claim:**
    > "The null is informative. At the registered seed-level unit the CI excludes an L1−TopK gap larger than ±25% of the 0.055 base rate. This is not an underpowered null in sample size." (results/real/SUMMARY_round12.md)
*   **Why it fails / is unsupported:**
    This assertion is statistically fallacious on multiple fronts:
    1.  **Wrong Unit of Replication:** The "seed-level unit" measures variance across random initializations of the SAE optimization on the **exact same cached activation set** and evaluated on the **exact same words**. Bootstrapping over seeds only resamples the *optimization noise* of your algorithm. It treats the linguistic features (the 24 letters) and the dataset as fixed and known without error. This is a "fixed-effects" fallacy; it has zero generalization power to other letters, tasks, or datasets.
    2.  **The Feature-Level Reality:** The true linguistic features of interest are the $N=24$ letters. Because absorption is wildly heterogeneous across letters (ranging from $0.004$ to $0.197$ for L1 and $0.000$ to $0.231$ for TopK), the variance across features is massive. When you correctly pair and cluster by letter to generalize over the linguistic task, the 95% confidence interval for the difference is:
        $$\text{Letter-level difference (n=24, seed-paired): } +0.0050 \quad 95\% \text{ CI } [-0.0096, +0.0189]$$
    3.  **Failure of the 25% Exclusion:** Your base absorption rate is $0.0554$. An effect of $25\%$ of this base is $\pm 0.0139$. Your letter-level confidence interval $[-0.0096, +0.0189]$ spans a width of $0.0285$ (**$51.4\%$ of the base rate**) and its upper limit $+0.0189$ is **$34.1\%$ of the base rate**. You *cannot* exclude a gap of $25\%$ of the base rate at the feature level.
    4.  **Infeasibility of $N=8$ Bootstrap:** Bootstrapping a sample of size $N=8$ is mathematically invalid. Bootstrapping is an asymptotic method. At $N=8$, there are only $\binom{8+8-1}{8} = 6435$ possible unique bootstrap samples. The resulting bootstrap distribution is highly discrete and severely underestimates the true sampling variance, leading to an artificially narrow confidence interval.
*   **Specific Evidence / Experiment to settle it:**
    You must report the confidence intervals paired and clustered by **letter** rather than seed, and perform a proper two-one-sided-test (TOST) equivalence analysis at the letter level. If the letter-level CI cannot exclude a $25\%$ gap (which your post-hoc diagnosis shows it cannot), the claim that the null is "informative" and "not underpowered" is refuted and must be withdrawn.
*   **Severity:** **MAJOR**
*   **Classification:** `[PRE-RESULTS-OK]` (This requires no new training or locked analysis; the letter-level data has already been collected and is analyzed in your post-hoc diagnosis; the statistical reporting and interpretation in your summary must be corrected immediately).

---

#### Finding 2: "Slipping the Null" — Accepting the Null Hypothesis under Severe Underpowering
*   **The Claim:**
    > "L1 vs TopK differ sharply — but in **splitting** (2.61 vs 1.25 latents/letter), not in absorption (0.0536 vs 0.0548)." (results/real/SUMMARY_round13a.md)
*   **Why it fails / is unsupported:**
    This statement makes a positive assertion of equivalence ("differ ... not in absorption") based on a failure to reject the null hypothesis. This is a classic statistical error: asserting the null (accepting $H_0$) based on a lack of statistical significance ($p > 0.05$), while ignoring that the study is severely underpowered at the feature level. 
    A non-significant result in an underpowered study is simply an "absence of evidence," not "evidence of absence." You cannot claim that L1 and TopK do not differ in absorption when your feature-level confidence interval is wide enough to hide an effect size equal to $34\%$ of your entire base rate.
*   **Specific Evidence / Experiment to settle it:**
    To positively claim that the architectures "do not differ in absorption," you must run a power analysis and design an equivalence study (using TOST) over a much larger set of features (e.g., sweeping across multiple layers, models, or tasks to increase the number of independent features $N$) with a pre-registered equivalence margin (e.g., $\pm 10\%$ of the base rate). Lacking this, you must soften your language to state that "no statistically significant difference was detected" and explicitly disclose the wide letter-level confidence interval as a major limitation of your statistical power.
*   **Severity:** **MAJOR**
*   **Classification:** `[PRE-RESULTS-OK]` (This is a matter of correcting scientific framing and claims in the summaries and paper draft without modifying any locked data or running new experiments).

---

### PART II: STATISTICAL POWER & MDE ANALYSIS

To demonstrate why your design is underpowered to detect scientifically interesting effects, let us calculate the **Minimum Detectable Effect (MDE)** of your current experimental design.

#### 1. MDE over Seeds ($N = 8$)
If we treat the seed as the unit of replication, your standard error ($\text{SE}$) is approximately $0.0021$ (derived from your seed differences: $s \approx 0.0059$, $\text{SE} = s / \sqrt{8} \approx 0.0021$).
At $80\%$ power and a $5\%$ significance level ($\alpha = 0.05$) for a paired two-tailed t-test with $df = 7$:
$$\text{MDE}_{\text{seed}} = (t_{\alpha/2, 7} + t_{\beta, 7}) \times \text{SE} \approx (2.365 + 0.896) \times 0.0021 \approx 0.0068$$
Expressing this as a percentage of your base rate ($0.0554$):
$$\frac{\text{MDE}_{\text{seed}}}{0.0554} \approx 12.3\%$$
*While a $12.3\%$ MDE appears small, it only represents the power to detect differences in optimization noise on this specific dataset.* It holds no scientific generalizability.

#### 2. MDE over Features / Letters ($N = 24$)
To make a general scientific claim about Pythia-1.4B's feature absorption, the letter (the linguistic feature) must be the unit of replication.
From your post-hoc letter-paired CI of $[-0.0096, +0.0189]$ (width $= 0.0285$):
$$\text{SE}_{\text{letter}} \approx \frac{\text{CI Width}}{2 \times t_{0.025, 23}} \approx \frac{0.0285}{2 \times 2.069} \approx 0.0069$$
At $80\%$ power and a $5\%$ significance level for $df = 23$:
$$\text{MDE}_{\text{letter}} = (t_{\alpha/2, 23} + t_{\beta, 23}) \times \text{SE}_{\text{letter}} \approx (2.069 + 0.858) \times 0.0069 \approx 0.0202$$
Expressing this as a percentage of your base rate ($0.0554$):
$$\frac{\text{MDE}_{\text{letter}}}{0.0554} \approx \frac{0.0202}{0.0554} \approx \mathbf{36.5\%}$$

#### 3. Verdict
An experimental design that can only detect architectural differences larger than **$36.5\%$ of the base rate** is **severely underpowered** by any standard of experimental statistics. 

Your seed-level CI is an artifact of holding the data completely constant across both treatment arms. It is a mathematical sandbox. Your slide from "failed to reject" to "there is no difference" is an overclaim that will not survive peer review. 

Candidate, how do you defend publishing an "informative null" claim when your feature-level MDE is a massive $36.5\%$?
