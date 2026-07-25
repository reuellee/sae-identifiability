# Defense round13b — Q7: generalisation-single-model-single-task

_Examiner: Gemini 2.5 Pro via gemx. Generated 2026-07-25T16:44:03Z. Single-shot._

I am starting our thesis defense investigation by scanning the repository for any round-13b and round-12 files, ensuring we have a complete picture of the empirical results and pre-registrations before formulating the critique.
I am transitioning to the strategic critique of the generalizability claims. This phase will focus on structuring a devastating, peer-review-grade critique of the paper's empirical sandbox, detailing the limits of its statistical generalizability, exposing the structural flaws in the first-letter metric, and identifying specific over-broad phrasings that must be corrected.
# Thesis Defense Interrogation: The Mirage of Generalizability

**Examiner:** Hostile External Examiner  
**Focus Area:** Empirical Scope, Construct Validity, and Representation Mechanics in Pythia-1.4B L12  

Candidate, we must strip away the self-congratulatory narrative of your pre-registration discipline and confront the physical reality of your data. You have presented a document that attempts to bridge a 2D undercomplete synthetic sandbox with real-scale language-model representations. Under close scrutiny, this bridge collapses. Your empirical findings do not generalize; they are highly localized artifacts of a specific model, a specific layer, a highly artificial typographic task, and a construct-compromised metric.

Below is the strongest case that nothing in your real-data arc generalizes, followed by an honest, fair bounding of what survives, and a detailed audit of your paper's overclaims.

---

## 1. The Sandbox Critique: Why Nothing Here Generalizes

The empirical portion of your thesis rests on a highly narrow, multi-layered sandbox. Here is the case for why your findings are trapped inside it:

### A. The "Fixed-Effects" Fallacy: One Model, One Cache, Eight Seeds
Your statistical apparatus treats optimization seeds as the primary unit of replication. By bootstrapping over $N=8$ seeds, you are resampling only the **initialization noise of the SAE training algorithm** while holding the underlying activation distribution completely constant. 
*   **The Data is Fixed:** You train and evaluate on the exact same cached activation file (`acts_train.pt`, `acts_eval.pt`) drawn from a single corpus. This is a classic "fixed-effects" fallacy in experimental statistics. You have calculated confidence intervals conditional on this exact text sequence. You have zero statistical coverage over dataset variance, out-of-distribution shifts, or domain changes. 
*   **The Model is Frozen:** You evaluate a single model (Pythia-1.4B). Models of different sizes, trained on different datasets, or with different embedding-to-residual ratios have wildly different noise floors, feature densities, and geometric alignments. Your results are a single-point measurement in model space.

### B. Layer-12 Residual Stream: The Unrepresentative Additive Bus
You present your findings on **Layer 12 of Pythia-1.4B** (the exact midpoint of a 24-layer transformer) as if they reveal general sparse representation dynamics. They do not. 
*   **The Mid-Layer Communication Bus:** The residual stream of a mid-layer transformer is not a repository of clean, localized, final features. It operates as an additive communication bus where attention heads and MLP layers continuously write, read, update, and erase intermediate computations. To prevent interference, features on this bus are aggressively distributed and "hedged" (split) across multiple directions.
*   **Splitting Dominance:** This specific communication role explains why feature splitting is so dominant in your Layer 12 L1 SAEs, where a single letter feature is split across an average of **2.61 latents** (`|F_L|`). Early layers (which process clean typographic/token inputs) and late layers (which compile stable semantic concepts) are not subject to the same high-throughput communication pressures. Their representation geometry is more localized, meaning your mid-layer splitting results are completely unrepresentative of early or late transformer representations.

### C. The First-Letter Task: Typographic Assembly, Not Semantic Concepts
Your "first-letter" task is fundamentally not a semantic or conceptual feature. It is a **typographic character-assembly task** forced to operate in a sub-word tokenized language model.
*   **Tokenizer Contradiction:** A model like Pythia-1.4B does not natively "see" characters; it sees tokens. For a token like ` "starts"`, the model must actively reconstruct its constituent character spelling. The features you are tracking ("starts with S") are typographic sub-token artifacts. 
*   **Severe Signal Heterogeneity:** This task is wildly non-uniform. Your own post-hoc analysis (`round12_posthoc_diagnosis.txt`) reveals that **just three letters (`s`, `r`, and `c` / `p`) carry up to 73% of your entire measured absorption signal**. This means your "general" first-letter absorption rate of $5.4\%$ is actually an average driven by a tiny handful of outlier typographic features. You are not measuring a general feature absorption pathology; you are measuring how Pythia-1.4B represents the letters `s`, `r`, and `c` in Layer 12. This tells us absolutely nothing about how a model represents abstract, hierarchical semantic concepts.

---

## 2. Methodological Rot: The First-Letter Metric as a Construct

Even if we accept your sandbox, your primary operational tool—the first-letter "absorption rate" ($1 - \text{Selectivity}$ or $1 - \text{Family Selectivity}$)—is a compromised proxy for mechanistic absorption.

### A. The Splitting Confound (The 25.2% Illusion)
The standard single-latent absorption metric in current use (e.g., in SAEBench) flags a letter as "absorbed" if the single highest-selectivity latent fails to fire. In Round 13a, you demonstrated that **25.2% of this signal was a pure feature-splitting artifact**:
*   The letter feature did not merge into a parent. It was simply represented by *other* split latents in the dictionary.
*   Because L1 SAEs split features twice as much as TopK SAEs ($|F_L| = 2.61$ vs. $1.25$), any comparison of L1 and TopK using a single-latent metric is heavily confounded by this architectural difference. Once you corrected for splitting using your `rate_fam` metric, the apparent architectural difference collapsed to a dead null (paired difference $-0.0012$, CI $[-0.0081, +0.0049]$).

### B. Threshold Suppression vs. Mechanistic Absorption
Even your family-corrected metric (`rate_fam`) is not a clean measure of absorption. It flags a trial as "absorbed" whenever *no* latent in the entire split family $F_L$ fires, despite the letter's linear presence in the reconstruction. 
*   **Sparsity Shrinkage & Loss:** You cannot distinguish between true mechanistic absorption (where the child's activation is captured by a broader parent/composite latent) and simple **threshold suppression** or **reconstruction loss**. 
*   Under capacity scarcity ($m = 2048$), weaker activation signals are suppressed below the encoder's activation threshold ($\theta = 0.05$) or are lost due to MSE-driven feature pruning. 
*   Your metric lacks any decoder-level or activation-level verification (e.g., a residual projection check showing that the parent latent's activation compensated for the missing child). You are labeling "feature loss due to thresholding and low capacity" with the more prestigious name of "hierarchical feature absorption."

---

## 3. Fair Adjudication: What Actually Survives?

To be fair, we must not manufacture objections where your work is robust. If we strip away the overclaims, your paper contains a highly rigorous, valuable set of findings that **are appropriately scoped and fully survive this criticism**:

1.  **The Conceptual Toy Theory (§1–§13):** Your analytical derivations of the exact crossover boundary $\varepsilon^*(\lambda, q) \approx 1.17 \lambda q$, the coherence penalty limit $p_0^*$, and the "anti-rotation" evasion are mathematically correct under their stated conditions (undercomplete, orthonormal 2D plane). They are valuable as a mathematical isolation of the geometric forces competing during dictionary learning.
2.  **The Splitting Divergence:** Your finding that L1 SAEs split features significantly more than TopK SAEs ($2.61$ vs. $1.25$ latents per feature, supported by $22/24$ letters) is extremely robust, statistically significant, and survives any letter-level or seed-level variance checks.
3.  **Methodological Demolition of the Single-Latent Metric:** Your analysis in Round 13a is a devastating, peer-review-resilient critique of current SAE Benchmarks. You have empirically proved that single-latent absorption metrics inflate absorption by $25\%$ by confounding it with feature splitting, and that this inflation systematically biases architectural comparisons.
4.  **The Real-Background Null (CDX Class):** Your Round 8 natural-feature adjudication is an excellent, honest null result. You proved that un-injected real backgrounds do not contain clean, natural-feature absorption hierarchies matching your parameters, and that real-world "stable clusters" are instead typographic UTF-8 artifacts or anti-correlated linguistic co-firings (the CDX equivalence class).

---

## 4. The Offending Phrasings: Where the Paper Overclaims

To make this manuscript resilient to peer review, we must target and rewrite the following over-scoped, misleading, or unsupported phrasings:

### A. The "Gated Absorption" / "Code Identifiability" Overclaim
*   **The Offending Phrasing (Section 7):**
    > "Consequently the binarized code separates the sub-populations nearly perfectly (conditional TV 0.9999), signature counting recovers $\rho$ to $\le 0.02$ given the pair ... dictionary identifiability and code identifiability are distinct properties — and absorption here destroys the former while encoder gating preserves the latter..."
*   **Why it is a violation:** 
    This claim of "perfect code identifiability" is a synthetic illusion of a noise-free ($\sigma = 0$) distribution. When confronted with real language-model activations (Round 9), this clean gating model completely collapses. Real background activations are highly leaky, with background mixture biases ($h_B$) ranging from $0.36$ to $0.54$. This leakage corrupts the counting estimators and rendered both of your operational predictions on real GPT-2 activations inconclusive. To claim gating "preserves code identifiability" in a general sense is false.
*   **The Actionable Fix [PRE-RESULTS-OK]:** 
    Restrict the "perfect separation" and "preserved code identifiability" claims to the stylized synthetic setting ($\sigma = 0$). Add an explicit, prominent disclosure in Section 7 stating that under real-world background activations, gating suffers from significant leakage ($h_B \approx 0.36 - 0.54$), which severely degrades code-level separation and corrupts statistical counting.

---

### B. The Detector "Partial Transfer" Overclaim
*   **The Offending Phrasing (Abstract):**
    > "...and a detector combining decoder geometry with code co-firing identifies planted parent/composite pairs on matched synthetic data and partially transfers, with strong width dependence, to semi-synthetic GPT-2 activations."
*   **Why it is a violation:** 
    "Partially transfers" is a cosmetic cover for a complete practical failure in the wild. Your detector has **$0\%$ precision** on un-injected real backgrounds. Every single one of your 15 stable real-background candidate clusters was rejected during natural-feature adjudication (0/15). The detector cannot distinguish true feature absorption from UTF-8 typographic artifacts or anti-correlated linguistic co-firings.
*   **The Actionable Fix [PRE-RESULTS-OK]:** 
    Rewrite the Abstract, Section 8, and Section 10. Demote the detector from a proposed "solution" to a "quantified baseline of co-occurrence signatures." State plainly that on un-injected real backgrounds, the detector achieves **$0\%$ precision** because its co-firing signals are completely dominated by typographic and semantic co-occurrence structures (the CDX class), and that a practical detector remains an open challenge requiring an explicit containment filter on an ASCII-clean corpus.

---

### C. The Capacity Causal Extrapolation
*   **The Offending Phrasing (Section 6.2):**
    > "In this generative model, nominal headroom moves the learned solution from absorption toward redundant composition — capacity scarcity is the operative cause of the two-latent transition here. This suggests — a hypothesis following from the model, not an evidenced claim — that for real-world models, where the landscape of feature combinations is likely far richer than the number of available latents, capacity scarcity may be a primary driver of the absorption pathology."
*   **Why it is a violation:** 
    While you soften this with "suggests... a hypothesis," the text still builds an intuitive bridge from your 2D orthonormal sandbox directly to real-world models. In real models, background competition—not raw scarcity or rarity—is the primary driver. If capacity scarcity were the dominant real-world driver, then your overcomplete SAEs ($m = 16,384$) should have shown significantly less absorption than your undercomplete/capacity-limited runs. But in your own data, the real-world L1-vs-TopK architectural contrast was a dead null at large capacity.
*   **The Actionable Fix [PRE-RESULTS-OK]:** 
    Add a dedicated "Limits of Capacity Transfer" paragraph in Section 10 (Limitations). Explicitly state that while capacity scarcity is the mathematically proved cause in your low-dimensional toy model, real-world model activations introduce high-dimensional background competition and feature-splitting dynamics that completely decouple real-world absorption from simple architectural "headroom" limits, as evidenced by your null real-data results at $m=16,384$.

---

### D. The Paper's Title: An Honest Re-framing
*   **The Offending Title:**
    > **"A Solvable Model of Feature Absorption in $L_1$ Sparse Autoencoders"**
*   **Why it is a violation:** 
    This title implies that the feature absorption observed in real-world $L_1$ SAEs has been modeled and solved. It has not. The solvable model is strictly a 2D undercomplete sandbox, while your empirical sections document the consistent limits, leakage, and failures of that model's quantitative scaling when transferred to overcomplete, leaky transformer activations.
*   **The Actionable Fix [PRE-RESULTS-OK]:** 
    Retitle the paper to reflect its true intellectual contribution:
    > **"A Solvable Toy Model of Feature Absorption and its Empirical Limits in Real-Scale SAEs"**

---

Candidate, your thesis is a highly valuable, scientifically honest work of *methodological critique* and *geometric toy-model exploration*. Do not undermine its integrity by pretending it is a unified, generalizable theory of transformer representation mechanics. 

How do you defend the current broad framing of these results when your own real-data evidence consistently points to localization, leakage, and nulls?
