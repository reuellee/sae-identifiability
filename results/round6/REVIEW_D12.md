# Adversarial Review of D12 Interpretations

This document provides an adversarial review of the interpretations from the D12 experiment series. Each section presents a verdict on the safety of an interpretation and proposes concrete steps to disambiguate or strengthen the claim.

## 1. Claim: 'absorption transfers to real activations'

**Interpretation:** The trend of increasing `func_cos_child` with `eps` in the vanilla condition is evidence of absorption, similar to what's seen with synthetic data.

**Verdict: Unsafe.** The data for vanilla at `eps=0.002` is too noisy to be confidently distinguished from simple undertraining or reconstruction noise. The `func_cos_child` values `[0.45, 0.01, 0.43, -0.03]` are inconsistent and one is near zero, which does not strongly support the absorption hypothesis. While higher `eps` values show a clear trend, the crucial low-`eps` regime, where absorption should be most distinct from simple feature representation, is ambiguous.

**Proposed Fix:**
To distinguish absorption from noise/undertraining, we need to analyze the characteristics of the feature activations themselves, not just their cosine similarity.

*   **New Metric: Feature Activation Sparsity & Distribution.**
    *   **Hypothesis:** If it's true absorption, the child feature's activations on the parent's activating dataset examples should be sparse and high-magnitude. If it's noise or undertraining, the activations would be more evenly distributed and lower magnitude.
    *   **Implementation:**
        1.  Identify the top activating examples for the parent feature.
        2.  For these examples, plot the distribution of the *child* feature's activations.
        3.  Measure the Gini coefficient or L1/L2 norm ratio of this activation distribution. A higher value would indicate sparsity, favoring the absorption hypothesis.
        4.  Compare this to the child feature's activation distribution on a random dataset sample as a baseline.

---

## 2. Claim: 'residual weighting is the best... and beats the oracle'

**Interpretation:** Residual-weighted SAEs outperform the oracle, as seen by the oracle's `func_cos_child` collapsing to near-zero at `eps=0.01` and `0.05`.

**Verdict: Unsafe. The headline 'practical beats oracle' is not justified.** The oracle condition's failure mode is highly suspicious. The `max_cos_child` remains high (`~0.70-0.72`) while the `func_cos_child` collapses. This strongly suggests an **encoder/probe artifact**, not a failure of the dictionary itself. The functional metric relies on the encoder's argmax response to a synthetic probe, which may be brittle. The oracle SAE likely learned a perfectly valid representation of the child feature, but the specific way `func_cos_child` is calculated fails to detect it. The oracle may have learned a different feature that is also activated by the probe.

**Proposed Fix:**
Disambiguate with a metric that is less dependent on the specific encoder implementation.

*   **New Metric: Decoder-Based Reconstruction Score.**
    *   **Hypothesis:** If the child feature is correctly represented in the dictionary, activating it should reconstruct the child feature in the model's output space, regardless of what other features the encoder might prefer for a given probe.
    *   **Implementation:**
        1.  For a given test activation (e.g., the child feature vector itself), get the SAE's internal latent representation (the vector of feature activations).
        2.  Force activation of the single feature with the highest `max_cos_child` (the putative child feature). Set all other feature activations to zero.
        3.  Decode this single-feature activation vector back into model activation space: `x_hat = W_dec * feature_activation`.
        4.  Calculate the cosine similarity between this `x_hat` and the original child feature vector. A high similarity would confirm the feature is represented correctly in the dictionary, even if the `func_cos_child` metric failed.

---

## 3. Claim: 'natural-absorption coverage-gap improvements'

**Interpretation:** The reduction in the coverage gap from 0.768 (vanilla) to 0.735 (residual) is due to residual weighting mitigating natural absorption.

**Verdict: Potentially Misleading.** The improvement could be an artifact of the residual-weighted SAEs simply having **higher overall feature firing rates**. The weighting scheme (`clamp(res^2/mean,1,50)`) directly increases the L2 norm of activations sent to the SAE, which could lead to denser/higher activations across the board. This might reduce the "coverage gap" (by making it easier to find *any* feature, not necessarily the *right* one) without actually improving the representation of specific, sparsely-activating features. The drop in `n_split` (6.4 -> 5.6) offers weak support, but is not conclusive.

**Proposed Fix:**
Normalize for overall firing rates to isolate the effect on feature representation.

*   **Control Statistic: Per-Feature Activation Rate & Magnitude.**
    *   **Hypothesis:** A genuine improvement would involve the features corresponding to the "uncovered" tokens having selectively higher activation rates, not a uniform increase across all features.
    *   **Implementation:**
        1.  Calculate the mean firing rate (L0 norm) and mean activation magnitude (L1 norm) for all features in both vanilla and residual-weighted SAEs across a large, representative dataset.
        2.  If the residual-weighted SAE shows a significantly higher global mean, the coverage-gap improvement is suspect.
        3.  **Crucially:** For the specific tokens that were previously "uncovered" by the vanilla SAE, identify the features that now activate on them in the residual-weighted SAE.
        4.  Compare the activation rates of *these specific features* to the average feature activation rate. A true improvement would see these features having typical or even sparse activation patterns, while still covering the token. A simple global increase in density would not be a targeted improvement.

---

## 4. Claim: 'audit null result implies no hierarchical pairs'

**Interpretation:** The audit found 0 hierarchical pairs in a public GPT-2 SAE, suggesting such structures are rare or non-existent in normally trained SAEs.

**Verdict: Weak Conclusion, Likely Over-fitting to a Brittle Definition.** The null result is more likely a consequence of overly strict criteria than a true absence of hierarchical structure. The `share > 0.7` threshold for co-occurrence is extremely high and vulnerable to noise. Two features could be strongly related (e.g., one fires on 80% of the other's activations), but noise, multidetector neurons, or slight feature mismatch could easily push the observed share below this rigid threshold. The "post-absorption" caveat in the summary is key: we are not looking at ground truth.

**Proposed V2 Criteria:**
Move from a single, rigid threshold to a more statistical and multi-faceted approach.

*   **Statistical Enrichment:** Instead of a fixed share, test for statistically significant co-occurrence.
    *   For a feature pair (A, B), calculate the conditional probability `P(A fires | B fires)`.
    *   Compare this to the baseline probability `P(A fires)`.
    *   A strong hierarchical candidate would have a `P(A|B)` that is many times higher than `P(A)`. Use a significance test (e.g., Fisher's exact test) to find pairs with non-random co-occurrence.
*   **Directional Information Flow:** The current metric is symmetric. Look for asymmetry.
    *   A parent feature should predict a child, but the child should not predict the parent as strongly.
    *   Measure and compare `P(Child | Parent)` vs. `P(Parent | Child)`. A high ratio suggests a directional relationship.
*   **Activation Value Correlation:**
    *   For co-occurring activations, are the *magnitudes* correlated? A positive correlation between the parent's and child's activation values on the same token would be further evidence of a relationship.
    *   Run a simple correlation test on the activation values for significantly co-occurring pairs.

By combining these softer, more statistical signals, a v2 audit would be more likely to uncover the true extent of hierarchical structure, which is likely encoded more loosely than the v1 criteria can detect.
