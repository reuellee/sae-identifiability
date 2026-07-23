Excellent. This is a high-quality project with a rigorous methodology. As an adversarial reviewer, my goal is to find the weak points to strengthen the final result. My review follows the requested structure.

---

### **Adversarial Review: Round 10 (TopK Absorption)**

**To:** Project Lead, `sae-identifiability`
**From:** External Reviewer
**Date:** 2026-07-25
**Subject:** Adversarial review of DRAFTs `topk_absorption.md` and `prereg-topk-absorption.md`

This is an impressive and important extension of the project's core theory. Moving from L1 to TopK SAEs is critical for relevance, and the theoretical results—the `2q` crossover and the sharp capacity collapse—are elegant and surprising. The pre-registration methodology remains a model of rigor. My review is aimed at stress-testing the claims and experimental design before they are locked.

### 1. Algebra and Theoretical Claims

The algebra appears **correct**. I have verified the core claims.

*   **Per-event loss table:** The values are correct. The `loss=1` for faithful/joint/κ=1 follows from dropping one of two equal-projection components. The `loss=0` for absorbed/joint/κ=1 follows from the composite atom perfectly matching the event.
*   **Nonnegativity signature (loss=1/2):** The derivation is sound. Reconstructing `v_c` from `{v_p, d_comp}` requires `v_c = sqrt(2)·d_comp - 1·v_p`. The non-negativity constraint on the coefficient of `v_p` is key. The best non-negative least squares (NNLS) solution is indeed to project `v_c` onto the cone spanned by `{v_p, d_comp}`, which yields `(1/sqrt(2))·d_comp = 0.5(v_p+v_c)`. The residual `v_c - 0.5(v_p+v_c) = 0.5(v_c-v_p)` has norm-squared `0.25 * (||v_c||^2 + ||-v_p||^2) = 0.5`. This holds for any `κ >= 1`, as the second slot for `v_p` cannot be used with a negative coefficient. This is the linchpin of the capacity collapse, and it is solid.
*   **`eps*_TopK = 2q` derivation:** `E[L_faithful, k=1] = q` and `E[L_absorbed, k=1] = 0.5*eps`. The crossover `q = 0.5*eps` gives `eps = 2q`. Correct.
*   **Capacity collapse claim:** At `κ>=2`, `E[L_faithful] = 0` while `E[L_absorbed] = 0.5*eps`. The faithful dictionary is strictly better for any `eps > 0`. Correct.

**Unstated Assumptions/Minor Weakness:** The theory is clean. The only minor point is that the geometric intuition for the `loss=1/2` could be slightly sharpened. It arises because `v_p` and `d_comp` are non-orthogonal, so the cone they span does not contain `v_c`. The projection of `v_c` onto this cone is the best NNLS solution. This is a minor pedagogical point, not an error.

### 2. Subsystem Budget (κ) vs. Global Budget (k)

The mapping `κ ≈ k - B` is a reasonable and necessary heuristic to bridge the idealized 2D theory with a more complex SAE. The experimental design correctly uses this heuristic to probe the two key regimes.

*   **Soundness:** The logic is sound. `k` is a global resource, and other active features (`B`) consume it, leaving a smaller effective budget `κ` for the subsystem.
*   **Realization in Experiment:**
    *   **Arm M (isolated):** Using low-amplitude background features to ensure the pair gets priority for `k=1` and `k=2` slots is a clever way to directly instantiate the theoretical `κ=1` and `κ=2` conditions. This is a well-designed control.
    *   **Arm C (background sweep):** Sweeping `k` across the expected background load (`B ≈ 2.4`) is the correct way to test the "gating" hypothesis. The chosen `k` values correctly span the tight (`k < B+2`), transitional (`k ≈ B+2`), and spare (`k > B+2`) regimes.

**Critique:** The model `κ ≈ k - B` treats `B` as a fixed value (the expectation). In reality, `B` is a random variable per token (specifically, `Bin(n_bg, bg_rate)`). This means that even in the `k=4` Arm C cell, some tokens will have `B=1` (leaving `κ=3`) while others have `B=4` (leaving `κ=0`). The trained SAE will average over this stochasticity. This will likely "soften" the sharp theoretical transition, making it appear more gradual in the experiment. The theory note should explicitly state this to manage expectations.

### 3. Experimental Design

The design is strong, but an adversarial mindset reveals potential failure modes.

*   **`phi_child` Measurement:** The heuristic (`in-plane latent... largest angle in (20°,120°)`) is pragmatic. However, it is a proxy.
    *   **Attack:** What if a trained SAE at `k=2` doesn't find a single neat feature near 90°, but instead represents the child-space with two features, e.g., one at 70° and one at 110°? The heuristic would pick the 110° one, which passes the `>75°` bar, but the "story" is more complex. More critically, what if the system learns features that cause the measurement to be undefined (e.g., all child-like features are <20° or >120°)? The exclusion rule handles this by dropping the seed, but if an entire experimental cell becomes unscoreable (`<16/24` seeds), the claim is not falsified, it is merely untested for that cell. This is a potential way for the experiment to inconclusively "dodge" a falsification.
*   **Theory-Experiment Gap (SGD vs. Global Optimum):** The theory proves the properties of the *global* optimum of the population loss. SGD is a stochastic local optimizer. For very small `eps > 0`, the loss for the absorbed dictionary (`0.5*eps`) is only marginally worse than the faithful one (`0`). The gradient signal pushing the system out of the (locally attractive) absorbed configuration might be vanishingly small, making it easy for SGD to get stuck.
    *   **Attack:** The primary claim T2 could **fail while being true**. Specifically, for the `eps=0.05` cell, the `k=2` runs might fail to escape the absorbed basin and return `phi <= 55°`, thus triggering the falsification bar. This would not mean the capacity collapse theorem is wrong, but that its effects are not reachable by SGD under those conditions. The pre-registration should explicitly foresee this possibility.
*   **Seeds and Cells:** 24 seeds is robust. The exclusion rule (`>=16/24`) is standard and fair. No issues here.

### 4. Falsification Bars

The bars are generally well-calibrated, specific, and tied directly to the theory.

*   **T2 (Capacity Collapse):** The dual condition (`phi >= 75°` AND `gap >= 15°`) with a tight falsification (`phi <= 55°` OR `gap <= 5°`) is excellent. It creates a clear pass/fail/inconclusive space. The `phi` bands (`<=55` absorbed, `>=75` faithful) are reasonable, allowing for some noise around the 45° and 90° poles.
*   **T1 (Crossover):** The range for `eps_mid` (`[0.3*2q, 1.0*2q]`) is appropriately wide to account for SGD dynamics, and the falsification bar (`[0.15*2q, 1.3*2q]`) provides a generous margin. Testing that `eps_mid` scales with `q` is the most critical part, and the bar `eps_mid(q=0.2) > eps_mid(q=0.1)` correctly captures this.
*   **T3 (Capacity Gating):** This is a good practical test. The condition `phi(k=16) - phi(k=1) >= 20°` (pass) vs `< 5°` (fail) is a clear test of whether capacity has a meaningful effect in the presence of background noise.

**Critique:** As noted in (3), the T2 bar is vulnerable to failing for the smallest `eps` value due to optimization dynamics, not a failure of the underlying theory. A failure in the `q=0.1, eps=0.05` cell should not be treated as a catastrophic failure of the entire theory without this context.

### 5. L1-vs-TopK Contrast

The contrast is stated **correctly and is not overclaimed**.

*   The core distinction—L1 absorption is driven by a `lambda`-tax incentive, while TopK absorption is driven by a `k`-slot scarcity—is the central insight of this work, and it is articulated clearly and accurately.
*   The document correctly identifies that L1 *prefers* a redundant composite atom at spare capacity to save on the L1 tax, whereas TopK is at best indifferent to it once `k` is sufficient for a faithful reconstruction.
*   The note in §4 wisely frames the `eps*` comparison as a statement about mechanism, not a direct performance race, which is the correct scientific interpretation.

### 6. Verdict and Required Changes

This is high-quality work on track for a significant result. The issues are minor and addressable with small clarifications to improve rigor and manage expectations.

**Verdict: MINOR REVISION**

---

**Required Changes (to be implemented before lock):**

1.  **Acknowledge SGD vs. Global Optimum Risk in Preregistration (T2):** In `prereg-topk-absorption.md`, under the **T2** prediction, add a sentence acknowledging that for very small `eps`, the gradient separating the faithful and absorbed optima is small, and SGD may fail to find the global optimum. State that a failure of the bar *only* at the lowest `eps` value(s) would be interpreted in this light, rather than as a wholesale rejection of the capacity collapse theorem. This is critical for honest scoring.
2.  **Acknowledge Stochasticity of Background Load:** In `topk_absorption.md`, at the end of §9 ("From subsystem budget κ..."), add a sentence clarifying that the number of active background features `B` is a stochastic variable for each token, and the trained SAE will optimize for the *average* case. This implies the sharp theoretical transition at `κ=2` will likely manifest as a softer, more gradual transition in the Arm C experiment as `k` is swept.

**Suggestions (for consideration):**

1.  **Clarify Arm M Design Rationale:** In `prereg-topk-absorption.md`, Arm M description, briefly state *why* the background is low-amplitude (e.g., "...to ensure the high-amplitude parent/child pair features reliably win the top-k slots, thus closely approximating the theoretical κ=1 and κ=2 conditions."). This makes the design logic more explicit.
2.  **Note `phi` Measurement Failure Mode:** In the `prereg-topk-absorption.md` "Measurements" section, consider adding a sentence acknowledging the possibility of a cell becoming unscoreable if a particular configuration systematically produces features outside the `phi_child` measurement heuristic's range.
3.  **Sharpen Non-negativity Intuition:** In `topk_absorption.md` §3, consider adding a phrase to the `loss=1/2` explanation emphasizing the geometric picture, e.g., "...ReLU forbids negative codes, so the best reconstruction is the projection of `v_c` onto the non-orthogonal cone spanned by `{v_p, d_comp}`...".

These changes are minor but will make the theoretical claims more precise about their scope and the experimental claims more robust to edge-case outcomes. I look forward to seeing the locked versions and the results. This is poised to be a foundational result for SAE theory.
