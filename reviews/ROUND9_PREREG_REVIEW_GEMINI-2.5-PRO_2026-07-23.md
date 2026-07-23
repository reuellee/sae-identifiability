Excellent. This is a high-quality pre-registration and theory note, demonstrating a mature and self-critical research process. My review will therefore focus on subtle but potentially critical points that could undermine the claims, in the spirit of adversarial collaboration.

**Verdict: MINOR REVISION.** The core logic is sound and the proposed estimators are a significant step forward. However, there are several unstated assumptions, potential confounds, and definitional ambiguities that MUST be addressed before the documents are locked and the experiment is run.

---

### Adversarial Review

Here is a point-by-point analysis addressing the five requested areas.

#### 1. Algebraic & Theoretical Claims

The theory note is mostly solid, but contains one significant unstated assumption and one major omission in the analysis.

*   **(Error/Unstated Assumption) Conditional Independence in Pattern-Cell Equations:** The derivation of the `P_xy` equations in Section 3 implicitly assumes that, conditional on the token class (J, S, or B), the binarized firings `b_par` and `b_comp` are independent. For example, `P(b_par=1, b_comp=0 | S) = P(b_par=1 | S) * P(b_comp=0 | S) = g0 * (1-a0)`. This is a reasonable assumption, but it is not stated. If there were some residual correlation *within* a class (e.g., on S-tokens, a particularly strong `par` activation slightly increases the chance of a leaky `comp` activation crossing the threshold), these equations would be incorrect. **This assumption must be stated explicitly.**

*   **(Error/Omission) Background Contamination of the `11` Cell:** The analysis in Section 5 only considers background contamination of the exclusive `10` and `01` cells. It completely ignores background tokens that fire *both* latents (`11`-tokens from class B). These tokens will be fed into the `ρ̂_D` dominance-partitioning step. There is no *a priori* reason to believe `act_comp` will be consistently greater or less than `act_par` for these noise-driven co-firings. Their classification will depend on the idiosyncrasies of the decoder directions and noise distribution. This introduces an unanalyzed source of bias or variance into `ρ̂_D`, which could be significant in noisy regimes (high `σ` or real data). The theory note is incomplete without addressing this.

*   **(Minor Point) `ρ̂_count` Bias Explanation:** The numerical example in Section 2 (`[0.32 + 0.036]/[0.4 + 0.070] ≈ 0.757`) is presented without derivation of the denominator terms. While plausible, this hand-waving makes the quantitative link between the model and the observed bias less rigorous. The argument is illustrative, so this is not a major flaw, but it weakens the section.

#### 2. Experimental Design

The design is strong but has a critical missing cell and a potential loophole in its exclusion criteria.

*   **(Major Flaw) Missing Cell in Real-Activation Harness:** The RC cells test `ρ ∈ {0.3, 0.5, 0.7}`. The SC cells correctly include `ρ = 0.1`. The most dramatic bias in `ρ̂_count` occurs at the extremes of `ρ`, and background contamination bias towards 0.5 is most visible at the extremes. Excluding `ρ=0.1` from the more realistic GPT-2 harness avoids testing the estimator in the most challenging and informative conditions. The injection protocol (`Q+P0=0.4`) can support this cell (`Q=0.04, P0=0.36`). **This cell must be included.**

*   **(Loophole) Exclusion Criteria:** The rule to exclude runs where absorption "does not form" is standard, but it's a potential confound. What if certain experimental conditions (e.g., low `ρ`) systematically produce weaker `comp` features that fall just below the `comp cos > 0.98` threshold? This would mean the experiment is systematically excluding the hardest test cases, leading to an artificially inflated performance metric. **The pre-registration must commit to reporting the *rate* of exclusions per cell**, to detect and diagnose this potential failure mode.

*   **(Weakness) Statistical Power:** 8 seeds per cell is low. While MAE is a good metric, it can be sensitive to single-run outliers with N=8. A single failed run where `ρ̂_D` gives a nonsensical result could cause an entire cell to fail the falsification bar, even if the estimator is sound 7/8 times. Conversely, a few lucky runs could mask a real issue. The project should at least acknowledge this limitation and the corresponding width of the bootstrap CIs is expected to be large.

#### 3. Estimator Definitions

The definitions are mostly precise, but contain ambiguities and an arbitrary choice.

*   **(Major Flaw) Ambiguity of `ρ̂_count` Definition:** The definition `ρ̂_count = P(b_lo) / P(b_lo ∨ b_hi)` where `lo = rarer latent` is ambiguous and potentially circular. Rarer over which population of tokens? All tokens? Only parent-event tokens? The identity of the "rarer" latent depends on the true `ρ`. If `ρ < 0.5`, `comp` is rarer. If `ρ > 0.5`, `par` is rarer. At `ρ = 0.5`, the definition is unstable. This must be replaced with a precise, non-circular definition (e.g., "the latent with lower total firings over the evaluation dataset"). This is critical for `P3`, which compares `ρ̂_D` to `ρ̂_count`.

*   **(Minor Flaw) Tie-breaking in `ρ̂_D`:** The rule "ties → S" is an arbitrary, hard-coded choice. While ties are likely rare, this choice could introduce a small, systematic bias. The pre-registration correctly states the tie rate will be reported, but it should also briefly justify the choice or acknowledge its arbitrary nature. A more neutral choice might be to discard ties or split them 0.5/0.5, though this adds complexity.

*   **(Unstated Dependence) `θ = 0.05`:** The project correctly identifies the need to avoid tuned constants like magnitude margins. However, the entire estimation pipeline depends on the binarization threshold `θ = 0.05`. This is still a "tuned constant," albeit one inherited from the detector. The validity of the F1 (`a₁≈1`) and F2 (`g₀≈1`) assumptions depends on this choice. This dependence should be acknowledged in the theory note as a scope limitation.

#### 4. Falsification Bars

The bars are mostly well-calibrated, but the bar for the mechanism check is too lenient.

*   The primary bars for `ρ̂_D` (MAE ≤ 0.05 pass / > 0.15 fail) are reasonable. The 0.10 "inconclusive" gap is wide enough to absorb noise from the low seed count (N=8) while still demanding a clear signal. The pass bar is ambitious and the fail bar represents a clear failure to substantially improve on the naive estimator's bias (which can be >0.25).

*   **(Too Lax) P4 Mechanism Bar:** The bar for dominance inversion rate (≤ 0.05 pass / > 0.20 fail) is too generous. If 15% of `11`-tokens are misclassified, that represents a significant failure of the core assumption C-dom and will introduce a non-trivial bias into `ρ̂_D`, especially if the `11` cell is large. A failure bar of `> 0.10` would be more rigorous and would flag a meaningful breakdown of the proposed mechanism.

---

### Required Changes vs. Suggestions

#### Required Changes (must be addressed before locking)

1.  **Theory Note:** Explicitly state the conditional independence assumption underlying the pattern-cell equations in Section 3.
2.  **Theory Note:** Add an analysis of background (`B`-class) `11`-token contamination to Section 5, discussing its potential impact on `ρ̂_D` and how it represents an unmodeled source of error.
3.  **Pre-reg `Estimators`:** Replace the ambiguous `lo = rarer latent` definition for `ρ̂_count` with a precise, deterministic procedure (e.g., based on total firings on the eval set).
4.  **Pre-reg `Confirmatory cells`:** Add the `ρ = 0.1` cell to the real-activation (RC) harness.
5.  **Pre-reg `Exclusions`:** Add a clause requiring the reporting of the exclusion rate for each experimental cell to monitor for systematic drop-outs.

#### Suggestions (recommended for increased rigor)

1.  **Theory Note:** In the `ρ̂_D` definition (Section 4), acknowledge the orientation-equivariance property (`ρ̂_D → 1 − ρ̂_D` under label swap) is a valuable feature for downstream automatic orientation.
2.  **Pre-reg `Estimators`:** Briefly note that the "ties → S" rule is an arbitrary choice and that the reported tie rate will quantify its potential impact.
3.  **Pre-reg `Registered predictions`:** Consider tightening the falsification bar for P4 (dominance inversion) to `> 0.10` to represent a more meaningful failure of the C-dom assumption.
4.  **Project-wide:** Consider adding a small, exploratory experiment in a future round to test the sensitivity of the F1/F2 assumptions and the `ρ̂_D` estimator to the choice of the binarization threshold `θ`.
5.  **Experimental Design:** If compute budget is flexible, increasing the seed count from 8 to 12 or 16 would significantly improve the statistical reliability of the MAE results and narrow the bootstrap CIs.
