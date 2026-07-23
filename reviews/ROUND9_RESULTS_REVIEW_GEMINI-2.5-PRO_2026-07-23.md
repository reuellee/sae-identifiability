Excellent. This is a model case for a results-stage adversarial review. The preregistration, frozen scorer, and detailed reporting provide all the necessary components for a rigorous check.

### Verdict: ACCEPT

The results are reported with exemplary integrity. The analysis is faithful to the locked preregistration, falsifications and inconclusive results are handled with diagnostic honesty rather than spin, and the claims are appropriately scoped. No revisions are required.

---

### Detailed Review

#### 1. Is the scoring faithful to the locked bars?

**Yes, entirely.** I have cross-checked every verdict against the registered bars and the frozen scorer's output.

*   **P1M/P2M (Mechanism MAE):** The pass bar was `MAE <= 0.03`. The worst-case observed MAE across all 16 cells was 0.0026. This is an order of magnitude better than the bar. The **PASS** verdicts (4/4 and 12/12) are correct.
*   **P1O/P2O (Operational MAE):** The bars were `pass <= 0.05` and `falsify > 0.15`, with the region in between being inconclusive.
    *   In P1O (RC), the `rho=0.1` cell scored 0.0695, which is correctly identified as being in the inconclusive zone. The other 3 cells were `< 0.05` (pass). The overall **INCONCLUSIVE** verdict is correct.
    *   In P2O (SC), the `rho=0.1, sig=0.1` cell scored 0.0660, also correctly identified as inconclusive. The other 11 cells were `< 0.05` (pass). The overall **INCONCLUSIVE** verdict is correct.
*   **P3 (Dominance):** The bar was `rho_c-O MAE - rho_d-O MAE >= 0.05`. The verdict states two cells failed: `SC sig=0` at `rho=0.1` and `rho=0.3`.
    *   Cell `rho=0.1`: `0.0350 - 0.0013 = 0.0337`, which is `< 0.05`. **Fail.**
    *   Cell `rho=0.3`: `0.0245 - 0.0021 = 0.0224`, which is `< 0.05`. **Fail.**
    The calculations are correct. With 2/14 failures, the overall **FALSIFIED** verdict is correct and faithfully reported.
*   **P4 (Side Effects):** The pass bar was `median delta <= 0.05`. The worst observed median delta was 0.0043 (`med dS` in the `RC rho=0.1` cell). All 16 cells are well within the pass threshold. The **PASS 16/16** verdict is correct.

There is no evidence of post-hoc reinterpretation, softened language, or verdict spin. The reporting is a direct and honest application of the preregistered scoring plan.

#### 2. Is the P3 FALSIFIED verdict handled honestly?

**Yes, with textbook-perfect honesty.** This is a model for how to handle a falsification.

*   **Clarity:** The writeup states "P3 is falsified as registered" without equivocation.
*   **Diagnosis, not Excuse:** The explanation is a diagnosis of the experimental design, not an excuse for the estimator. It correctly identifies that the *precondition* for the test (the baseline `rho_C` having a large enough error) was not met in the two failing cells.
*   **Evidence-Based:** The diagnosis is supported by data from the frozen scorer. The claim that the synthetic harness was "almost cleanly" gated is backed by the reported leak diagnostics (`a0` and `g1` near zero). The claim that the baseline's error was too small to clear the 0.05 margin is backed by the reported `rho_c-O MAE` values (0.0350 and 0.0245).
*   **Learning:** The writeup correctly extracts the lesson: "leak magnitudes do not transfer across synthetic harnesses." This turns the falsification into a valuable scientific finding about the experimental setup itself.

The framing is legitimate and demonstrates a mature approach to confirmatory research.

#### 3. Are the two INCONCLUSIVE operational cells framed correctly?

**Yes.** The language is precise. The writeup uses the term "inconclusive zone" and correctly states that these cells are neither passes nor failures. The diagnosis provided—that these are the cells where the background bias term `w_B·(h_B − ρ)` is largest—is a valid and insightful explanation for why these specific cells fell into the inconclusive zone while others passed. Highlighting that one of these was flagged as "at-risk" a-priori further strengthens the credibility of the analysis.

#### 4. Does the writeup's claim language stay inside what the design can support?

**Yes.** The authors have been careful to scope their claims correctly.

*   The "Headline numbers" section clearly distinguishes between the "mechanism" claim (on parent-event tokens, given oracle pair/orientation) and the "operational" results (on all tokens), correctly attributing the latter's errors to a predictable background bias.
*   The dedicated "Scope" section is excellent. It explicitly lists what has *not* been established (detection, automatic orientation, etc.), which is crucial for preventing over-interpretation by others. This is exactly what a rigorous report should do.

The claims are appropriately modest and strictly limited to what was demonstrated.

#### 5. Anything in the numbers that looks anomalous?

No, the numbers appear consistent and tell a coherent, powerful story.

*   **"Too Clean" in a Good Way:** The `rho_d-M MAE` values (all `< 0.0027`) are exceptionally low, beating the pass threshold by over 10x. This isn't an anomaly to be suspicious of; it's a sign of a remarkably effective and well-specified estimator for the mechanism-level task.
*   **Negligible Side Effects:** The `med dJ` and `med dS` values are effectively zero, confirming that the measurement intervention itself is not disrupting the model's function. This is a critical sanity check that passes with flying colors.
*   **Baseline Failure:** The `rho_c-O MAE` values are large where leak is present, confirming that the problem the new estimator was designed to solve is real and significant.
*   **Diagnostic Consistency:** The leak diagnostics (`a0`, `g1`, `h_B`) vary substantially and systematically across cells, just as expected. The fact that `rho_D` remains accurate despite this wild variation is the core evidence for its robustness.

The data is clean, internally consistent, and strongly supports the conclusions drawn.

### Suggestions for Improvement (Optional)

These are minor points to consider for a final publication; they are not required changes for accepting the current summary.

1.  **(Suggestion) Emphasize the P3 Falsification as a Methodological Win:** The writeup handles the P3 falsification perfectly. It could be worth adding a sentence to frame this not just as a finding about synthetic harnesses, but as a demonstration of the power of preregistration: it allows for clean falsification and diagnosis, which is often more scientifically valuable than a vague or post-hoc-justified "pass."
2.  **(Suggestion) Quantify the "Decisive" Mechanism Win:** The summary states the mechanism claim is "confirmed decisively." This could be made even more concrete by stating that the estimator beat its preregistered pass criterion by more than a factor of 10 in every one of the 16 experimental conditions.
