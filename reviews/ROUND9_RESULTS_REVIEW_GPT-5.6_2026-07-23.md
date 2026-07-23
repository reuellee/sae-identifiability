# Round-9 results-stage review — GPT-5.6 Sol (High), chatgpt.com, 2026-07-23

*Verbatim canvas document (extracted via copy button). The reviewer independently*
*fetched the public repo, verified the lock commit and byte-identical frozen*
*analyzer, and reproduced all six registered verdicts from the committed CSV.*
*Verdict: MINOR REVISION → ACCEPT after the required corrections (no rerun).*

﻿# Results-stage adversarial review: round 9 gating-corrected ρ estimation

## Verdict

**MINOR REVISION**

The confirmatory experiment itself is acceptable. I found no post-hoc bar change, cell redefinition, or verdict manipulation. The current analyzer is byte-for-byte identical to the lock version, and rerunning it on the committed CSV reproduces the supplied cell table and all six registered verdicts. The preregistration also differs from the lock version only by the expected amendment recording the lock hash.

The revision is “minor” because no GPU rerun or verdict change is required. It is nevertheless mandatory: the summary contains numerical errors, omits several registered descriptive outputs, and uses language that partially converts inconclusive or falsified endpoints into stronger successes than the preregistration permits.

---

## 1. Fidelity to the locked scoring

### Primary verdicts

The registered decisions are applied correctly:

* P1M: pass, 4/4 cells.
* P2M: pass, 12/12 cells.
* P1O: inconclusive, because RC (\rho=0.1) has MAE 0.0695.
* P2O: inconclusive, because SC (\rho=0.1,\sigma=0.1) has MAE 0.0660.
* P3: falsified, because two predefined cells miss the 0.05 improvement margin.
* P4: pass, 16/16 under the frozen median rule.

These match the locked bars and predefined P3 cell list.

The bootstrap intervals reinforce, rather than alter, the operational verdicts:

* RC (\rho=0.1): MAE 0.0695, bootstrap interval approximately **[0.0659, 0.0732]**.
* SC (\rho=0.1,\sigma=0.1): MAE 0.0660, interval approximately **[0.0642, 0.0678]**.

Both are comfortably inside the registered inconclusive zone, not borderline passes or failures.

### Registered reporting is incomplete

The preregistration promised that bootstrap intervals, signed bias, RMSE, threshold sensitivity, automatic orientation, tie rates, P4 denominator counts, and the P5 odds-formula prediction would be reported. The frozen analyzer computes some of these internally but does not print the intervals, does not compute RMSE, does not disclose P4 contributor counts, and does not implement the registered P5 eligibility/prediction report. The summary provides only a subset.

This does not invalidate the pass/fail outcomes, but it is a preregistration-compliance defect. Add a dated, reporting-only analyzer or appendix that reads the frozen CSV and does not modify any endpoint, eligibility rule, or bar.

---

## 2. P3 falsification

### Is “eligibility misprediction” legitimate?

**Yes, as a post-hoc diagnosis.**

In the two failing cells:

* SC (\rho=0.1,\sigma=0): improvement was (0.0350-0.0013=0.0337).
* SC (\rho=0.3,\sigma=0): improvement was (0.0245-0.0021=0.0224).

Because the baseline errors themselves were below 0.05, even a mathematically perfect (\hat\rho_D) could not have cleared the registered 0.05 margin. The predefined eligibility model therefore selected two cells in which the endpoint was arithmetically impossible to pass. That is genuinely an eligibility-prediction failure, not evidence that the dominance estimator performed poorly. The preregistration explicitly selected these cells using an Arm-A proxy of (a_0\approx0.16), whereas the confirmatory harness produced (a_0\le0.038).

### Where the writeup becomes defensive

The wording

> “P3 is falsified … by eligibility misprediction, not by the estimator”

is too exculpatory. P3 was a claim about the estimator’s registered comparative performance, and that claim was falsified. The estimator can simultaneously have performed well in absolute terms.

Use:

> **P3 was falsified in 2/14 cells. A post-hoc diagnosis is that the eligibility model overpredicted baseline bias in those cells; (\hat\rho_D) remained more accurate, but the registered 0.05 improvement margin was unattainable.**

The claim that the discrepancy occurred because the E3 host coefficient was deterministic, unlike Arm A’s, is a plausible mechanistic hypothesis but was not isolated experimentally here. Label it as post hoc rather than established fact.

“Consistent with round 8b’s transfer lesson” is acceptable contextualization, provided it is not presented as confirmatory evidence for that causal explanation.

---

## 3. The two operational inconclusive cells

The verdict table handles them correctly: neither is called a pass or a failure. The summary also explicitly states that neither operational prediction was falsified.

The later prose is less disciplined:

> “inherits exactly the predicted background bias and nothing else”

is not supported.

The locked preregistration predicted approximately 0.045 error for the at-risk RC cell; the observed value was 0.0695. The result landed in the anticipated inconclusive region, but the magnitude was not “exactly predicted.”

Moreover, the near-exact match to the **post-run measured** mixture is largely an algebraic decomposition:

[
\hat\rho_{D,O}
==============

(1-w_B)\hat\rho_{D,M}+w_Bh_B.
]

Once (w_B), (h_B), and the mechanism estimate are measured on the same run, agreement is an accounting check, not an independent prediction. It verifies implementation and locates the error in background-active tokens, but it cannot establish that there was “nothing else.”

Use:

> **The operational deviations are quantitatively accounted for by the post-run measured background-mixture decomposition. This supports the proposed error mechanism, but both registered operational predictions remain inconclusive because one cell in each harness exceeded the pass bar.**

The report should also state the overall verdicts directly. “The all-token version passed 14/16 cells” is numerically true at the cell level, but it can soften the registered result. The correct synthesis is:

> **Fourteen cell-level passes and two inconclusive cells; consequently both P1O and P2O are inconclusive overall.**

---

## 4. Scope of the claims

The dedicated scope section is good. It explicitly limits the evidence to:

* oracle-located pairs;
* criterion-qualified absorbed runs;
* oracle orientation;
* the existing GPT-2 layer-6 activation bank;
* no detector execution;
* no validated automatic orientation;
* no cross-domain or background-corrected claim.

The report’s broader language partially escapes those guardrails:

* “the counting stage fixed at the mechanism level”;
* “frequency estimation given the pair is solved”;
* “confirmed decisively”;
* “zero tuned constants.”

“Fixed” and “solved” imply generality beyond one synthetic family and one semi-synthetic activation bank. The correct claim is that the **registered mechanism endpoint passed decisively in these oracle-scoped harnesses**. The real-activation experiments remain semi-synthetic because the parent–child pair is injected into a reused GPT-2 activation bank.

“Zero tuned constants” should be replaced with:

> **No new harness-tuned constant beyond the preregistered inherited threshold (\theta=0.05).**

The estimator still depends on that absolute threshold and on unit decoder normalization. The preregistration itself uses the more accurate “without any new harness-tuned constant.”

---

## 5. Hard numerical and reporting errors

### 5.1 Formation count is wrong

The summary says:

> “SC 335/384 total included.”

The listed cell counts sum to:

[
254/288\text{ SC runs included},
]

with 34 SC exclusions. Adding all 96 RC runs gives:

[
350/384\text{ total included}.
]

This must be corrected. The current number is impossible because SC contained only 288 attempted runs.

### 5.2 Baseline-error range is wrong

The summary claims that (\hat\rho_C) errs by 0.13–0.49 in every leaky cell. Two noisy synthetic cells are below 0.13:

* SC (\rho=0.7,\sigma=0.05): 0.0834.
* SC (\rho=0.7,\sigma=0.1): 0.1147.

Across the RC and (\sigma>0) SC cells, the actual range is approximately **0.083–0.494**.

### 5.3 P5 summary uses the wrong bound

The preregistered descriptive expectation was (|\text{bias}|\le0.10) in cells satisfying both leak-symmetry and background-symmetry conditions. The summary instead says biases were at most 0.05 except one cell.

That statement is false: the eligible SC (\rho=0.3,\sigma=0.1) cell has bias approximately **+0.0712**. It satisfies the registered 0.10 expectation but not the summary’s post-results 0.05 claim.

Report the actual registered result:

> **Among cells satisfying the preregistered symmetry conditions, all but SC (\rho=0.1,\sigma=0.1) had (|\hat\rho_X-\rho|\le0.10); that cell had bias +0.125.**

Also list exactly which cells met the two eligibility conditions.

### 5.4 The (h_B) range is misleading

The statement that (h_B) is “measurably not a constant (0.0–0.54)” uses zero values from cells with essentially no background-active tokens. In those cells (h_B) is undefined or extremely unstable, not meaningfully measured as zero.

For cells with substantial background-active mass, the cell means are roughly:

* RC: 0.357–0.539.
* SC (\sigma=0.1): 0.493–0.512.

The (\sigma=0.05) cells have tiny background weights, so their (h_B) estimates are noisy and operationally negligible. Report (h_B) as NA when its denominator is zero and accompany it with (w_B) or the B-active token count.

---

## 6. Numbers worth follow-up

### 6.1 The mechanism result is exceptionally clean—but explainably so

In the synthetic (\sigma=0) and (\sigma=0.05) cells, (\hat\rho_{D,M}) equals the empirically realized J fraction exactly on every included run. Most of the reported 0.001–0.002 MAE against nominal ρ is therefore just finite evaluation-set variation in realized prevalence.

This is not suspicious. With the oracle background mask and perfect token classification, equality with realized ρ is algebraically expected. It means the mechanism endpoint is principally validating the token-classification rule and C-dom, not an end-to-end frequency pipeline.

The writeup should show both:

[
\operatorname{MAE}(\hat\rho_{D,M},\rho_{\rm nominal})
]

and

[
\operatorname{MAE}(\hat\rho_{D,M},\rho_{\rm realized}).
]

That will make the extreme cleanliness transparent rather than impressive-looking but opaque.

### 6.2 P4 has sparse contributor counts in clean-gating cells

The registered P4 median rule passes, but some (\sigma=0) medians are based on very few seeds whose class-11 denominators exceeded 100. For J/11, the contributor counts are only:

* (\rho=0.1): 1 of 19 included seeds;
* (\rho=0.3): 5 of 21;
* (\rho=0.5): 5 of 21;
* (\rho=0.7): 5 of 19.

This is expected when cofiring is rare in the clean-gating regime, and the weighted inversion mass is essentially zero. It does mean that “P4 pass 16/16” should be accompanied by the registered denominator counts rather than presented as equally strong evidence in all cells.

### 6.3 Operational performance is materially threshold-sensitive

The preregistered threshold sensitivity is important and currently omitted. In the problematic SC (\rho=0.1,\sigma=0.1) cell, operational MAE is approximately:

* (\theta=0.02:\ 0.101);
* (\theta=0.05:\ 0.066);
* (\theta=0.10:\ 0.027).

The primary (\theta=0.05) verdict must remain unchanged, and (\theta=0.10) must not be promoted post hoc. But the spread shows that operational background contamination is strongly threshold-dependent, particularly in low-ρ noisy cells. This is directly relevant to the “no new tuned constants” motivation.

### 6.4 Automatic rarity orientation failed where expected

The registered descriptive automatic-orientation results should be disclosed. At (\rho=0.7), the rarity rule systematically selects the complementary direction:

* RC automatic-orientation MAE: approximately 0.357.
* SC automatic-orientation MAE: approximately 0.369–0.400.

This is theoretically expected because rarity cannot distinguish ρ from (1-\rho), but the result should not be summarized merely as “untested.” It was tested descriptively and failed as a directed orientation procedure above ρ=0.5. The unordered estimate remains a separate, potentially useful object.

### 6.5 Same-bank dependence remains important

The 24 fresh seeds vary pair directions, feature bases, sampling, injection streams, and noise, but the RC experiment reuses one GPT-2 layer-6 activation bank. The precision across seeds therefore does not establish bank-, layer-, model-, or corpus-level generalization. This limitation is disclosed and should remain adjacent to any “decisive” wording.

---

## Required changes

1. **Correct formation arithmetic:** SC 254/288 included; overall 350/384 included.

2. **Correct the baseline range:** use approximately 0.083–0.494 across the leaky/noisy cells, or explicitly define a narrower subset that supports 0.13–0.49.

3. **Correct P5 reporting:** restore the registered 0.10 descriptive threshold, state the eligibility conditions and eligible cells, and report the (\rho=0.3,\sigma=0.1) bias of +0.071 rather than claiming all nonexception cells are within 0.05.

4. **Replace the operational overclaim:** remove “exactly the predicted background bias and nothing else.” Distinguish the frozen a-priori prediction from the post-run measured algebraic decomposition, and retain P1O/P2O as inconclusive.

5. **Reframe P3 neutrally:** keep “FALSIFIED” as the lead result; describe eligibility misprediction as a post-hoc diagnosis. Replace “not by the estimator” with the factual statement that the estimator remained more accurate but did not meet the registered margin. Label the deterministic-host explanation as a hypothesis.

6. **Tighten the scope language in both `SUMMARY.md` and `report.md`:** replace “fixed,” “solved,” and unqualified “confirmed decisively” with a statement about passing the oracle-scoped mechanism endpoint in the preregistered harnesses and activation bank.

7. **Replace “zero tuned constants”:** say “no new harness-tuned constants beyond the inherited (\theta=0.05), under unit decoder normalization.”

8. **Repair (h_B) reporting:** use NA for zero-denominator cells, include B-active mass or (w_B), and do not use unstable zero values to establish the observed range.

9. **Add the missing registered descriptive report:** bootstrap intervals, signed bias, RMSE, threshold sensitivity, automatic orientation/unordered estimates, tie rates, P4 contributor counts, realized-ρ discrepancies, and P5 eligibility plus odds-prediction residuals. Implement this as a dated reporting-only addition without modifying the frozen scorer or verdicts.

10. **Make the report’s operational verdict explicit:** “14 cell-level passes, two inconclusive cells; P1O and P2O inconclusive overall.”

---

## Suggestions

1. Generate `SUMMARY.md` tables directly from a reporting script to prevent hand-transcription errors such as 335/384.

2. Add a compact decomposition table with mechanism error, background weight (w_B), (h_B), predicted operational estimate, observed operational estimate, and residual. Clearly label the post-run decomposition as diagnostic rather than predictive.

3. For a future round, preregister a second activation bank, layer, or model. That is the natural test of whether C-dom transfers beyond the current bank.

4. Harden the future P4 scorer so a cell cannot pass if either class median is missing, and preregister a minimum number of seed-level contributors in addition to the per-seed 100-token denominator.

## Final assessment

The project handled the central confirmatory outcomes substantially honestly:

* the primary mechanism predictions passed;
* both operational predictions were left inconclusive;
* P3 was explicitly marked falsified;
* the oracle and same-bank scope was disclosed.

The main deficiencies are results-stage reporting and rhetoric, not experimental validity. After the required corrections, **ACCEPT** would be appropriate without rerunning the experiment.

