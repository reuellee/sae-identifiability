# Round-10 results-stage review — Gemini 2.5 Pro (Vertex), 2026-07-24

*Verdict: ACCEPT. (agy-Gemini with repo access was attempted per the owner's
suggestion but failed to recover a response; ran via stream_gemini.py with the
SUMMARY + frozen scorer output embedded.) Validated: scoring faithful, the
m=2-degenerate framing is a legitimate SGD-reachability finding not an excuse,
the background-driven-absorption inference is sound, P4 refutation is real (not a
selectivity-threshold artifact), no bug. Full text below.*

Excellent, this is a model of how to present a negative result. The summary is clear, directly confronts the failed predictions, and uses the failure to derive new, more refined hypotheses. My adversarial review will focus on stress-testing the interpretations and ensuring no subtle excuses are slipping through.

**Verdict: ACCEPT**

The `SUMMARY.md` is an honest, correctly-scoped, and scientifically valuable report of a negative round. The analysis correctly identifies why the predictions failed and uses the results to generate sharp, actionable new hypotheses. The negative results are not explained away; they are diagnosed, and the diagnosis itself becomes a key finding. This is exemplary.

Below is the detailed adversarial review, structured by your questions.

---

### Adversarial Review

#### 1. Is the scoring faithful to the locked bars (any post-hoc softening)?

**Assessment:** The scoring is 100% faithful to the locked pre-registration. The verdicts are reported as failures without any attempt to soften them.

*   **P1 (INCONCLUSIVE):** The `eps_mid` calculation requires recovery to cross 0.5. The data shows it never does (max recovery ~0.25). Declaring this "INCONCLUSIVE" due to a right-censored metric is the only mechanically correct option under the locked plan. It is not an excuse; it's a direct consequence of the massive failure of Arm A to learn anything useful.
*   **P2 (FALSIFIED):** The prediction was that `k=2` would enable recovery where `k=1` failed. The data shows neither recovered, and the gap was effectively zero or negative. This is a clear falsification, reported correctly.
*   **P3 (PARTIAL):** The bar was recovery `> 0.75` for all `ε`. The data shows recovery at `ε=0.1` was `0.62`. This is a clear-cut failure to meet the pre-specified bar, correctly reported as PARTIAL.
*   **P4 (REFUTED, inverted):** The hypothesis was that TopK would recover *more* than L1. The data shows the opposite, with a large negative delta (`-0.17` to `-0.38`). This is a direct, strong refutation, reported correctly.

The author is to be commended for the discipline here. The "What actually happened" section serves as a *diagnosis* of the failure, not a re-litigation of the verdict. This is precisely the right way to handle a negative result.

#### 2. Is the "m=2 is degenerate" framing legitimate, or an excuse?

**Assessment:** It is a legitimate and crucial scientific finding, not an excuse.

The framing would be an excuse if it were used to dismiss the result (e.g., "the experiment was broken, so we can't conclude anything"). Instead, the author frames it as: "The experiment to test the 2-atom theorem failed because SGD, under these specific constraints (`m=2`), cannot find the clean optima the theorem describes. *This failure of SGD reachability is itself the finding.*"

The evidence is dispositive:
*   **High Reconstruction Loss:** A `rec` of ~0.5 means the dictionary is failing to explain half the variance of the simple 2-feature input data. A clean solution (faithful or absorbed) would have `rec` near 0.
*   **Non-Selective Atoms:** The fact that the "child" atom fires almost as often on parent-only data (`~0.6`) as on child-only data (`~0.67`) is damning. It proves the atom has not learned the child feature; it has learned some smeared, non-identifiable concept. It fails the pre-registered selectivity metric (`diff >= 0.3`) by a mile.

This is not dodging a negative result about the theorem. It is correctly identifying that the experimental conditions required to test the theorem were not met by the training process. The conclusion rightly shifts from "the theorem is wrong" to "the theorem's global optima are not reachable by SGD in this highly constrained `m=2` regime." This is a more precise and valuable conclusion.

#### 3. Is the "isolated L1 doesn't absorb => prior L1 absorption is background-driven" inference sound?

**Assessment:** The inference is sound and is one of the most important takeaways from this round.

This is a strong example of using a null result to falsify a prior, simpler hypothesis. The logic is a direct comparison:
*   **Prior Rounds:** L1 SAE + Parent/Child + Background Features (`n_bg=30`) -> Child feature is absorbed.
*   **This Round:** L1 SAE + Parent/Child + No Background Features (`n_bg=0`) -> Child feature is recovered perfectly (1.00 recovery).

The only significant variable changed between these conditions is the presence of the background features. Therefore, concluding that background competition is a necessary driver for the previously observed L1 absorption is a well-founded inference. It refutes the simpler hypothesis that child rarity (`ε`) alone is sufficient to cause absorption in L1 SAEs. This sharpens the project's understanding of the absorption phenomenon significantly.

#### 4. Is the P4 refutation (L1 > TopK) interpreted correctly, or could it be a metric artifact?

**Assessment:** The interpretation is correct and does not appear to be a metric artifact.

The potential artifact would be if the TopK atoms were *just barely* failing the recovery metric, for instance, if the selectivity threshold of `0.3` was the problem. Let's check the numbers:
*   TopK `fire_psolo` (firing on parent-solo) is `0.13-0.29`.
*   TopK `fire_csolo` (firing on child-solo) is `0.96-1.00`.
*   The selectivity metric is `fire_csolo - fire_psolo`. In the worst case for selectivity, this is `0.96 - 0.29 = 0.67`. This is more than double the required `0.3` threshold.

Therefore, when a TopK atom is found, it is cleanly selective. The lower recovery rates (`0.62-0.83`) mean that in a non-trivial fraction of runs, *no atom meeting the criteria was formed at all*. The author's interpretation—that the hard TopK budget provides fewer training signals for the rare child feature, leading to a less robust learning process—is a very plausible explanation for this higher failure rate compared to L1. The finding is real.

#### 5. Anything in the numbers that looks like a bug vs. a real effect?

**Assessment:** No, the numbers appear internally consistent and reflect real, albeit unexpected, effects.

The `r10_runs.csv` file structure is as expected. The results are stark but consistent: Arm A is a total failure, Arm C is a total success, and Arm B is a partial success. The high reconstruction loss in Arm A is not a bug in the code, but a measured outcome of the "degenerate SGD" phenomenon. The perfect `1.00` recovery in Arm C (L1) is surprising but plausible in the absence of background competition. The slightly degraded performance of Arm B (TopK) vs. Arm C is the key finding for P4. There are no red flags here that suggest a technical error over a genuine experimental result.

#### 6. What is the single most valuable next experiment?

**Assessment:** The single most valuable next experiment is the one proposed by the author: **an L1-vs-TopK head-to-head comparison *with background features included*.**

*   **Why:** This round established two critical facts: 1) L1 absorption is driven by background competition, not rarity alone. 2) In isolation, TopK's hard budget *hurts* rare feature recovery compared to L1.
*   **The Open Question:** The original hypothesis ("TopK resists L1 absorption") was formulated for a regime where L1 absorption *actually happens*. This round tested it in a regime where it doesn't. The crucial, unanswered question is what happens in the presence of background competition.
*   **Two Possibilities:**
    1.  The effect from this round holds: TopK's hard budget continues to provide fewer training signals, hurting the child atom and potentially leading to *even worse* absorption than L1.
    2.  The original hypothesis holds: The hard TopK budget forces a choice. An atom must either represent the child cleanly or represent background features. It cannot easily form a "good enough" composite feature that explains a bit of the child and a bit of the background, which is the hypothesized mechanism for L1 absorption. In this scenario, TopK would be *more* identifiable than L1.

Disambiguating between these two outcomes is the most direct path to understanding the practical impact of SAE architecture on feature identifiability, which is central to the project's north star. The proposed experiment (b) in the summary is the correct next step.
