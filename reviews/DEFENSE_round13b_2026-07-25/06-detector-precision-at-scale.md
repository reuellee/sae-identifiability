# Defense round13b — Q6: detector-precision-at-scale

_Examiner: Gemini 2.5 Pro via gemx. Generated 2026-07-25T16:41:44Z. Single-shot._

### Thesis Defense Interrogation — Detector Utility & Precision

Candidate, you have presented a label-free pair detector as a significant deliverable of your empirical work, framing it as a "synthetic proof of concept." Under close scrutiny, this framing is a semantic euphemism for a tool that is **practically unusable** and suffers from a complete precision collapse when confronted with real-world representation spaces. 

Your reported "statistically real enrichment over baseline" is an academic ornament; for any actual engineering or interpretability workflow, your detector is a noise generator.

---

### PART I: ADJUDICATION OF DETECTOR UTILITY

#### Finding 1: The "Synthetic Proof of Concept" is a Semantic Euphemism for $0\%$ Real-World Precision
*   **The Claim:** 
    > "and a detector combining decoder geometry with code co-firing identifies planted parent/composite pairs on matched synthetic data and partially transfers, with strong width dependence, to semi-synthetic GPT-2 activations. The detector remains a synthetic proof of concept: all-pairs specificity on real backgrounds, orientation, scaling, and cutoff transfer are open, pre-registered problems." (Abstract)
*   **Why it fails / is unsupported:** 
    The term "proof of concept" implies a path toward viability. However, your own data demonstrates that the detector fails to isolate mechanistic absorption in the wild, collapsing under two distinct failures:
    1.  **The Real-Background Null ($0\%$ Precision):** In your pre-registered natural-feature adjudication (Round 8), you scanned un-injected real background activations and identified 15 stable candidate clusters. When evaluated against your asymmetric-nesting containment criteria ($C \ge 0.80$), **not a single candidate qualified (0/15)**. They instead split into typographic families (UTF-8 tokenization artifacts) and anti-correlated linguistic pairs (the CDX co-occurrence class). On raw background activations, your detector has exactly **$0\%$ precision**—it is a co-occurrence detector, not an absorption detector.
    2.  **The Combinatorial Scaling Catastrophe:** For a real-world SAE of width $m = 16,384$, there are $\approx 134.2$ million candidate pairs. Even using your optimistic synthetic false-positive rate of $\approx 214$ per million pairs, a full-scan will yield **over 28,700 false positives** per SAE. 
    3.  **The Prevalence Collision:** If the true prevalence of absorbed pairs in a model is a generous $10^{-5}$ (~1,340 true pairs), your synthetic false positive rate yields a precision of only **$4.5\%$**. At a more realistic prevalence of $10^{-6}$ (~134 true pairs), the precision collapses to **$0.46\%$**. A practitioner using this tool would have to manually audit over 200 false positives (all typographic or linguistic co-firings) to find a single true absorbed latent pair.
*   **Specific Evidence / Experiment to settle it:** 
    To support even the "proof of concept" framing, the detector must incorporate your post-hoc **asymmetric-containment gate** ($C(\text{parent}\mid\text{child}) \ge 0.80$ and $C(\text{child}\mid\text{parent}) < 0.80$) directly into its online filtering pipeline, and the authors must run an un-injected full scan on an ASCII-clean corpus to prove that the false positive rate is suppressed to a level that yields a non-zero, practically auditable precision.
*   **Severity:** **MAJOR**
*   **Classification:** `[PRE-RESULTS-OK]` (This is actionable by revising the paper's Abstract, Section 8, and Section 10 to strip away any implication of practical viability, explicitly state the $0\%$ real-background precision, and present the multiple-comparisons precision collapse as an fundamental scaling limit rather than an "open problem").

---

### PART II: THE DISTINCTION BETWEEN STATISTICAL ENRICHMENT AND PRACTICAL UTILITY

As an expert in experimental statistics, I must force you to distinguish clearly between **statistical enrichment** and **practical utility**:

| Metric | Enrichment over Baseline (Statistical Reality) | Usable Tool (Engineering Utility) |
|---|---|---|
| **What it measures** | Whether the detector's true-positive rate exceeds random chance ($P(\text{flag} \mid \text{absorbed}) > P(\text{flag})$). | Whether a flagged pair is actually absorbed ($P(\text{absorbed} \mid \text{flag})$ — i.e., Precision). |
| **Real-Data Value** | **$5.3\times$ to $11.2\times$ enrichment** (Round 12 P3: TopK recall 0.333 vs 0.030 baseline; L1 recall 0.812 vs 0.153 baseline). | **$0.7\%$ to $2.5\%$ precision** (Round 12 P3) and **$0\%$ precision** on un-injected real backgrounds (Round 8). |
| **Scientific Meaning** | The geometric and co-firing signals carry real information about the joint firing structure of features. | The signals are shared by an enormous background of typographic and semantic co-firing pairs (the CDX equivalence class). |
| **Practitioner Impact** | High-confidence rejection of a pure independent null hypothesis. | A complete operational failure. The practitioner is buried under tens of thousands of false alarms. |

### Conclusion for the Candidate
The detector is a valuable diagnostic tool *within a controlled semi-synthetic simulator* to verify that gating and geometry survive model scale. However, **as a label-free tool for the wild, it is a failure.** 

Your manuscript must be revised to reflect this distinction. You must demote the "detector arc" from a proposed "solution" to a "quantified baseline of co-occurrence signatures," documenting precisely why standard geometric and co-firing statistics are structurally incapable of resolving semantic absorption without an oracle or an explicit containment filter.

Defend your framing, candidate.
