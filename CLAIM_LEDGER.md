# Claim ledger

*Added 2026-07-24 (whole-repo review finding #9). One place mapping each
pre-registered confirmatory claim to its lock, scorer, verdict, and where it
appears in the deliverables. "Review" = LLM-assisted adversarial review
(Gemini 2.5 Pro + GPT-5.6), not human peer review. Confirmatory rounds lock a
prereg + scorer before results; exploratory rounds are labeled.*

| Round | Claim (endpoint) | Prereg / lock | Scorer | Verdict (as registered) | Where |
|---|---|---|---|---|---|
| 1–6 | ε\*(λ,q) pure-strategy crossover; non-identifiability wall; coherence no-go + p₀\*; event-weighted oracle remedy | `notes/` + `theory/verify_*.py`, `general_no_go_check.py` (machine checks, not a proof assistant) | symbolic/numeric checks | Supported **within stated scope** (2-latent/pure-candidate; no-go in the O₂ class, m≤d) | report §3–§13, PAPER §2–§13 |
| 7–8 | Label-free absorbed-**pair** detector | `notes/prereg-round8-scaling-robustness.md` (amended in-flight, disclosed) | `analysis/analyze_round8.py` | Detects **planted** pairs; natural-feature adjudication **null (0/15)** | report §17, `results/round8/` |
| 8 (S1/natfeat) | Wild natural absorption | `notes/prereg-natfeat-adjudication.md` | `analysis/natfeat_adjudicate.py` | **NULL** (max child→parent containment 0.46 ≪ 0.80) | `results/round8/natfeat_SUMMARY.md` |
| 9 | Gating-corrected ρ̂ (dominance partition) | lock **`b0276cc`** | `analysis/analyze_round9.py` (frozen) | **P1M/P2M PASS** 16/16 (MAE≤0.0026); **P1O/P2O INCONCLUSIVE**; **P3 FALSIFIED**; P4 PASS 16/16 (sparse σ=0 counts) | report §18, PAPER §8, `results/round9/SUMMARY.md` + `REPORTING_APPENDIX.md` |
| 10 | TopK absorption (2-atom ε\*=2q; capacity collapse) | lock **`f2e92fc`** | `analysis/analyze_round10.py` (frozen) | **P1 INCONCLUSIVE** (m=2 SGD degenerate); **P2 FALSIFIED**; P3 PARTIAL; **P4 direction inverted** (was "refuted") | report §19, PAPER §10, `theory/topk_absorption.md`, `results/round10/SUMMARY.md` |
| 11 | Real Pythia-1.4B SAEs; L1-vs-TopK detector counts | **exploratory, no lock** | `experiments/real_analyze.py` (post-hoc fixed) | Infra milestone (real SAEs, in-cache FVU 0.043/0.056). **Semantic "~27× redundancy" claim WITHDRAWN** (confounded); corrected confirmatory experiment queued | `results/real/SUMMARY.md` + `ARTIFACT_MANIFEST.md` |

**Review trail** (all in `reviews/`): round-8 external + research reviews;
round-9 dual pre-lock + dual results-stage; round-10 dual pre-lock + Gemini
results-stage; round-11 Gemini results-stage; **whole-repo review 2026-07-24**
(`WHOLE_REPO_REVIEW_GPT-5.6_2026-07-24.md`, verdict: major revision) with
responses beside each.

**Honesty note carried by the reviews:** the whole-repo review states it
"found no evidence that failed results were hidden or reclassified into
successes." The identified problem is *interpretive overshoot after
exploratory results* (chiefly round 11), addressed above.
