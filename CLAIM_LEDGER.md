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
| 12 | Real-scale L1-vs-TopK first-letter absorption (m=16384, matched L0=32, 8 seeds/arch) | lock **`0722212`** | `analysis/analyze_round12.py` (frozen) | **P1 NOT CONFIRMED** (clean paired diff +0.0030 CI [−0.0010,+0.0067]; contamination caught by registered gates, disclosed clean re-score); P2 CONCENTRATED(+) both arches; P3 detector enrichment 11.2×/5.3× | report §20, PAPER §8b, `results/real/SUMMARY_round12.md` |
| 13a | Family (splitting-corrected) endpoint on the frozen round-12 SAEs | lock **`9728663`** (+ pre-results `0ea34f1`) | `analysis/analyze_round13a.py` (frozen) | **P1 SURVIVES** (0.0542 CI [0.0494,0.0592]; single-latent metric inflates ~25%); P2 R²=0.381 PASS (narrow); P3 heterogeneity unchanged; **P4 arch null persists** (−0.0012 CI [−0.0081,+0.0049], H2 refuted); P5 splitting L1 2.61 vs TopK 1.25 (+1.36 CI [+0.94,+1.88]) | report §20, PAPER §8b, `results/real/SUMMARY_round13a.md` |
| 13b | Capacity sweep m∈{2048,4096,16384} × {L1,TopK} × 8 seeds at matched L0 (H1) | lock **`c934d33`** (+ pre-results Amendments `7501486`) | `analysis/analyze_round13b.py` (frozen) | Gates 1–4 + manipulation check PASS; **P1 FALSIFIED-DIRECTION** (−0.0445 CI [−0.0493,−0.0397]: absorption *falls* under scarcity; H1 refuted); **P2 CONFIRMED** (+0.0070 CI [+0.0014,+0.0135], registered power caveat, opposite-regime reading); P5 confound does not fire. Blind theory notes scored: splitting-asymmetry largely HIT; matched-L0 boxed P2≈0 **FALSIFIED** | report §20, PAPER §8b, `results/real/SUMMARY_round13b.md` |
| 14 | Does "absorbed" have a carrier? (32 frozen SAEs re-analysed) | lock **`708211f`** (+ Amendment 1 `d2f32fa`) | `analysis/round14_carrier.py` + `analyze_round14.py` (frozen) | P1 CONFIRMED **but uninformative** (registered selection defect, disclosed); **P2 NOT CONFIRMED** (−0.199 CI [−0.217,−0.181] at m=16384); per-trial diagnostic: carriage is trial-specific composites, **"loss" headline WITHDRAWN** pre-publication after adversarial review | `results/real/SUMMARY_round14.md`, reviews/ |
| 15 | External-suite transfer: Gemma Scope 2 (JumpReLU, Gemma 3 1B), width/L0/layer series, letters as unit | lock **`9b8d203`** (+ Amendment 1 `7b4c03b`: in-dist oracle) | `analysis/analyze_round15.py` (frozen) | All gates PASS (in-dist FVU 0.037–0.042); **P1 CONFIRMED** (+0.0745 CI [+0.0351,+0.1188]: absorption rises with width — 13b's direction transfers across model/arch/pipeline; D4 sensitivity agrees); **P2 NOT CONFIRMED** (fam_size diff 0.0000 — splitting co-movement does NOT transfer); **P3 NOT CONFIRMED** at the material bar (CI lower +0.068 < 0.10; cell inflation +23%/+14%/+6%) | `results/real/SUMMARY_round15.md` |

**Review trail** (all in `reviews/`): round-8 external + research reviews;
round-9 dual pre-lock + dual results-stage; round-10 dual pre-lock + Gemini
results-stage; round-11 Gemini results-stage; **whole-repo review 2026-07-24**
(`WHOLE_REPO_REVIEW_GPT-5.6_2026-07-24.md`, verdict: major revision) with
responses beside each; round-13b hostile thesis defense
(`DEFENSE_round13b_2026-07-25/`, 8 attack surfaces; its Q2 Finding 1 —
endpoint construct validity — is adopted as the program's top open item).

**Honesty note carried by the reviews:** the whole-repo review states it
"found no evidence that failed results were hidden or reclassified into
successes." The identified problem is *interpretive overshoot after
exploratory results* (chiefly round 11), addressed above.
