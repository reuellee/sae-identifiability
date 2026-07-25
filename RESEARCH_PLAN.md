# Research plan (living document)

*Updated 2026-07-25 (post round-13b). Every confirmatory
experiment gets a pre-registration note + pre-results commit; exploratory work
is labeled. "Review" = LLM-assisted adversarial review (Gemini + GPT), not
human peer review.*

## North star (owner, 2026-07-24)

The program's intended arc: **SAE geometry → identifiable codes → causally
valid features → reusable abstractions → improved novel-task adaptation.**
Rounds so far live at the first two stages: *geometry* (the ε\* crossover, the
coherence no-go, decoder-vs-code identifiability) and *identifiable codes* (the
detector, the gating-corrected ρ̂, and round 10's TopK/width study of when a
child code is recoverable at all). The natural pivot the results now point to
is **causal validity**: whether a "recovered" child code actually mediates the
child feature's effect (ablation/intervention), which is the bridge to reusable
abstractions and novel-task transfer. Queue items are tagged with the stage
they advance.

## Completed (provenance in README table)

| # | Result | Where |
|---|---|---|
| R1–6 | Solvable model: ε\* pure-strategy crossover, non-identifiability wall, coherence no-go + p₀\* domain, event-weighted oracle remedy, capacity-dependent regime structure on semi-synthetic GPT-2, two label-free estimators refuted | report §3–§15b |
| Arm A | Trained absorption is encoder-gated ("leaky"): dictionary absorption without code-level information loss; both pre-registered hypotheses inverted | §16 |
| m≥33 rerun | Capacity scarcity is the operative cause of the two-latent transition in this model (K1–K3) | §8, `results/capacity_m33/` |
| Pair-ID Arm 1 | Label-free detector: synthetic proof of concept (v1.0 confirmatory; only D3 CI-established; v1.1 = development-set) | §17, `results/prereg_pairid/` |
| Pair-ID Arm 2 | Held-out transfer: statistic separates cleanly, toy-locked cutoff knife-edge (8/8 m=256, 1/8 m=128, recorded as registered) | §17 |
| Reviews | GPT-5.6 ×3 rounds + research review: all revisions applied, responses archived | `reviews/` |
| Natural-feature adjudication (S1) | **Null on wild absorption:** 0/15 seed-stable candidate clusters meet asymmetric-nesting; all are correlated (typographic byte-fragment family, incl. the 4-clique) or anti-correlated linguistic-feature pairs = the CDX equivalence class. Max child→parent containment 0.46 ≪ 0.80. | `results/round8/natfeat_SUMMARY.md` |
| Round 9 | **Gating-corrected ρ̂ (dominance partition): mechanism endpoints P1M/P2M PASS 16/16 cells (MAE ≤ 0.0026 vs naive 0.25 bias); P4 inversion check PASS 16/16; P1O/P2O INCONCLUSIVE overall (14 cell-level passes + one ρ=0.1 cell per harness in the zone — measured h_B background pull, RC one disclosed a-priori); P3 margin FALSIFIED in 2 σ=0 synthetic cells (post-hoc diagnosis: eligibility model overpredicted baseline bias; ρ̂_D still more accurate there).** Lock `b0276cc`; dual pre-lock review (Gemini minor / GPT-5.6 major) + dual results-stage review (Gemini ACCEPT; GPT-5.6 minor→accept after corrections — it independently reproduced all six verdicts from the public repo). | §18, `results/round9/SUMMARY.md` + `REPORTING_APPENDIX.md` |
| Round 10 (TopK, largely NEGATIVE) | **Theory (2-atom oracle): ε\*_TopK = 2q, capacity collapse — verified (M0), incl. GPT-5.6's 3-atom zero-loss counterexample that scopes it to two atoms. SGD experiment: P1 INCONCLUSIVE, P2 FALSIFIED (the m=2 SGD arm is degenerate — high rec, non-selective atoms — not the clean 2-atom optimum), P3 PARTIAL (overcomplete TopK recovers 0.62–0.83), P4 REFUTED (L1 recovers 1.00 > TopK — the hard budget HURTS rare-feature recovery). Findings: dictionary width (not per-token k) drives recovery; isolated L1 does NOT absorb → prior L1 absorption is background-driven, not rarity alone; "TopK resists absorption" refuted.** Lock `f2e92fc`; dual pre-lock review (Gemini minor / GPT-5.6 major — reframed the round). | `theory/topk_absorption.md`, `results/round10/SUMMARY.md` |
| Round 11 (real model, EXPLORATORY) | **Graduated to real SAEs on Pythia-1.4B (A100 quota=0 → L4 + Pythia; extract→cache→train). Two matched m=16384 SAEs train to high quality (TopK FVU 0.043 / L1 FVU 0.056, L0=32).** Infra + pipeline; weights/acts in GCS. *(The exploratory "~27× redundancy / TopK resists L1's splitting-absorption" reading was subsequently **WITHDRAWN** as confounded — see `CLAIM_LEDGER.md`; superseded by the registered rounds 12–13b below.)* | `results/real/SUMMARY.md`, `experiments/real_*.py` |
| Round 12 (real model, REGISTERED) | **Causal L1-vs-TopK first-letter absorption, m=16384, matched L0=32, 8 seeds/arch: P1 NOT CONFIRMED — clean paired diff +0.0030, CI [−0.0010, +0.0067]. Contamination by a stale out-of-config file caught by the registered gates; disclosed clean re-score. P3 detector enrichment positive (L1 0.812 vs 0.153; TopK 0.333 vs 0.030). Post-hoc (exploratory): endpoint letter-concentrated and largely a selectivity re-expression → H1 (spare capacity) and H2 (splitting artifact) registered as rescues.** Lock `0722212`. | `results/real/SUMMARY_round12.md` |
| Round 13a (real model, REGISTERED) | **Family (splitting-corrected) endpoint re-score of round 12: P1 SURVIVES (0.0542, CI [0.0494, 0.0592]; single-latent metric inflates absorption ~25% via splitting — a SAEBench-relevant validity finding); P2 R²=0.381 narrow PASS; P4 arch null persists (−0.0012, CI [−0.0081, +0.0049]) → H2 REFUTED; P5: the real arch difference is splitting (L1 2.61 vs TopK 1.25 latents/letter, +1.36, CI [+0.94, +1.88]).** Lock `9728663` (+ pre-results amendment `0ea34f1`). | `results/real/SUMMARY_round13a.md` |
| Round 13b (real model, REGISTERED) | **Capacity sweep m ∈ {2048, 4096, 16384} × {L1, TopK} × 8 seeds at matched L0 (48 fresh SAEs, round-12 activations). Gates + manipulation check PASS (dead% 53.1→6.3). P1 FALSIFIED-DIRECTION: absorption falls under scarcity (−0.0445, CI [−0.0493, −0.0397], monotone in both arches; denominator + retention confounds excluded) → H1 REFUTED — round 12's null is now doubly robust. P2 CONFIRMED as registered (+0.0070, CI [+0.0014, +0.0135]) with the registered power caveat and an opposite-regime reading (gap opens where absorption is 4–8× lower; splitting the likely mechanism). P3: L1 families grow with width (1.84→2.61), TopK flat. Two blind-committed theory notes scored: splitting-asymmetry largely HIT; matched-L0 boxed P2≈0 prediction FALSIFIED. Endpoint construct validity (fragmentation, not merging?) is now the central open question.** Lock `c934d33` (+ pre-results Amendments `7501486`). | `results/real/SUMMARY_round13b.md`, `theory/matched_L0_invariance.md`, `theory/splitting_asymmetry.md` |

## Round 8 (in flight): `notes/prereg-round8-scaling-robustness.md`

| Exp | Question | Design | Status |
|---|---|---|---|
| **E1 — v1.2 held-out cutoff transfer** (confirmatory) | Does L_HI = 1.9 (calibrated on Arm 2 m=256) give high recall on FRESH real-data configs, with enough runs for a meaningful CI? | GPT-2 capacity-limited, m ∈ {128, 256}, **24 fresh seeds** (8–31) at ε=0.002 + 8 faithful controls; v1.1 otherwise unchanged | prereg locked, GPU queued |
| **E2 — width-scaling null calibration** (pre-registered descriptive) | Does FP/million-pairs fall with width (detector survives scale) or stay flat (practical precision doomed)? | (d, n_bg, m) ∈ (64,30,32)…(512,254,256); null (16 seeds) + planted-absorbed (8) per scale; FP/M, recall, compute cost vs width | prereg locked, GPU queued |
| **E3 — robustness cells** (pre-registered descriptive) | Nonorthogonal pairs (cos 0.3/0.5), prevalence ρ=0.6 (composite not rarer → orientation stress), TopK encoder | m=32 cells, 8 seeds each | prereg locked, GPU queued |
| **S1 — audit-v3 candidate stability** (exploratory, CPU) | Do Arm 2's flagged real-feature pairs recur across seeds? | Match flagged pairs across the 8 saved SAEs per width by decoder cosine | running locally |

## Real-model track (the credibility jump — highest priority)

~~**THE single highest-value next experiment (whole-repo review): the
confirmatory real-scale L1-vs-TopK comparison.**~~ **EXECUTED as rounds
12 → 13a → 13b (2026-07-24/25; Completed table above). Answer: the toy
geometry does NOT predict a real-scale L1-vs-TopK absorption difference at
matched L0 — NOT CONFIRMED (round 12), robust to the splitting correction
(13a) and to the spare-capacity rescue (13b, H1 refuted). The capacity
story is moreover inverted at real scale on this endpoint: absorption falls
monotonically as capacity shrinks. The architectures do differ — in
splitting (13a P5), and in a small scarce-regime gap (13b P2, power-capped)
most plausibly driven by splitting.**

**New top priorities (2026-07-25, post-13b):**

1. **Residual-projection construct-validity check** on the in-hand weights
   (48 round-13b + 16 round-12 files, plus 13a's family-endpoint
   artifacts). On trials the endpoint calls "absorbed", does a
   parent/composite latent's activation increase to carry the child's
   missing mass (true absorption), or does nothing compensate (fragmentation
   / threshold loss)? Post-hoc and CPU-cheap, but **decisive for what the
   endpoint means** — it adjudicates the fragmentation reading of 13b P1
   and the defense's Q2 Finding 1. Label as exploratory/post-hoc; no new
   training.
2. **Prereg a fragmentation-corrected absorption endpoint** informed by #1
   (e.g. requiring parent-mass pickup, or normalizing for family size /
   live-latent count), then re-run the architecture and capacity contrasts
   on it. Only after this does any further L1-vs-TopK absorption claim make
   sense.
3. **TopK aux-k / ghost-grads splitting prediction test**
   (`theory/splitting_asymmetry.md` §7, "TopK trap vs revival tricks"): the
   rank-gate trap predicts that a TopK stack with an aux-k loss (gradient to
   below-cutoff latents) escapes the merged local minimum and splits more,
   shrinking the 2× family-size gap. A stated, falsifiable prediction of the
   blind-committed theory note — cheap to prereg at one width.

Still live from the round-11 framing: detector validation against
ground-truth absorption labels (blocked on #1/#2 — the labels themselves are
what's in question); Gemma-2-2B (SAEBench standard) once an HF token is on
the box; scale to 8B+ only with a bigger-GPU quota grant. Single model /
single layer / single task remains the standing generalization caveat
(prereg §"what this cannot do").

## Queued (toy-model / theory track; each needs its own prereg)

1. ~~**Natural-feature evaluation** of seed-stable audit-v3 candidates.~~
   **DONE 2026-07-23 (`results/round8/natfeat_SUMMARY.md`, prereg
   `notes/prereg-natfeat-adjudication.md`, lock `0603d38`).** Null: none of the
   15 seed-stable candidates is natural absorption — they are correlated /
   anti-correlated real-feature families (the CDX class). No surviving A ⇒ no
   causal / cross-corpus confirmation to escalate. Successor idea (if the wild
   hunt is revisited): re-run the audit with a **positive-cosine + asymmetric-
   containment ≥ 0.80** gate on an **ASCII-clean / monolingual corpus** to
   suppress the byte-fragment typography family. Folds into #2/#4 below.
2. ~~**Gating-corrected counting estimator.**~~ **DONE 2026-07-23 as
   round 9** (Completed table; `results/round9/SUMMARY.md`). Successors
   spawned: (a) **h_B-corrected / background-excluded operational
   estimator** — h_B is measured per pair (0.0–0.54, not a constant) and
   the all-token bias matches w_B(h_B − ρ) exactly; exploit it. (b) The
   estimator's swap-equivariance (ρ̂_D → 1 − ρ̂_D) as an **orientation**
   signal under a prevalence prior → folds into #2b. (c) Eligibility/leak
   prediction does not transfer across harnesses even at σ = 0 (P3's
   registered falsification) — any future cross-harness prereg must
   measure, not proxy, per-harness leak in its D-phase.
2b. ~~**Containment-based orientation.**~~ **TRIED AND FALSIFIED
   2026-07-23** (`results/round8/orientation_SUMMARY.md`, prereg
   `notes/prereg-containment-orientation.md`, lock `0a6db51`, S2 results
   `72a3dc7`). Plain firing-containment: perfect when determinate but
   indeterminate 54–79% of E1's own regime. Amendment 1 (magnitude
   cross-delta) fixed E1 coverage (1.000, development-set) but **P1
   falsified on the confirmatory ρ=0.6 GPU stress cell: 0.095 accuracy**
   (below chance) — the fix does not transfer across harnesses/widths;
   MARGIN_MAG was scaled to E1's swing size, ~5–10× larger than the m=32
   synthetic harness produces. Orientation at ρ≥0.5 is still open. Next
   idea (untested): a harness-adaptive or scale-invariant (rank/
   likelihood-ratio) version of the cross-delta statistic, not a retuned
   constant.
3. **Noise-mechanism pre-registration**: retain activation histograms; test the
   σ≥0.2 absorption-destruction and the σ≈0.1 GMM-calibration mystery from
   Arm A. (1 session)
4. **Encoder/objective robustness at full depth**: TopK/JumpReLU ε\*-analogue —
   does the transition survive without L1 shrinkage, and where? (Theory first:
   TopK has no λ, so the L1 crossover doesn't port; capacity competition
   remains. 1 session after theory note.)
5. **Overcomplete no-go extension** (m > d): the reviewer-flagged open corner of
   `theory/general_no_go.md`. (theory)
6. **Full (ε, β) phase diagram + symbolic ε\*\*(λ, q, β)** (confirmatory polish;
   low priority — no conclusion depends on it).
7. **Write-up decision** (owner): LW/AF post or arXiv note. Repo is
   preprint-circulatable per external review; blocked only on owner choice.

## Standing constraints

- Confirmatory claims: prereg → pre-results commit → run frozen → report as
  registered (failures included). Post-hoc refinements are development-set
  results until they pass held-out data.
- Seed counts: 16/cell was too few to CI-establish 0.90-level recalls; size
  confirmatory cells ≥ 24 seeds and report the pre-registered 10k seed
  bootstrap alongside point estimates.
- Every GPU session: batched programs, weights SAVED, env pinned in run log,
  box stopped/deleted by the pipeline, results → `results/<round>/` + commit.
- Claim guardrails: `reviews/EXTERNAL_REVIEW_GPT-5.6_2026-07-22.md` §5 and the
  research review §13 checklist govern language.
