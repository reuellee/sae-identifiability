# Research plan (living document)

*Updated 2026-07-24 (post round-10). Every confirmatory experiment gets a
pre-registration note + pre-results commit; exploratory work is labeled.*

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

## Round 8 (in flight): `notes/prereg-round8-scaling-robustness.md`

| Exp | Question | Design | Status |
|---|---|---|---|
| **E1 — v1.2 held-out cutoff transfer** (confirmatory) | Does L_HI = 1.9 (calibrated on Arm 2 m=256) give high recall on FRESH real-data configs, with enough runs for a meaningful CI? | GPT-2 capacity-limited, m ∈ {128, 256}, **24 fresh seeds** (8–31) at ε=0.002 + 8 faithful controls; v1.1 otherwise unchanged | prereg locked, GPU queued |
| **E2 — width-scaling null calibration** (pre-registered descriptive) | Does FP/million-pairs fall with width (detector survives scale) or stay flat (practical precision doomed)? | (d, n_bg, m) ∈ (64,30,32)…(512,254,256); null (16 seeds) + planted-absorbed (8) per scale; FP/M, recall, compute cost vs width | prereg locked, GPU queued |
| **E3 — robustness cells** (pre-registered descriptive) | Nonorthogonal pairs (cos 0.3/0.5), prevalence ρ=0.6 (composite not rarer → orientation stress), TopK encoder | m=32 cells, 8 seeds each | prereg locked, GPU queued |
| **S1 — audit-v3 candidate stability** (exploratory, CPU) | Do Arm 2's flagged real-feature pairs recur across seeds? | Match flagged pairs across the 8 saved SAEs per width by decoder cosine | running locally |

## Queued (priority order; each needs its own prereg)

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
