# Round 9: gating-corrected ρ estimators — registered scoring

Prereg `notes/prereg-gating-corrected-rho.md`, **lock `b0276cc`** (bars from
D1's error budget; nothing changed after lock). Run: 384 fresh-seed runs
(SC 288 + RC 96), one spot-L4 session (us-central1-a, 36 min wall,
≈ $0.20), weights saved (`weights_r9_*.pt`), scored by the frozen
`analysis/analyze_round9.py`. Formation: **RC 96/96 formed** (0 exclusions,
including the a-priori-risky Q = 0.04 cell); **SC 254/288 included**
(per-cell exclusions 0–6, all cells ≥ 18/24 — every cell scoreable);
**total 350/384**. Registered descriptive readouts (bootstrap CIs, signed
bias, RMSE, θ-sensitivity, auto-orientation, tie rates, P4 contributor
counts, P5 eligibility + odds-formula predictions, MAE vs realized ρ,
h_B/w_B) are in `REPORTING_APPENDIX.md` (dated reporting-only addition,
required by the results-stage review; no endpoint or bar touched).

*(Revision note, same day: the first committed version of this summary
contained three numerical/reporting errors caught by the GPT-5.6
results-stage review — a formation-count arithmetic error (335→350/384),
an overstated baseline-error range, and P5 reported against a 0.05 bound
instead of the registered 0.10. Corrected below; verdicts unchanged. The
reviewer independently reproduced all six registered verdicts by rerunning
the byte-identical frozen scorer on the committed CSV.)*

## Verdicts (as registered)

| Prediction | Verdict | Detail |
|---|---|---|
| **P1M** (RC mechanism, MAE ≤ 0.03 / falsify > 0.07) | **PASS 4/4** | worst cell 0.0026 |
| **P2M** (SC mechanism) | **PASS 12/12** | worst cell 0.0021 |
| **P1O** (RC operational, ≤ 0.05 / > 0.15) | **INCONCLUSIVE overall** | ρ = 0.1 cell 0.0695, bootstrap CI [0.066, 0.073] — inside the inconclusive zone; other 3 cells pass (0.0116–0.0431); none falsified |
| **P2O** (SC operational) | **INCONCLUSIVE overall** | ρ = 0.1 σ = 0.1 cell 0.0660, CI [0.064, 0.068] — inconclusive; other 11 cells pass; none falsified |
| **P3** (beats ρ̂_C by 0.05 in 14 predefined cells) | **FALSIFIED** | 12/14 cells cleared the margin; SC σ = 0 ρ ∈ {0.1, 0.3} did not — see diagnosis |
| **P4** (median δ_J, δ_S ≤ 0.05 / > 0.10, ≥100-token denominators) | **PASS 16/16** | worst median 0.0043; contributor counts are sparse in the clean-gating σ = 0 cells (1–10 seeds with ≥100-token class-11 denominators) — disclosed in the appendix, so the σ = 0 P4 evidence is thinner than in leaky cells |
| **P5** (descriptive, registered bound \|bias\| ≤ 0.10 in eligible cells) | reported | 8 cells met both registered symmetry conditions; 7 satisfied the 0.10 expectation; SC ρ = 0.1 σ = 0.1 exceeded it (+0.125). The §4 odds-formula prediction from measured (a₀, g₁, q₀₁, q₁₀) matches observed ρ̂_X to ≤ 0.006 in every cell |

Cell-level passes are reported for transparency; the registered overall
verdicts are: **P1M, P2M, P4 pass; P1O and P2O inconclusive; P3
falsified.**

## Headline numbers

- **The registered mechanism endpoint (P1M/P2M) passed in all 16 cells in
  these oracle-scoped harnesses** — the estimator beat its pre-registered
  MAE pass bar by >10× on the P1M/P2M endpoint. (The P4 side-effect check
  also passed 16/16, but with sparse contributor counts in the clean-gating
  σ=0 cells — 1–10 seeds with adequate class-11 denominators — so the σ=0 P4
  evidence is thinner than "decisive"; disclosed in the appendix.) Given the
  pair and orientation, on parent-event tokens, ρ̂_D recovers ρ with MAE
  ≤ 0.0026 in every cell — both harnesses, ρ ∈ {0.1 … 0.7},
  σ ∈ {0, 0.05, 0.1} — with no new harness-tuned constant beyond the
  inherited θ = 0.05, under unit decoder normalization — while the leak
  structure it corrects varies wildly across cells (a₀: 0.014 → 0.67;
  g₁: 0.005 → 0.88, strongly ρ-dependent on real data). The oracle-comp
  baseline ρ̂_C errs by 0.083–0.494 across the leaky (RC and σ > 0 SC)
  cells. Transparency note (appendix): against *realized* ρ the mechanism
  MAE is ≤ 0.0002 (SC) / ≤ 0.0025 (RC) — with oracle masks and near-zero
  inversion rates the endpoint chiefly validates the token-classification
  rule (C-dom), and most residual vs *nominal* ρ is finite-eval prevalence
  variation.
- **Both operational predictions are inconclusive overall** (14 cell-level
  passes, one inconclusive cell in each harness, none falsified). The
  post-run measured background-mixture decomposition
  (1−w_B)·ρ̂_D-M + w_B·h_B accounts for the operational values with
  residual ≤ 0.0001 in every cell (appendix table) — this is a same-run
  *diagnostic* that locates the deviation in background-active tokens, not
  an a-priori prediction; the lock's a-priori estimate for the at-risk RC
  ρ = 0.1 cell was ≈ 0.045 vs 0.0695 observed. Where B-active mass is
  substantial, h_B is measured at 0.36–0.54 (RC) and ≈ 0.49–0.51 (SC
  σ = 0.1); it is NA in near-zero-background cells. The θ-sensitivity
  readout shows the operational contamination is strongly
  threshold-dependent in the noisy low-ρ cell (MAE 0.101/0.066/0.027 at
  θ = 0.02/0.05/0.10) — reported as registered, with θ = 0.05 remaining
  the frozen primary. Background removal or h_B-correction is the queued
  follow-up, not a claim.
- **P3 was falsified in 2/14 cells, as registered.** A post-hoc diagnosis:
  the eligibility model overpredicted baseline bias in the two SC σ = 0
  cells (predicted from Arm A's σ = 0 leak a₀ ≈ 0.16; this harness
  measured a₀ ≤ 0.038), so ρ̂_C was only 0.025–0.035 wrong there and the
  registered 0.05 margin was arithmetically unattainable; ρ̂_D remained
  more accurate in both cells (0.0013 vs 0.0350; 0.0021 vs 0.0245) but did
  not meet the registered comparative claim. A hypothesis for the leak
  discrepancy — the E3-family host coefficient is deterministic while
  Arm A's varied — is post-hoc and not isolated experimentally. Leak
  magnitudes failing to transfer across harnesses is consistent with
  round 8b's constant-transfer lesson. This is pre-registration doing its
  job: the margin prediction was locked, it was wrong, and the
  falsification localizes why cleanly.
- **Auto-orientation (registered descriptive) was tested and failed as a
  directed procedure above ρ = 0.5**, as theory predicts (rarity cannot
  distinguish ρ from 1−ρ): directed MAE ≈ 0.36–0.40 in the ρ = 0.7 cells
  vs ≤ 0.07 elsewhere, while the unordered estimate {ρ̂, 1−ρ̂} stays
  accurate everywhere (appendix). Tie rate: 0 in all 350 included runs.

## Scope (unchanged from the prereg)

Oracle-located, criterion-qualified pairs, oracle orientation, the reused
round-8 GPT-2 layer-6 activation bank (RC evidence is semi-synthetic and
same-bank — seed variation does not establish bank/layer/model/corpus
generalization). Not established: detection-conditioned performance,
automatic directed orientation, cross-domain transfer,
background-corrected operational estimation (h_B measured but not yet
exploited).

## Data

`r9_runs.csv` (384 rows: all estimators at O/M endpoints, θ-sensitivity,
full per-run diagnostics incl. δ_J, δ_S, q_ij-B, π_B, h_B, realized ρ),
`REPORTING_APPENDIX.md` (registered descriptive readouts),
`d1_e1_recompute.csv` (D-phase), weights for every run, `r9.log`,
`smoke_r9.log`. Scoring: `analysis/analyze_round9.py` (frozen at lock);
appendix: `analysis/round9_reporting_appendix.py` (reporting-only).
