# Pair-ID Arm 1: label-free detection of gated absorbed pairs works

176 SAEs (A/CD/CDX/F/N conditions), detector v1.0 locked pre-run (pilot-
calibrated two-sided lift rule; provenance: structure `e586f02` → pilot →
lock). Scored by `analysis/analyze_prereg_pairid.py` on `arm1_runs.csv`;
weights saved (`weights_arm1.pt`).

## Registered verdicts (v1.0, confirmatory)

| # | Metric | Point | 10k seed-boot 95% CI | Threshold | Verdict |
|---|---|---|---|---|---|
| D1 | recall on absorbed A-runs (73/96 formed, disclosed) | **0.9315** | [0.851, 1.000] | ≥ 0.90 | point PASS; CI does not establish |
| D2a | planted correlated-independent pair flag rate | **0.0625** | [0.000, 0.156] | ≤ 0.10 | point PASS; CI does not establish |
| D2b | false-positive pairs/SAE (A,CD,F,N) | 0.1062 | [0.062, 0.150] | ≤ 0.10 | INCONCLUSIVE (by 0.006) |
| D3 | median child-direction recovery from pair geometry | **0.9791** (orientation 100%) | [0.976, 0.983] | > 0.9 | **PASS (CI-established)** |
| D4 | mean \|ρ̂−ρ\| by signature counting, σ=0 (locked scoping) | **0.0134** | [0.002, 0.032] | ≤ 0.03 | point PASS; CI does not establish |

*(CIs: pre-registered 10,000-draw seed bootstrap, np seed 0 — implemented in
`analysis/analyze_prereg_pairid.py` after external research review flagged the
omission. With 16 seeds/cell, only D3 is established at population level; the
other metrics passed as point estimates. Scaling context, same script: v1.0
FP ≈ 214/million candidate pairs at m=32; precision 0.81/0.30/0.04 at assumed
absorbed-pair prevalence 1e-3/1e-4/1e-5 — production-width null calibration
required before practical-use claims. D3 additionally validates
implementation + pair-ID jointly in the MATCHED ORTHOGONAL synthetic model
only.)*

The two-sided lift rule performed exactly as pilot-calibrated: σ=0 absorbed
pairs flag via lift ≪ 1 (gating), σ=0.1 via lift ≈ 3 (leak coupling); recall
misses live in the dead zone (0.5, 2.0) where partial leak lands ~7% of σ=0
runs. G5 (exploratory): the exclusive-correlated "equivalence class" was
flagged only 37.5% — trained encoder cross-talk blurs the idealized
exclusivity, echoing Arm A's gating discovery.

## D2b diagnosis → detector v1.1 (amendment, locked for Arm 2)

Every null-condition FP (0.81/SAE, the entire D2b overage) is a
**feature-splitting doublet**: a spare latent partially duplicating a
background feature (cos 0.7–0.87 to it) and co-firing with its main latent at
lift 10–12.6. Splits have a signature absorbed pairs lack — **containment**
(overlap = P(∧)/min(P) ≈ 0.95–1.0 vs ≤ 0.81 for true pairs). v1.1 adds
`overlap < 0.9`; re-scored on Arm 1 as disclosed training data
(`arm1_v11_rescore.csv`): D1 0.918, D2a 0.0625, **D2b 0.031**. **These v1.1
numbers are development-set performance, not confirmation** — v1.0 exposed
the feature-splitting confound, v1.1 was derived from it and evaluated on the
same data. v1.1 was frozen (commit `06d3005`-adjacent, pre-Arm-2) for the
Arm 2 held-out transfer (real GPT-2 activations, capacity-limited m=128/256),
which ran without any tuning. Arm 1's confirmatory record remains the v1.0
table above.

## Significance

Combined with Arm A: the label-free pipeline for the §15b open problem is now
**detect the pair (cos band + two-sided lift + overlap veto) → orient by rate
→ count signatures → ρ̂**, validated end-to-end in the toy model at σ=0
(ρ̂ error 0.013) with the σ-regime boundary mapped. Companion result, same
session: `results/capacity_m33/SUMMARY.md` (absorption is capacity-scarcity
throughout; K1–K3 confirmed).

---

# Arm 2 (held-out transfer, real GPT-2 activations): statistic transfers, cutoff lands on a knife edge

32 vanilla capacity-limited SAEs (m ∈ {128, 256} × ε ∈ {0.002 absorbed, 0.05
faithful} × 8 seeds, §15's exact setup), detector v1.1 transferred UNCHANGED
(`arm2_runs.csv`, weights saved). Registered readouts:

| # | Readout | Result |
|---|---|---|
| R1 | recall on absorbed ε=0.002 runs (16/16 formed, 8/8 each m) | **0.56 overall — but m=256: 8/8, m=128: 1/8.** Every m=128 true pair sits at lift 1.95–1.99, every m=256 at 2.04–2.08: the real-data absorbed-pair statistic concentrates at **lift ≈ 2.00 ± 0.05**, and the toy-locked cutoff L_HI = 2.0 lands in the middle of it. A threshold-margin failure, not a structural one — the pairs are exactly where the leak-coupled regime predicts, ~3% from the cutoff. |
| R2 | faithful-control (ε=0.05) oracle-pair flag rate | **0.000** — pair cos 0.27–0.31, far outside the band; specificity transfers perfectly. |
| R3 | audit-v3 descriptive scan | 3.2 flags/SAE (m=128), 10.6 (m=256) — 0.03% of pairs; on real background features these are *candidates*, not false positives, and constitute the first natural-absorption candidate list from this program (follow-up: inspect). |
| R4 | ρ̂ on flagged pairs (true child-given-parent = 0.5; descriptive) | 0.750 ± 0.005 — inflated exactly as the leak-regime analysis predicts (counting assumes gating; real activations are leaky). Child residual recovery bimodal: 0.99 (5/9) / 0.66 (4/9). |

**Honest verdict:** confirmatory transfer PASSES at m=256, FAILS at m=128 by
the registered letter. No post-hoc threshold change is applied to these
claims. Exploratory note, clearly labeled: L_HI = 1.9 would flag 16/16 with
R2 still 0.000; confirming that requires new data (a v1.2 pre-registration —
different layer/seeds/widths — is the obvious next step). The transfer
evidence is that the *statistic* (cos band + two-sided lift + overlap veto)
separates absorbed from faithful cleanly on real activations; the *cutoff*
needs one recalibration step ported from toy to real scale.
