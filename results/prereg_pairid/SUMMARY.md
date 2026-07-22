# Pair-ID Arm 1: label-free detection of gated absorbed pairs works

176 SAEs (A/CD/CDX/F/N conditions), detector v1.0 locked pre-run (pilot-
calibrated two-sided lift rule; provenance: structure `e586f02` → pilot →
lock). Scored by `analysis/analyze_prereg_pairid.py` on `arm1_runs.csv`;
weights saved (`weights_arm1.pt`).

## Registered verdicts (v1.0, confirmatory)

| # | Metric | Result | Threshold | Verdict |
|---|---|---|---|---|
| D1 | recall on absorbed A-runs (73/96 formed, disclosed) | **0.9315** | ≥ 0.90 | **PASS** |
| D2a | planted correlated-independent pair flag rate | **0.0625** | ≤ 0.10 | **PASS** |
| D2b | false-positive pairs/SAE (A,CD,F,N) | 0.1062 | ≤ 0.10 | INCONCLUSIVE (by 0.006) |
| D3 | median child-direction recovery from pair geometry | **0.9791** (orientation 100%) | > 0.9 | **PASS** |
| D4 | mean \|ρ̂−ρ\| by signature counting, σ=0 (locked scoping) | **0.0134** | ≤ 0.03 | **PASS** |

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
(`arm1_v11_rescore.csv`): D1 0.918, D2a 0.0625, **D2b 0.031** — all
thresholds met. v1.1 is locked, unchanged, for the Arm 2 held-out
confirmatory transfer (real GPT-2 activations, capacity-limited m=128/256).
Arm 1's confirmatory record remains the v1.0 table above.

## Significance

Combined with Arm A: the label-free pipeline for the §15b open problem is now
**detect the pair (cos band + two-sided lift + overlap veto) → orient by rate
→ count signatures → ρ̂**, validated end-to-end in the toy model at σ=0
(ρ̂ error 0.013) with the σ-regime boundary mapped. Companion result, same
session: `results/capacity_m33/SUMMARY.md` (absorption is capacity-scarcity
throughout; K1–K3 confirmed).
