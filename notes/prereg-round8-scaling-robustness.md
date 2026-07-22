# Pre-registration, round 8: v1.2 cutoff transfer, width-scaling null calibration, robustness cells

**Date:** 2026-07-22. **Status:** locked before the GPU session; this note +
scripts are the pre-results commit. Successor to
`notes/prereg-pair-identification.md` (Arm 1/Arm 2 outcomes recorded there)
and the research review's §12 queue.

## E1 — v1.2 held-out cutoff transfer (confirmatory)

**Calibration source, disclosed:** Arm 2 (m=256 passed at lift 2.04–2.08;
m=128 failed at 1.95–1.99; faithful controls lift 1.63–1.77 but cos 0.27–0.31,
excluded by the cosine band). **The single change from v1.1: L_HI = 2.0 →
1.9.** All other parameters identical (C ∈ [0.45, 0.90], L_LO = 0.5,
overlap < 0.9, θ = 0.05, rate window [5e-4, 0.6]). Arm 2 is therefore
calibration data for v1.2; E1 is fresh, held-out data.

**Design:** §15/Arm-2 setup exactly (GPT-2-small layer-6 activations, AMP=5,
Q=P0=0.2, λ=1.0, 20k steps); m ∈ {128, 256}; **ε=0.002 with 24 FRESH seeds
(8–31 → new injected pair directions)** = 48 absorbed-expected runs; ε=0.05
faithful controls, seeds 8–15 (16 runs).

**Registered thresholds:**
- **T1 (recall):** point recall ≥ 0.90 over absorbed-formed runs
  (cos_comp > 0.98 filter, disclosed); falsified < 0.70. Report the 10k
  seed-bootstrap CI alongside; with ~48 runs a lower bound ≥ 0.75 is the
  stated aspiration, not a pass condition.
- **T2 (specificity):** faithful-control oracle-pair flag rate ≤ 0.10.
- **T3 (descriptive):** flags/SAE and per-million-pairs on the real-feature
  background (audit-v3 candidates, not thresholded); ρ̂ reported vs true 0.5
  (expected inflated — the gating-corrected estimator is queued separately).

**Decision rule:** T1 ∧ T2 pass → the detector's real-data recall claim is
confirmed on held-out data (still scoped: semi-synthetic pairs, this layer,
these widths). T1 fail → the lift statistic's real-data concentration
(2.00 ± 0.05 in Arm 2) did not generalize; report and stop tuning cutoffs —
the next move would be a distribution-calibrated threshold, new prereg.

## E2 — width-scaling null calibration (pre-registered descriptive)

**Question:** FP per million candidate pairs vs width. One soft, falsifiable
expectation, stated now: v1.1's null FPs were O(1) per SAE (splitting
doublets scale with *spare slots*, here fixed at 2), while candidate pairs
grow ~m²/2 — so **FP/M should FALL with width**. If FP/M is flat or rising,
practical precision at production scale is unreachable for this detector.

**Design:** scales (d, n_bg, m) ∈ {(64,30,32), (128,62,64), (256,126,128),
(512,254,256)}, BG_RATE=0.08, Arm-A generative model. Per scale: **null**
(background only, 16 seeds) and **absorbed** (planted pair, ρ=0.10, σ=0.1 —
the leak regime, the harder branch; 8 seeds). Detector v1.1 unchanged
(synthetic thresholds). Metrics: FP/SAE, FP/M pairs (null), recall over
absorbed-formed (absorbed), formation rate (disclosed), wall-clock + peak
memory per scale. No pass/fail — calibration study.

## E3 — robustness cells (pre-registered descriptive; m=32, σ=0.1, 8 seeds/cell)

1. **Nonorthogonal child**, cos(v_p, v_c') ∈ {0.3, 0.5}: predicted pair cosine
   √((1+c)/2) = 0.806 / 0.866 — inside the band. **Registered D3-analogue:**
   residual recovery target is the *orthogonalized* child
   u\* = normalize(v_c' − (v_c'·v_p)v_p) (the residual construction cannot
   recover the parent-parallel component, by definition); report cos(u, u\*).
2. **Prevalence stress**, ρ = 0.6: composite becomes the COMMONER latent of the
   pair; registered expectation: rate-based orientation FAILS (accuracy well
   below 1), detection may still fire. Orientation needs a better rule
   (containment direction) — this cell measures the damage first.
3. **TopK encoder** (k=4, λ=0): exploratory — does gated absorption and the
   detector signature survive a hard-sparsity encoder with no L1 shrinkage?

## S1 — audit-v3 candidate stability (exploratory, CPU, from saved Arm 2 weights)

For each width: collect flagged non-planted pairs across the 8 seed-SAEs
(same background activations, different init + injected pair); match pairs
across seeds by decoder cosine (both members > 0.9 to a counterpart);
report the count of pairs stable in ≥ 4/8 seeds. Stable pairs = shortlist
for the queued natural-feature evaluation.

## Analysis

`analysis/analyze_round8.py`, committed with this note; 10k seed bootstrap
(np seed 0) for E1; all exclusions disclosed; no threshold changes after the
run under any outcome.

---

## AMENDMENT (2026-07-22, committed BEFORE round-8 collection; run in flight)

Adopted from the round-8 external review, before any round-8 result has been
seen (the GPU session is mid-run; collection has not occurred):

1. **E1 endpoints are width-specific (P0).** T1 splits into
   **T1a: recall(m=128) ≥ 0.90** and **T1b: recall(m=256) ≥ 0.90**, each
   confirmatory with its own seed-bootstrap CI; the pooled number is
   secondary and cannot override a failed width. Falsifier per width: < 0.70.
2. **E1 scope language.** E1 is a **fresh-run resampling-stability test of
   v1.2 within the same GPT-2 domain** (same model, layer, corpus, injection
   protocol) — not cross-domain transfer. Arm 2's L_HI=2.0 outcome stands as
   the completed pre-registered result; v1.2 is a new detector version, not a
   retrospective correction.
3. **Orientation and downstream stages become separate endpoints (P1).**
   Reported per width, conditional on detection: (a) orientation accuracy
   (derivable from recorded rates: detected composite = rarer member; correct
   iff it equals the oracle composite index); (b) child-residual cosine under
   automatic orientation AND under oracle orientation (recomputed from saved
   weights); (c) ρ̂ error conditional on orientation correctness. No
   confirmatory thresholds on (a)–(c) this round — measured endpoints,
   thresholds to be set in a future prereg once baseline rates exist.
4. **All-pairs specificity reporting (P0, partial).** From the in-flight run:
   flags/M pairs and the proportion of faithful-control SAEs with ≥ 1
   full-scan flag, per width, reported alongside the (necessary but
   insufficient) oracle-pair specificity. The missing condition — a
   **no-injection real-background null** — cannot be added mid-run and is
   queued as **round 8b**, together with a fixed-(d, n_bg) width sweep and an
   overcomplete (m > d) setting (the review is right that E2 conflates width
   with scale; E2 is interpreted as a proportional-scale family only).
5. **S1 matcher corrected before first use (P0).** The seed-0-anchored,
   non-bijective matcher in `analysis/s1_candidate_stability.py` is replaced
   (same commit) by clustering over all candidates from all seeds with the
   bijective pair score
   `max(min(|d_i·e_k|,|d_j·e_l|), min(|d_i·e_l|,|d_j·e_k|)) > 0.9`,
   at most one candidate per seed per cluster. S1 has not yet run (it awaits
   the activations file from the collection), so no stability result predates
   this fix.
6. **Terminology.** "Natural-absorption candidate list" → **"real-background
   candidate list"** (candidates in SAEs trained on real activations; their
   status as natural absorbed pairs is unknown pending adjudication).
