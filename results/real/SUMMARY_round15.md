# Round 15 — the width→absorption direction reproduces on Gemma Scope 2; the splitting co-movement does not

**Prereg `notes/prereg-round15-gemmascope-crossval.md`, LOCK `83acf67`, Amendment 1
(PRE-RESULTS) `b925580`. Frozen evaluator `analysis/analyze_round15.py` output:
`results_round15.txt` (verbatim below). 8 public Gemma Scope 2 JumpReLU SAEs on
Gemma 3 1B resid_post; no training; letters as the unit (single-seed suite);
dual pre-lock review (Gemini + GPT) in `reviews/`.**

## Registered verdicts (verbatim, from the frozen evaluator)

```
gates: g0_frozen_config ✓  g0_cells_present ✓  conform_16k/65k/262k ✓
       (fvu_indist 0.0419 / 0.0392 / 0.0367)  letters 24/22/24, inter 23 ✓
P1 (PRIMARY) rate_family 262k-16k: CONFIRMED
    (mean=+0.0745, CI [+0.0351,+0.1188], n_letters=23, positive 17/23)
P2 (KEY SECONDARY) fam_size_uncapped 262k-16k: NOT CONFIRMED
    (mean=+0.0000, CI [-0.3478,+0.3913], n_letters=23, positive 4/23)
P3 (secondary) relative inflation (bar 0.10): NOT CONFIRMED
    (mean=+0.351, CI [+0.068,+0.829], n_letters=18, dropped_below_floor=3)
D4 (sensitivity) P1 with 16k-fixed eligibility: mean=+0.0802,
    CI [+0.0416,+0.1234], n_letters=24, letters_losing_clean_at_262k=1
```

| cell (layer 13, l0_medium) | rate_family | rate_single | word L0 / cfg | in-dist FVU |
|---|---|---|---|---|
| 16k | 0.0963 | 0.1186 | 73.9 / 60 | 0.0419 |
| 65k | 0.1571 | 0.1798 | 71.7 / 60 | 0.0392 |
| 262k | 0.1920 | 0.2031 | 77.2 / 60 | 0.0367 |

## What transferred: P1, the inverted capacity direction

Family-endpoint absorption **rises monotonically with dictionary width**
(0.096 → 0.157 → 0.192), per-letter paired CI excluding zero, on a suite this
program did not train — different model family (Gemma 3 1B vs Pythia-1.4B),
different architecture (JumpReLU vs L1/TopK), different training pipeline and
data (DeepMind's). This is the direction round 13b found on Pythia — the
*opposite* of the toy theory's capacity-scarcity story — now with its first
external corroboration. The D4 sensitivity (eligibility fixed at the 16k cell,
τ-waived at 262k; exactly one letter churned) reproduces it, so the
clean-letter selection concern from the pre-lock review does not carry the
result.

## What did not transfer: P2, the splitting co-movement

On Pythia, absorption co-moved with split-family size across widths (13b). Here
the family size **does not move at all**: mean uncapped fam_size 1.6–2.5,
paired diff exactly 0.0000 with 4/23 letters changing in either direction, zero
FAM_CAP hits. Absorption grows 2× across a 16× width range while the
sel-defined letter family stays put. **The width→absorption effect on this
suite is not mediated by family growth as this endpoint measures it.** Candidate
readings, none registered: (a) the absorbing structure is the trial-varying
token-specific composites round 14 identified, which never clear the sel ≥ τ
family bar; (b) JumpReLU's trained per-latent thresholds gate firing patterns
differently from L1/TopK, so the sel-based family undercounts split structure
here; (c) the Pythia co-movement was pipeline-specific. Distinguishing these is
a successor round, not a footnote.

## P3 — inflation exists but missed the registered material bar

The single-latent endpoint inflates absorption at every width (cell-level
+23.2% / +14.4% / +5.8%, shrinking with width; 13a found 23–33% on Pythia at
one width), and the letter-mean point estimate is +0.351 — but the registered
bar was CI-lower > 0.10 and the CI lower bound is +0.068. **NOT CONFIRMED, as
registered.** The bar was set high deliberately after the pre-lock review
showed the sign alone is tautological; the honest reading is "inflation is
real but heterogeneous across letters, and materially smaller at large width."

## Descriptive series (hypothesis-generating only)

- **D1 (L0 axis, new territory):** absorption falls as the active budget rises
  — rate_family 0.194 (L0≈25) → 0.157 (≈72) → 0.094 (≈165) at fixed 65k width,
  while fam_size rises 1.56 → 1.91 → 2.50. So the two capacity axes point in
  opposite directions: more *total dictionary* → more absorption (P1); more
  *active budget per token* → less. 13b deliberately matched L0 and never
  varied this axis on Pythia. Prime candidate for the next prereg.
- **D2 (depth):** rate_family 0.111 (L7) / 0.096 (L13) / 0.041 (L17) / 0.061
  (L22); layer 22 nearly loses first-letter selectivity (7 clean letters) — the
  task is a mid-depth phenomenon, consistent with the single-layer caveat.
- **D3 (θ grid):** completely flat at every width — JumpReLU's own trained
  thresholds make the fire definition θ-insensitive, so round 12's θ-matching
  concern has no analogue on this architecture.

## Gates / infra

All gates passed. Amendment 1's in-distribution oracle: FVU 0.037–0.042 across
the width series (implementation exact); word-set FVU 0.40–0.46 and word L0
~20% above config are coherent domain shift on single BOS+word tokens, reported
descriptively. Encoder variant `raw` selected blindly in every cell (centered
gives L0 in the thousands), matching the GS2 technical paper. Provenance:
`PROVENANCE.json` (HF repo SHAs + sha256 of all 16 SAE files + 4 words caches),
`PROVENANCE.txt` (package versions).

## What this does and does not establish

- It does **not** touch L1-vs-TopK — JumpReLU is a third architecture; rounds
  12/13a/13b stand as they were.
- One seed per config (suite limitation); letters were the registered unit.
- Model weights via the ungated `unsloth/gemma-3-1b-pt` mirror; bit-identity
  with `google/gemma-3-1b-pt` asserted by the mirror, hashes recorded.
- What it **does** establish (scoped per `reviews/CODEX_REVIEW_2026-07-28.md`
  finding 1): **the positive within-suite width→absorption association
  reproduced in one independently trained JumpReLU suite.** The CI is over
  letters conditional on one model, one word sample, and one released SAE per
  width — it quantifies letter heterogeneity, not retraining or cross-suite
  uncertainty. A *transfer claim* ("not an artifact of any training recipe")
  requires multi-seed replication (Gemma Scope 2 ships `262k_l0_medium_seed_1`
  — a cheap partial) and/or replication on further suites (4b-pt, 270m-pt);
  queued in the plan. Within that scope, the assumed mediator (splitting as
  family growth) failing to move while absorption doubles remains the round's
  sharpest fact.
- The endpoint was measured under domain shift (isolated BOS+word tokens;
  word-set FVU 0.40–0.46 vs in-dist 0.037–0.042). The similar FVU across
  widths makes a trivial reconstruction-quality account of P1 unlikely, but a
  sequence-context replication matching the training distribution would
  materially strengthen it (queued).

## Cost / ops

CPU-only ephemeral `r15` (e2-standard-8), ~4.5 h wall clock ≈ $1.30, deleted
after collection. Lessons banked in the driver comments: Windows plink needs
`echo y |` on first contact and rejects `~` in pscp remote paths; per-layer
probe fitting dominates runtime (~30 min/layer).
