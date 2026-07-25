# Round 12 — real-model causal L1-vs-TopK absorption test (Pythia-1.4B L12)

Registered, locked at commit `0722212`. Frozen scorer `analysis/analyze_round12.py`.
16 SAEs = 8 seeds × {TopK k=32, L1 λ=5.0}, m=16384, d=2048, matched at L0≈32.
Ran on L4 `dev-gpu-2` (us-east1-b), ~8h, box deleted on completion.

## Registered verdict (verbatim, from the frozen scorer)

```
gates: {'conformance': True, 'seeds': True, 'matched_L0': True, 'matched_letters': False}
P1 (PRIMARY): NOT CONFIRMED (dir=straddles-0; failing gates: ['matched_letters'])
P2 (descriptive): L1: conc mean=+0.0765 CI [+0.0703,+0.0828] -> CONCENTRATED(+)
                | TopK: conc mean=+0.0934 CI [+0.0841,+0.1036] -> CONCENTRATED(+)
P3 (secondary): topk: recall=0.333 vs baseline=0.030 (enrichment 11.23x);
                l1:   recall=0.812 vs baseline=0.153 (enrichment 5.32x)
```

**P1 NOT CONFIRMED.** The toy geometry does not predict a reproducible real-scale
L1-vs-TopK gap in first-letter absorption at this width.

| | L1 (λ=5.0) | TopK (k=32) |
|---|---|---|
| absorption rate (θ=0) | 0.0744 | 0.0715 |
| loss rate | 0.0424 | 0.0314 |
| mean L0 | 31.8 | 32.0 |
| FVU | 0.079 | 0.063 |
| dead latents | 46–57% (both arches, shared recipe) | |

Paired per-seed diff **+0.0030**, 95% CI **[−0.0010, +0.0067]**, n=8, not
sign-consistent (5/8 positive). The `matched_letters` gate reads False, but that
is tautological under a null — it only passes given a significant signal to
reproduce. With the clean set the intersection is all 24 letters and the
matched-letter diff equals the full diff exactly.

## Data-integrity incident and clean re-score (2026-07-25)

The originally collected `results_round12.txt` was **contaminated**. A stale
pythia-70m L3 smoke-test file (`sae_pythia-70m_L3_topk_x8_s0_fl.json`) survived
the resume script's cleanup, which `rm`'d only the 1.4b glob. It carried
**seed 0**, so in the scorer's seed-keyed pairing dict it *overwrote* the real
1.4B TopK seed-0 record (0.0826) with the 70m value (**0.2105**).

Effect: it poisoned the seed-0 pair (spurious diff −0.122) and every TopK mean.
The conformance and seed gates **correctly caught it** and P1 was reported
NOT CONFIRMED either way — but the reported effect size was wrong.

| | contaminated | clean |
|---|---|---|
| TopK absorption | 0.0874 | **0.0715** |
| paired diff | −0.0130 | **+0.0030** |
| 95% CI | [−0.0454, +0.0057] | **[−0.0010, +0.0067]** |
| hard gates | conformance ✗, seeds ✗ | all pass ✓ |

The clean re-score (`results_round12_clean.txt`) uses the **unmodified frozen
scorer** over the 16 in-config SAEs only. The contaminant is a documented
**post-hoc file exclusion**: it is out-of-config on the registered dimensions
(model, layer) and its removal is decided by the registered conformance rule,
not by its effect on the outcome. P2 TopK concentration de-inflates
0.0946→0.0934; P3 is bit-identical (it only ever read 1.4B pairs files).

## Post-hoc diagnosis (EXPLORATORY — `round12_posthoc_diagnosis.txt`)

Not registered; hypothesis-generating for round 13.

1. **The null is informative.** At the registered seed-level unit the CI excludes
   an L1−TopK gap larger than ±25% of the 0.055 base rate. This is not an
   underpowered null in sample size.
2. **But the endpoint is fragile.** Absorption is wildly heterogeneous across
   letters (L1 0.004→0.197; TopK 0.000→0.231) and the **top 3 of 24 letters carry
   53% (L1) / 73% (TopK) of all absorbed instances**. The endpoint is effectively
   driven by ~3 letters.
3. **The endpoint is ~2/3 a selectivity re-expression.** By construction
   `sel = P(fire|L) − P(fire|¬L)` and `rate ≈ P(main latent misses | L present)`,
   so `rate ≈ (1−FPR) − sel` algebraically. Empirically
   `rate = −0.587·sel + 0.555`, **R² = 0.673** over 384 SAE-letter cells.
   Feature **splitting** moves exactly this quantity. (A regression of rate on
   arch *controlling for* sel conditions on a function of the outcome and is
   invalid; it was computed, found to flip sign, and is **discarded**, not
   reported as a finding.)
4. **There is a clean architecture difference — in splitting, not absorption.**
   Mean main-latent selectivity L1 **0.828** vs TopK **0.872**; paired
   letter-clustered diff **−0.0444, CI [−0.0679, −0.0218]**, lower for L1 in
   19/24 letters. L1 splits the letter feature more. Consistent with the
   round-11 L1-splitting finding and with P3's L1 recall ≫ TopK.
5. Frequency (q) dependence is weak and confounded: spearman(n, rate) = +0.37
   (L1, p=0.073) / +0.16 (TopK, p=0.47).

**Read.** Two mechanisms would each produce a round-12 null *without the toy
theory being wrong*: (H1) m=16384 with 46–57% dead latents is a **spare-capacity**
regime, and the theory places absorption pressure under capacity *scarcity* —
the same trap that made the earlier GPT-2 POC at m=1536 a null; (H2) the
registered endpoint cannot separate "L1 absorbs more" from "L1 splits more",
and L1 demonstrably splits more. Round 13 tests H2 first because it is
answerable by re-scoring the existing weights.

## Status

- P1 stands as registered: **NOT CONFIRMED**. Not amended by anything above.
- Artifacts: all 56 round-12 objects in `gs://sae-identifiability-artifacts-ebd5a273/round12/`
  (16 weights, acts_train/acts_eval, words, JSONs). Analysis artifacts local in
  `results/real/`.
- Ops bug fixed: collection ran as the orchestrator SA, which has zero access to
  the bucket. See `ops/gcs_adc.sh` — use user ADC, not the SA.
