# Round 13b — capacity sweep: H1 is refuted, and the endpoint tracks fragmentation

**Frozen evaluator output: `results_round13b.txt`. Raw: `round13b_results.json`.
Prereg: `notes/prereg-round13b-capacity.md` (locked c934d33, Amendments 1+2 at
7501486, both pre-results). Evaluator `analysis/analyze_round13b.py` committed at
the lock. 48 SAEs = 3 widths × {L1, TopK} × 8 seeds, Pythia-1.4B L12, round-12's
cached activations, 15k steps.**

## Registered verdicts

| | verdict |
|---|---|
| Gates 1–4 (conformance, seeds/naming, provenance, matched-L0 ×3) | **all PASS** |
| Manipulation check | **PASS** — dead% 53.1 → 6.3 (46.8pp drop), live latents monotone [1919, 3256, 7682] |
| **P1 (primary)** | **FALSIFIED-DIRECTION** — absorption is *lower* under scarcity |
| **P2 (primary)** | **CONFIRMED** — the L1−TopK gap opens under scarcity |
| P5 (confound control) | does **not** fire |

λ per width: 3.784 / 4.1265 / 4.5 (m = 2048 / 4096 / 16384), all L0 in band.
Matched-L0 held at every width (|dL0| = 1.5, 0.9, 0.2).

*(As in round 12, the `lam: 5.0` shown in the TopK rows is an unused leftover
default — TopK uses k=32. Gate 1 only requires λ constancy within L1 cells.)*

## P1: the prediction is not just unconfirmed, it is backwards

Pooled paired diff `rate_family(m=2048) − rate_family(m=16384)` =
**−0.0445, 95% CI [−0.0493, −0.0397]** (n=16). Within arch: L1 −0.0410
[−0.0468, −0.0356], TopK −0.0480 [−0.0548, −0.0413].

The width profile is perfectly monotone in **both** arches —
`spearman(live latents, rate_family) = +1.00`:

| arch | m=2048 | m=4096 | m=16384 |
|---|---|---|---|
| L1 | 0.0133 | 0.0361 | 0.0542 |
| TopK | 0.0071 | 0.0145 | 0.0551 |

**H1 is refuted.** H1 held that round 12's null was regime-bound: that m=16384 at
46–57% dead was a spare-capacity regime in which the theory predicts nothing, and
that squeezing capacity would reveal the absorption the toy model calls for. The
opposite is true. m=16384 is the *high*-absorption regime by this metric, and
scarcity monotonically **reduces** measured absorption. Round 12's null cannot be
explained away by spare capacity.

### Two artifact explanations, both checked and excluded

1. **Denominator selection.** The scorer drops a letter when no latent reaches
   selectivity ≥ τ=0.30 (`if sel[j] < TAU: continue`), and those letters are
   exactly the ones with no clean representation. If small-m SAEs dropped more
   letters, rate_family would be biased down at small m and manufacture this
   result. Checked: **all 48 SAEs score the identical 24 letters** (only `x`, `z`
   drop) with an identical denominator **n = 17,981**. The comparison is exactly
   matched across widths and arches. Not an artifact.
2. **Retention confound (registered as P5).** The prereg warned that scarcity
   could convert absorption into outright *loss*, mechanically depressing P1, and
   that if so P1 "must NOT be read as 'capacity does not drive absorption'".
   It does not fire: loss **also falls** with capacity (0.0322 → 0.0216 pooled).
   Absorbed + lost — i.e. "no family latent fired at all" — falls from ~0.087 at
   m=16384 to ~0.032 at m=2048.

## What the endpoint is actually tracking

Since `absorbed + lost` is just "no sufficiently-selective latent fired", and that
quantity falls by ~2.7× as the dictionary shrinks 8×, the reading is:

**wider dictionaries fragment a letter's representation across more latents, and a
fragmented representation more often has *none* of its members fire on a given
token.** P3 shows the fragmentation directly, and shows it is an L1 phenomenon:
mean |F_L| grows 1.84 → 2.23 → 2.61 for L1 but is flat at 1.21 → 1.22 → 1.25 for
TopK. P4 replicates 13a: single-latent inflation is 23–33% across all widths.

This is the same construct-validity problem the thesis defense raised independently
(`reviews/DEFENSE_round13b_2026-07-25/`, Finding 1): the endpoint flags "no family
latent fired", and calls it absorption without ever verifying that the child's
missing mass is picked up by a parent. The capacity profile now makes that concern
concrete rather than hypothetical — the metric moves with fragmentation, which is a
property of dictionary width, not of hierarchical merging. **The proposed residual
projection check (does the parent latent's activation increase on absorbed trials?)
is now the highest-value next experiment.** It is post-hoc but cheap: all 48 weight
files plus round 12's are in hand.

## P2: confirmed as registered, but read it carefully

Interaction `[L1−TopK]@2048 − [L1−TopK]@16384` = **+0.0070, CI [+0.0014, +0.0135]**
(n=8). gap@m=2048 = +0.0062, gap@m=16384 = −0.0009. Per-seed:
[0.0075, 0.0, 0.0263, 0.0061, 0.0093, −0.0055, 0.0107, 0.0017] — 6/8 positive, one
exactly zero, one negative.

Reported as the registered verdict, unchanged. Three caveats stated in advance or
forced by the data:

- The **registered power caveat** applies: difference-of-differences on 8 seeds,
  CI half-width 0.0061, lower bound +0.0014 — barely clear of zero, and sensitive
  to seed 2 (+0.0263).
- It is **confirmed in the opposite regime from the one H1 posited.** H1 predicted
  the gap would appear where absorption is high; it appears where absorption is
  4–8× *lower*. The absolute effect is +0.0062 on a base of ~0.01.
- Because P1 inverted, "the gap opens under scarcity" no longer supports "round
  12's null was regime-bound" in the sense the prereg intended, even though that
  is the literal wording of the CONFIRMED branch. What survives is narrower: **at
  small m, L1 and TopK differ; at large m they do not.** Given P3, the most likely
  mechanism is splitting, not absorption — consistent with 13a's P5 finding that
  the arches differ in splitting rather than absorption.

*(Exploratory, not registered: on a relative scale the small-m gap is large — L1
absorbs 1.87× TopK at m=2048 versus 0.98× at m=16384.)*

## Standing

- Round 12's **NOT CONFIRMED** stands, and is now *strengthened*: 13a removed the
  splitting-artifact explanation, and 13b removes the spare-capacity explanation.
- **H1 is refuted** — the last pre-registered rescue of the architecture prediction.
- The capacity story from the toy model does **not** transfer to this endpoint at
  real scale in the predicted direction.
- The endpoint's construct validity is now the central open question, ahead of any
  further architecture comparison.

## Theory scorecard — two notes committed pre-unblinding (added post-unblinding)

Chronology, auditable from git + agent transcripts: results auto-committed by the ops
pipeline at 18:54:35 UTC (`25e3df0`); two theory notes were being derived at that time
by agents under an enforced blinding instruction (no file matching `*round13b*`/`*r13b*`);
transcript audit confirms neither agent touched any 13b artifact or ran git inspection
(only 13a files read). Notes committed 19:04:39 (`9adaea3`, splitting asymmetry) and
19:05:22 (`ac4b7ca`, matched-L0 agreement). First human/orchestrator read of any 13b
number: 19:06 UTC. So the predictions are blind-to-results, though the results predate
the commits on the clock — stated plainly.

**`theory/splitting_asymmetry.md` (blind predictions → outcome):**
- Families shrink as m falls with sign persisting (high conf) — **HIT** (L1 2.61→1.84,
  TopK flat, sign persists at all widths).
- Paired splitting gap shrinks monotonically (medium) — **HIT** (1.36→1.01→0.63).
- Quantitative ranges — TopK@2048 1.21 ∈ [1.0,1.25] **HIT**; gap@2048 +0.63 ≤ +0.7
  **HIT**; L1@2048 1.84 vs [1.2,1.8], L1@4096 2.23 vs [1.6,2.2] — high-edge misses by
  0.04/0.03.
- Mechanistic exposure "gap tracks dead%, not m" — consistent (dead% collapsed at
  m=2048 and the gap fell).
- Its mechanism (L1 self-gated nucleation splits; splitting consumes live latents) is
  also the natural reading of P2's positive sign, per §"read it carefully" above.

**`theory/matched_L0_invariance.md` (blind predictions → outcome):**
- Boxed P2 prediction (interaction ≈ 0) — **FALSIFIED** (CI [+0.0014,+0.0135] excludes
  0). The note's pre-stated escape channel ("L1 splitting consumes live capacity →
  positive sign; toy silent on sign if nonzero") matches the observed sign, but that is
  an escape, not a hit: matched-L0 single-pair agreement does not govern the scarce
  regime across pairs.
- Conditional P1 prediction (absorption rises if scarcity binds) — **FALSIFIED-DIRECTION**.
- What stands: the λ_c agreement theorem as toy mathematics (35/35 checks) and its
  account of the round-12/13a null in the spare-capacity regime; what fails is any
  extension of toy capacity-scarcity logic to this real endpoint (consistent with the
  construct-validity reading above).

## Cost / ops

One L4 (dev-gpu, us-west1-a), ~13h, ~$9. Box **TERMINATED** — the watchdog powered
it off after the collector set the RETRIEVED marker.

Collection did **not** use the driver's own uploads: this box was rebuilt from a
snapshot and runs as the default compute SA with no storage scopes, so every
`gcloud storage cp` in `ops/l4_r13b.sh` failed silently for ~9h (`set +e`). All 48
weights plus results were pulled with `ops/collect_r13b.sh` and pushed to
`gs://sae-identifiability-artifacts-ebd5a273/round13b/` from the orchestrator under
user ADC. `ops/preflight.sh` now catches this class before a round starts.
