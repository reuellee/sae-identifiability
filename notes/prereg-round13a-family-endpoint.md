# Pre-registration — Round 13a: is the first-letter absorption endpoint measuring absorption or feature splitting?

**Status: LOCKED before any 13a result is computed.** Lock = the commit that adds
this file. Analysis code `analysis/round13a_family_endpoint.py` is frozen at the
same commit. Report whatever it yields.

## Motivation

Round 12 returned a registered **NOT CONFIRMED** on P1 (no L1-vs-TopK real-scale
absorption gap; clean paired diff +0.0030, CI [−0.0010,+0.0067]). Post-hoc
diagnosis (`results/real/round12_posthoc_diagnosis.txt`, exploratory) found:

- the endpoint is **R² = 0.673** explained by the main latent's selectivity, which
  is near-algebraically tied to it (`rate ≈ (1−FPR) − sel`);
- **L1 splits more than TopK** (mean sel 0.828 vs 0.872, paired CI
  [−0.0679,−0.0218]).

A metric that fires when *the single designated latent* misses cannot distinguish
"the child was absorbed into a parent" from "the letter feature split across
several latents and the designated one wasn't the one that fired". These have
opposite implications and the round-12 architectures differ on the splitting axis.

## Design

**No new training.** Re-score the 16 **existing, frozen** round-12 SAEs
(`gs://sae-identifiability-artifacts-ebd5a273/round12/`, 8 seeds ×
{L1 λ=5.0, TopK k=32}, m=16384) on the **same held-out** `acts_eval.pt` and
`words_pythia-1.4b_L12.pt`, changing only the endpoint. CPU-only on a throwaway
ephemeral VM (no GPU).

Because this reuses round-12 weights and data, **every 13a result is
development-set / metric-validity evidence. It cannot confirm any architecture
claim.** A confirmatory architecture test requires fresh SAEs (round 13b).

### Endpoint definitions

Single-latent (round-12 registered, recomputed here as the comparison baseline):

- `sel_i = P(fire_i | L) − P(fire_i | ¬L)`; main latent `j = argmax_i sel_i`;
  letter scored iff `sel_j ≥ 0.30`.
- `absorbed = present ∧ retained ∧ ¬fire_j`; `rate = absorbed / present`.

**Family endpoint (new):**

- Family `F_L = { i : sel_i ≥ τ_fam }`, **τ_fam = 0.30** — reused from the
  existing `SEL_MIN`, no new tuning.
- Registered cap: if `|F_L| > 32`, keep the top 32 by `sel` (guards against a
  degenerate family swallowing the dictionary). Record `|F_L|` always.
- Letter scored iff `|F_L| ≥ 1` (identical scoring set to the single-latent rule,
  since `|F_L| ≥ 1 ⟺ sel_j ≥ 0.30`).
- `absorbed_fam = present ∧ retained ∧ (no i ∈ F_L fires)`;
  `rate_fam = absorbed_fam / present`.

All of `θ = 0`, `present`, `retained`, the probe fitting, and the eval split are
**unchanged** from `experiments/real_firstletter.py`.

## Pre-registered predictions and falsification criteria

**P1 (primary — substance). Does absorption survive the family correction?**
Pooled `rate_fam` across all 16 SAEs, seed-level bootstrap CI (10k).

- **SURVIVES** if `rate_fam ≥ 0.01` with CI excluding 0.01 from below.
  → real absorption exists beyond splitting; the round-12 arch null stands as a
  statement about genuine absorption.
- **DISSOLVES** if the CI upper bound < 0.01.
  → round 12's 0.055 was ~entirely feature splitting, and the SAEBench-style
  single-latent endpoint is **not a valid absorption measure at this width**.
  This would be a substantive negative methodological result about a metric in
  current use, and it would retro-explain the null.
- Anything between → reported as INDETERMINATE, no spin.

**P2 (primary — metric validity).** Regress `rate_fam` on `max_{i∈F_L} sel_i`
over all SAE-letter cells. Registered comparator: the single-latent endpoint's
**R² = 0.673**.

- PASS if `R²(family) < 0.40`. FAIL if `≥ 0.40`.
- A pass means the family endpoint is substantially less a re-expression of
  "did the top latent fire".

**P3 (secondary — heterogeneity).** Top-3-letter share of absorbed instances
under the family endpoint. Registered comparator: 53% (L1) / 73% (TopK).
Reported; no pass/fail bar.

**P4 (secondary, NOT confirmatory — architecture).** Seed-paired L1−TopK diff on
`rate_fam`, n=8, 10k bootstrap CI. Reported with the explicit caveat that it
reuses round-12 weights and therefore cannot confirm an architecture claim.
Its role is to size round 13b.

**P5 (descriptive).** Distribution of `|F_L|` per letter per arch, and the paired
L1−TopK diff in mean `|F_L|`. Direct measurement of splitting.
Registered expectation from the diagnosis: `|F_L|` larger for L1.

## Gates (any failure ⇒ P1/P2 not confirmable, reported as such)

1. **Conformance:** every scored SAE has `model=EleutherAI/pythia-1.4b`,
   `layer=12`, `m=16384`, `θ=0`, `τ_fam=0.30`; L1 arm all `λ=5.0`; TopK arm all
   `k=32`.
2. **Seeds:** exactly seeds {0..7} in each arm, 16 SAEs, no duplicates.
   *(Round-12's contamination was a duplicate seed — this gate is load-bearing.)*
3. **Provenance:** each weight file's SHA256 recorded in the output. No file may
   be scored that is not one of the 16 round-12 objects.
4. **Baseline reproduction:** the recomputed single-latent `rate` must match the
   frozen round-12 `fl.json` values to within 0.002 per SAE. If it does not, the
   re-score harness is not faithful and **nothing else is reported**.

Gate 4 is the important one: it proves the new harness reproduces the frozen
result before its new endpoint is believed.

## What this does NOT do

- It does not amend the round-12 P1 verdict.
- It does not test H1 (the capacity-regime hypothesis). That is round 13b: a
  width sweep `m ∈ {2048, 4096, 8192, 16384}` at fixed L0, both arches, fresh
  SAEs, with per-SAE dead% recorded (live latents, not `m`, being the real
  capacity measure). 13b is only worth its GPU cost if 13a's P1 SURVIVES.

## Ops

- CPU-only ephemeral VM; pull inputs from GCS via **user ADC** (`ops/gcs_adc.sh`)
  — the orchestrator service account has no access to the bucket, which is the
  round-12 collection bug.
- Weights and all outputs saved; results to `results/real/round13a_*`.
