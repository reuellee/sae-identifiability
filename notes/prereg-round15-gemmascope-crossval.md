# Pre-registration — Round 15: do the width/splitting patterns transfer to an external SAE suite? (Gemma Scope 2 cross-validation)

**Status: LOCKED before any round-15 quantity is computed.** Lock = the commit
adding this file. The experiment harness `experiments/gemmascope_crossval.py` and
the evaluator `analysis/analyze_round15.py` are committed at the same lock. No
round-15 number of any kind has been computed at lock time. Report what the frozen
evaluator yields.

## The question

Every real-data result in this program so far comes from **one training pipeline**:
our own SAEs, on one model (Pythia-1.4B), trained by our recipe. Three findings are
load-bearing and could all be artifacts of that pipeline:

1. **Width direction (13b P1, FALSIFIED-DIRECTION):** family-endpoint absorption
   *falls* as dictionary width shrinks (−0.0445 pooled as m drops 8×; Spearman
   +1.00 with m in both arches). The toy theory predicted the opposite.
2. **Splitting grows with width** (13b co-movement; 13a P5 family sizes).
3. **Single-latent inflation (13a P1):** the single-latent endpoint inflates
   absorption ~25% relative to the family endpoint.

Round 15 asks whether these patterns **transfer to SAEs we did not train**:
DeepMind's Gemma Scope 2 suite — a different model family (Gemma 3 1B vs
Pythia-1.4B), a different architecture (JumpReLU vs our L1/TopK), a different
training pipeline and data distribution, professionally trained with published
per-SAE configs. Transfer would be the first evidence these are properties of
sparse dictionary learning at real scale rather than of our recipe. Non-transfer
bounds the claims' scope accordingly.

This round tests the **patterns**, not the L1-vs-TopK architecture contrast —
JumpReLU is neither arch. The arch-null results (12 P1, 13a P4) are not at stake.

## Design

No training. Public weights only:

- **SAE suite:** `google/gemma-scope-2-1b-pt`, site `resid_post` (CC-BY-4.0,
  ungated). Every SAE ships a `config.json` declaring `hf_hook_point_in`
  (`model.layers.<ℓ>.output`), `width`, and trained `l0`.
- **Model:** Gemma 3 1B PT. `google/gemma-3-1b-pt` is license-gated on HF; weights
  are taken from the ungated mirror **`unsloth/gemma-3-1b-pt`** (recorded
  provenance caveat; config and tokenizer hashes logged in the output).
- **Cells (8 SAEs, ~4.8 GB):**
  - **Width series (primary):** layer_13 × l0_medium × width ∈ {16k, 65k, 262k}.
    Layer 13/26 = 50% depth, mirroring Pythia L12/24. 1m is excluded (memory
    scope, stated in advance).
  - **L0 series (descriptive):** layer_13 × width_65k × l0 ∈ {small, medium, big}.
  - **Layer series (descriptive):** width_16k × l0_medium × layer ∈ {7, 13, 17, 22}.
- **Task/endpoints:** the frozen first-letter machinery, ported verbatim in
  semantics from `analysis/round13a_family_endpoint.py` /
  `experiments/real_firstletter.py`: whole-word single tokens (`^ [a-z]{3,}$`
  after decode), BOS+token forward pass, residual at the config's hook point
  (`hidden_states[ℓ+1]`), out-of-fold presence probes, full-fit retention probes
  on the SAE reconstruction, `sel_i = P(fire_i|L) − P(fire_i|¬L)`,
  SINGLE (`argmax`, sel ≥ τ) and FAMILY (`{i: sel_i ≥ τ}`, cap 32) absorbed
  rates. Frozen params: **θ=0, τ=0.30, FAM_CAP=32, MIN_WORDS=30, PROBE_C=1.0,
  BOOT=10 000.**
- **Encoder-input convention (blind infra rule):** Gemma Scope 1's published
  encoder takes raw inputs (`pre = x·W_enc + b_enc`); Anthropic-style SAEs
  center by `b_dec` first; Scope 2's convention is not stated in its configs.
  The harness computes one batch under both variants and selects the one whose
  L0 is closest to the config's trained `l0` — a sparsity-conformance decision,
  blind to every endpoint, recorded per SAE (`enc_variant`) and backstopped by
  Gate 1. FVU is always computed in raw space against `f·W_dec + b_dec`.
- **Fire definition on JumpReLU:** `f = relu(pre) · 1[pre > threshold]` (the
  suite's own trained gate), so `fire ⇔ f > 0`. θ=0 therefore counts a latent as
  firing exactly when the SAE's own gate passes it — the same L0 definition the
  suite's `config.json` reports. No magnitude threshold is imposed on top
  (round 12's θ-matching rationale; the θ-grid {0, 0.01, 0.05, 0.1} is reported
  descriptively as D3).
- **Statistical unit: letters.** Gemma Scope 2 provides one seed per config, so
  seed-level CIs are impossible. All paired contrasts are per-letter, restricted
  to the intersection of letters with `clean_latent=True` (sel ≥ τ) in every cell
  being compared, with 10k-rep bootstrap over letters. This is the letter-clustered
  analysis 13a used for its splitting contrast, promoted to the primary unit.

## Pre-registered predictions

**P1 (PRIMARY) — width direction transfers.** Per-letter paired difference
`rate_family(262k) − rate_family(16k)` at l0_medium, layer 13; 10k bootstrap over
the letter intersection.
- **CONFIRMED (transfer)** if CI lower > 0 — absorption rises with width, as 13b
  found on Pythia.
- **FALSIFIED-DIRECTION** if CI upper < 0.
- **NOT CONFIRMED** if the CI straddles 0.
Also report the per-letter Spearman of `rate_family` against width over
{16k, 65k, 262k} (descriptive monotonicity check; no bar).

**P2 (KEY SECONDARY) — splitting grows with width.** Per-letter paired
difference `fam_size_uncapped(262k) − fam_size_uncapped(16k)` — the UNCAPPED
count of latents with sel ≥ τ (the endpoint's scoring family stays capped at 32
for 13a fidelity, but the splitting *measure* must not be censored at the cap);
same cells, same unit, 10k bootstrap.
- **CONFIRMED** if CI lower > 0; **FALSIFIED-DIRECTION** if CI upper < 0;
  else **NOT CONFIRMED**.
*(Multiplicity: P1 is the SOLE primary claim of this round; P2 and P3 are
secondary and carry no family-wise success claim. "Round 15 confirms transfer"
may be asserted only from P1.)*

**P3 (secondary) — single-latent inflation is MATERIAL.** The sign of
`rate_single − rate_family` is guaranteed by construction (the family contains
the argmax latent), so the registered bar is magnitude, not sign: letter-mean
relative inflation `mean_w(rate_single)/mean_w(rate_family) − 1` over the
three-width clean-letter intersection, excluding letters with mean
`rate_family < 0.005` (registered floor); 10k letter bootstrap; requires ≥ 8
surviving letters.
- **CONFIRMED** if CI lower > **0.10** (at least 10% relative inflation,
  materially transferring 13a's 23–33%); otherwise **NOT CONFIRMED**.

**D1 (descriptive) — L0 series.** `rate_family` and `fam_size` across
l0 ∈ {small, medium, big} at 65k. No prior finding fixes this direction; no
registered bar. Reported to seed a future prereg.

**D2 (descriptive) — layer series.** Same endpoints across layers {7, 13, 17, 22}
at 16k/medium.

**D3 (descriptive) — θ grid.** `rate_family` at θ ∈ {0, 0.01, 0.05, 0.1} per cell
at widths ≤ 65k (the 262k cell is omitted from the grid for memory — magnitudes
are not retained at that width; stated in advance).

**D4 (registered sensitivity for P1's eligibility rule).** The clean-letter
intersection conditions on an SAE outcome (a letter can *lose* its clean latent
at 262k because splitting dilutes per-latent selectivity — plausibly the very
effect under study). D4 re-computes the P1 contrast with eligibility fixed by
the **16k baseline cell alone**: letters clean at 16k contribute
`rate_family(262k)` when clean there, else the **τ-waived** rate (family =
argmax singleton). Reported with CI and the count of letters losing clean
status at 262k. Divergence between P1 and D4 is itself informative about
selection. D4 is descriptive; P1's registered rule stands.

## Gates

0. **Frozen-configuration gate.** The evaluator fails closed unless every row
   carries θ=0, τ=0.30, FAM_CAP=32 and all 8 registered cells are present.
   (The harness's env overrides exist for the SMOKE pilot only.)
1. **Conformance/hook gate (short-circuit).** Per SAE: measured mean L0 over the
   word set within **[0.5×, 1.5×]** of its `config.json` `l0`, and word-set
   FVU ≤ 0.5. A wrong hidden-states index or mis-oriented weight load fails this
   loudly (FVU ≈ 1). Any width-series failure → the affected predictions are
   reported NOT CONFIRMED (infrastructure) and the round stops for diagnosis; no
   silent re-run with different plumbing. *Caveat (registered):* the config `l0`
   was measured on pretraining-distribution sequences, this gate on isolated
   BOS+word tokens — a domain-shifted L0 can legitimately drift. The pilot adds
   an official-loader **oracle check** (sae-lens encode vs ours, tolerance 1e-3
   on a fixed batch); if the oracle PASSES but the L0 band fails, that is domain
   shift, and the response is re-registration of the band, never silent
   widening.
2. **Letters gate.** ≥ 15 letters with `clean_latent=True` in every width-series
   cell, and a compared-cell intersection ≥ 12 for P1/P2. Below that: the
   affected prediction is NOT CONFIRMED (underpowered), reported as such.
3. **Words gate.** ≥ 30 words/letter for ≥ 20 letters. If the tokenizer yields
   more than **24 000** word tokens, subsample to 24 000 stratified by letter
   (numpy seed 0) — registered memory bound, decided before any activation is
   computed.
4. **Probe parity.** Presence/retention probes are fit once on the raw residuals,
   SAE-independent, and reused across all SAEs — the 13a code path.

## Pilot (declared in advance)

A SMOKE pass on the single cell (16k, l0_medium, layer 13) checks: weights load,
key orientation, the conformance gate computes, letters count. **No threshold or
gate value is set from the pilot** — every number above is fixed here. If the
pilot reveals a defect (loader crash, hook mismatch), the fix is committed as a
pre-results amendment exactly as round 14's Amendment 1; if a *gate value* proves
badly chosen, the round is re-registered, not silently amended.

## What this cannot do

- It cannot adjudicate L1-vs-TopK anything — JumpReLU is a third architecture.
  Transfer failure does not touch rounds 12/13a/13b; it bounds their scope.
- One seed per config: a transfer failure could in principle be seed noise in the
  suite; letters are the replication unit, and this limitation is stated rather
  than solved.
- The letter-bootstrap CIs quantify letter-level heterogeneity **conditional on
  the word sample and fitted probes** — word-sampling and probe-fitting
  uncertainty are not propagated (a nested word-within-letter bootstrap with
  probe refits is out of scope and would be a different registered design).
- Family selection and scoring use the same words — inherited from the frozen
  13a endpoint **by design** (port fidelity is the point of a transfer test).
  A cross-fitted family-selection endpoint is a legitimate successor design,
  not a silent change to this one.
- Each layer-series SAE is scored against its own layer's activations and
  probes (all four hook layers are extracted in a single forward pass).
- Single model (Gemma 3 1B), single site (resid_post), single task (first-letter),
  as everywhere else in this program.
- Model weights come from a community mirror of a gated repo; config/tokenizer
  hashes are recorded, but bit-identity with `google/gemma-3-1b-pt` is asserted by
  the mirror, not verified here.

## Ops

CPU-only, laptop-driven (`ops/run_round15.sh`): ephemeral `e2-standard-8`
(us-central1-a, `--no-service-account --no-scopes`, `ops/vm_watchdog.sh` armed).
The VM pip-installs torch-cpu/transformers/safetensors/scikit-learn, downloads the
8 SAE param files + the model mirror from HF (public, no token), builds the words
file, scores all SAEs streaming (fires kept as bool; ~6.3 GB peak at 262k), writes
per-SAE JSON rows + `results_round15.txt` via the frozen evaluator. Results are
scp'd back, committed, uploaded to `gs://sae-identifiability-artifacts-ebd5a273/round15/`
from the laptop (user ADC — VMs have no GCS access), and the VM is **deleted**.
Estimated cost < $1.
