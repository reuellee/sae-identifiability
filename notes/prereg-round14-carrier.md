# Pre-registration — Round 14: does "absorbed" mean a parent carries the child, or just that nothing fired?

**Status: LOCKED before any round-14 quantity is computed.** Lock = the commit
adding this file. The evaluator `analysis/analyze_round14.py` and the scorer
`analysis/round14_carrier.py` are committed at the same lock. No round-14 number
of any kind has been computed at lock time. Report what the frozen evaluator yields.

Two stages are declared in advance: a **calibration pilot** on a single held-out
SAE to fix scales (§Pilot), then the locked run. The pilot is declared here rather
than discovered later, exactly as round 8 did.

## The question

Every real-data absorption number in this project (rounds 12, 13a, 13b) comes from
one operational definition:

> a letter is **absorbed** on a token when it is *present*, still *retained*
> (linearly decodable from the SAE reconstruction), and **no latent in the
> selectivity family `F_L` fires**.

That definition never checks *where the letter's information went*. It infers
absorption from an absence. The thesis defense
(`reviews/DEFENSE_round13b_2026-07-25/`, Finding 1) attacked exactly this:
threshold suppression and capacity-driven feature loss produce the same absence,
so "the remaining ~75% is genuine absorption" is an assumption, not a measurement.

Round 13b then made the concern concrete rather than hypothetical: measured
absorption is **monotone in dictionary width** (Spearman +1.00, both arches;
pooled −0.0445 as m falls 8×), and co-moves with split-family size. A metric that
tracks fragmentation is behaving like a fragmentation metric.

**Round 14 asks whether the absorbed state has a carrier.** True hierarchical
absorption requires that on absorbed trials the child's mass is picked up by some
*other* latent — a parent or composite. Mere loss requires no such latent.

## Design

No new training. Re-analysis of existing frozen weights:

- **Primary cell:** the 16 round-13b SAEs at m=16384 ({L1, TopK} × 8 seeds) — the
  width at which rounds 12/13a measured their headline endpoints.
- **Capacity contrast:** the 16 round-13b SAEs at m=2048.
- Same activations (`acts_eval.pt`), same probes and `words_pythia-1.4b_L12.pt`,
  same θ=0, τ_fam=0.30, FAM_CAP=32 as `analysis/round13b_scorer.py`, so the
  absorbed set is *identical* to the one round 13b scored.

### Carrier decomposition

For letter `L` with probe direction `u_L` (the round-13 logistic probe's weight
vector, unit-normalised), and reconstruction `x̂ = Σ_i f_i d_i + b_dec`, define the
per-latent contribution to the letter direction on a token:

    c_i = f_i · (d_i · u_L)

The **carrier** on a trial is `argmax_i c_i` restricted to `i ∉ F_L` (a family
latent cannot be the carrier — by construction none of them fired on an absorbed
trial, but the restriction is stated so the definition is total).

Three trial sets, all within letter `L` and all requiring the letter *present*:

- **A (absorbed):** retained, no `F_L` latent fires — the round-13b absorbed set.
- **C (control-fired):** an `F_L` latent fires — normal representation.
- **N (lost):** not retained, no `F_L` latent fires — round-13b's `lost` set.

## Pre-registered predictions

**P1 (PRIMARY) — compensation.** *This is the examiner's decisive test.*
For each (SAE, letter), let `κ` be the modal carrier on set A. Compare `κ`'s mean
activation on A versus on C, paired by (SAE, letter), 10k bootstrap over SAEs.
- **CONFIRMED (absorption)** if CI lower > 0: the carrier is *more* active exactly
  when the family is silent, i.e. it takes up the slack.
- **FALSIFIED-DIRECTION (loss)** if CI upper < 0.
- **NOT CONFIRMED** if it straddles 0.

**P2 (PRIMARY) — carrier consistency.** Absorption into a parent implies a
*repeated* carrier; diffuse loss implies a carrier that varies trial to trial.
Endpoint: top-1 carrier share `s = max_i P(carrier = i | A)` per (SAE, letter),
compared against a **random-direction null**: recompute the carrier on the *same*
trials using a random unit direction `u_rand` in place of `u_L`, 32 draws per
(SAE, letter), and take the null's mean top-1 share. 10k bootstrap over SAEs on
the paired difference `s(u_L) − s(u_rand)`.

*(A label-permutation null was considered and rejected as degenerate: permuting
carrier labels across A-trials leaves the multiset unchanged, so `max_i P(·)` is
invariant and the null would be uninformative by construction. The random-direction
null instead holds the firing structure fixed and asks whether concentration is
specific to the letter direction.)*
- **CONFIRMED** if mean `s` exceeds the null CI upper.
- **NOT CONFIRMED** otherwise.
A high `s` is necessary for the absorption reading; a null-level `s` would mean
"absorbed" trials have no common carrier at all.

**P3 (secondary) — carrier breadth.** A parent/composite should be *broader* than
the child family. Report firing rate of `κ` versus mean firing rate of `F_L`.
Absorption predicts `rate(κ) > rate(F_L)`. Reported with CI; no pass/fail bar.

**P4 (secondary) — concentration.** Share of the total positive letter-direction
projection carried by `κ` on A, versus by the top `F_L` latent on C. If A-trials
are diffuse (low share) while C-trials are concentrated, that favours loss.

**P5 (secondary) — capacity contrast.** All of the above at m=2048 vs m=16384. If
the carrier signature is present at one width and absent at the other, the
"absorption" label is width-dependent and rounds 12/13a/13b are measuring
different things at different widths.

## Gates

1. **Reproduction gate (short-circuit).** The round-14 harness must reproduce
   round 13b's `rate_family` per SAE to within 0.002 absolute. If not, nothing
   below is interpretable and the round stops. *(This is round 13a's Gate 4, which
   caught nothing but cost nothing and makes the comparison auditable.)*
2. **Set sizes.** Report |A|, |C|, |N| per cell. Any (SAE, letter) with |A| < 20 is
   excluded from P1/P2 and the exclusion count is reported. Registered in advance
   because absorbed trials are rare at m=2048 (rate 0.0071–0.0133).
3. **Probe provenance.** `u_L` must come from the same probe object used by the
   round-13 scorer, not a refit.
4. **Null.** P2's permutation null is computed within letter and within SAE.

## Pilot (declared in advance)

One SAE — `sae_pythia-1.4b_L12_l1_x8_s0.pt` — is used to check that the
decomposition runs, that |A| is large enough to estimate a modal carrier, and to
fix nothing else. **No threshold above is set from the pilot**; the only
pilot-dependent quantity is the |A| ≥ 20 floor in Gate 2, which is stated here and
will not be changed after seeing results. If the pilot shows the floor is badly
chosen, the round is re-registered rather than amended silently.

## What this cannot do

- It cannot prove absorption exists; it can show the absorbed set does or does not
  have the carrier signature absorption requires.
- A P1 confirmation does **not** rescue the architecture prediction — rounds 12,
  13a and 13b are unaffected either way. It licenses the *word* "absorption".
- A P1 falsification would mean the first-letter absorption metric (SAEBench's
  included, and every number in §8b) is substantially measuring representational
  loss, and the paper's real-data sections must be reworded accordingly.
- Single model, single layer, single task, as everywhere else in this program.

## Ops

CPU-only. The orchestrator has no pip/torch, so this runs on an `ephemeral`
e2-standard-8; weights come from `~/r13b_pull/`, activations from
`gs://…/round12/acts_eval.pt` via `ops/gcs_adc.sh` then `gcloud compute scp`
(ephemeral VMs are created `--no-service-account --no-scopes` and cannot reach
GCS; do **not** put a user ADC token on the VM). Run `ops/preflight.sh` first.
