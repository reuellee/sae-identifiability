# Pre-registration: real-model causal first-letter absorption, L1 vs TopK (round 12)

*Drafted 2026-07-24. The whole-repo review's single highest-value next
experiment, combining detector-recalibration + first-letter validation +
the confirmatory L1-vs-TopK comparison into one pre-registered run with a
**causal, ground-truth** primary endpoint. Status: DRAFT — under dual pre-lock
external review (Gemini + GPT-5.6); locks after review + the D-phase (SMOKE)
are committed. After lock, changes only by dated amendment.*

## Claim under test (primary)

On real Pythia-1.4B layer-12 activations, at **matched sparsity (L0)**, **L1
SAEs exhibit a higher first-letter feature-absorption rate than TopK SAEs**
(H1). Secondary (P2): the absorbed first letter is causally carried by the
word's dominant (magnitude-selected) latents, tested **non-circularly**.
Secondary (P3): the project's label-free pair detector, scored **blind**,
recalls the ground-truth main-L-latents better than its opportunity baseline.

Scope: this is a **first-letter absorption metric in the spirit of Chanin et
al. (arXiv:2409.14507)** — main-first-letter-latent fails to fire despite the
first-letter being linearly present — **not** the exact SAEBench pipeline
(no k-sparse-probing variant, no cross-model calibration). Registered as such.

## Ground-truth data (frozen)

- **Word set W:** Pythia vocab tokens whose decode matches `^ [a-z]{3,}$`
  (leading space + ≥3 lowercase letters — whole-word tokens, no fragments).
  First letter ℓ(t) = the first alphabetic char. (Expect a few thousand.)
- **Word activations:** for each word-token, `BOS + token` through Pythia-1.4B,
  take the layer-12 residual at the word-token position (batched). Same model
  revision as extraction.
- **Train/eval separation:** the SAE **training** activations come from a
  disjoint corpus (Gutenberg set A); the first-letter word set and its
  activations are **held out** from SAE training and shared identically by
  every SAE.

## SAEs (frozen)

- Pythia-1.4B layer 12, m = 16384 (x8), unit-norm decoder, ReLU, pre-decoder
  bias, dead-latent resampling; **acts held on GPU** for speed.
- **Matched-L0 comparison point (primary):** TopK **k = 32**; L1 **λ tuned so
  its eval L0 ∈ [28, 36]** (a pre-run λ-calibration on ONE seed picks λ; the
  chosen λ is recorded in the lock and frozen for all seeds). If no λ lands L1
  in [28,36], widen to [24,40] and disclose. **L0 is counted as mean #{act > 0}**
  — the same threshold as the metric's primary θ = 0, so "matched sparsity" and
  "fire" are the same event for both architectures (this is what makes P1
  θ-unbiased at the primary point). λ-calibration runs on the L4 as step 1 (the
  tiny model can't calibrate 1.4B's λ); the chosen λ is appended to this prereg
  by amendment before the 10-SAE run.
- **Seeds:** 5 per architecture (0–4), identical corpus/token-order/init-scheme/
  width/step-budget (15k steps). 10 SAEs primary.
- Weights → GCS; per-SAE stats (FVU on a **held-out** eval slab, L0, dead%).

## First-letter absorption metric (FROZEN definition)

Per SAE, per letter L ∈ {a…z} with ≥ 30 words:

1. **Residual probe (ground truth for "is the first letter present"):** a
   logistic-regression probe on the **residual** activations predicting
   ℓ(t)==L (balanced positives/negatives, 5-fold; use out-of-fold predictions).
   A word w is **letter-present** iff its out-of-fold probe prob > 0.5.
2. **The main L-latent:** for each SAE latent j, selectivity
   s_j = P(fire_j | L-word) − P(fire_j | non-L-word). **fire = act > θ, with
   the PRIMARY θ = 0** (i.e. the latent activates at all). θ = 0 is chosen
   deliberately: it is the *same* threshold the two architectures are matched
   on (L0 = mean #{act > 0}). A positive θ would systematically zero more of
   L1's soft-thresholded (smaller) activations than TopK's, biasing absorption
   toward L1 — the round-11 θ-asymmetry failure class. We therefore threshold
   at 0 for the primary and **report the θ-sensitivity grid** θ ∈ {0, 0.01,
   0.05, 0.1} (see P1 robustness). L-latent = argmax_j s_j; require s_j ≥ 0.30
   else "no clean L-latent" (that letter excluded from the SAE's score, disclosed).
3. **Absorbed instance:** an L-word w is **absorbed** iff it is letter-present
   (probe > 0.5) AND the L-latent does **not** fire on w (act ≤ θ). (The model
   represents the first letter, but the dedicated latent misses it → absorbed
   elsewhere.)
4. **Absorption rate (per SAE):** over letters with a clean L-latent,
   `Σ 1[letter-present ∧ L-latent misses] / Σ 1[letter-present]` pooled over
   all L-words of all scored letters — the fraction of first-letter-present
   words the main L-latent fails on. Computed at the primary θ and at every
   grid θ.

## Causal component (FROZEN) — non-circular by construction

The naïve test "ablate the latents most aligned with the L-probe direction,
then measure the L-probe logit drop" is **circular** (selecting latents by
readout alignment and then measuring readout alignment is an identity — it
cannot fail). We therefore select and measure along **independent** axes:

For each absorbed instance w:
1. **Carriers (selected by ACTIVATION MAGNITUDE, independent of the probe):**
   the top **N_CARRIERS = 3** firing latents of w by activation value f_j.
2. **Ablate** the carriers from w's SAE reconstruction: Δ = Σ_{j∈carriers}
   f_j · W_dec[:,j] (the removed contribution, in residual space).
3. **Letter-specific readout:** drop in the **true** letter's probe logit
   d_true = ŵ_L · Δ, versus the mean drop in **every other** scored letter's
   probe logit d_other = mean_{L'≠L} ŵ_{L'} · Δ (ŵ are unit probe directions).
4. **Per-instance statistic:** d_true − d_other. Because the *same* carriers
   are used for both, the magnitude confound (big latents drop any logit)
   cancels; because carriers are chosen by magnitude, not by L-alignment, a
   letter-agnostic carrier gives d_true ≈ d_other ≈ 0. The test **can fail.**

Registered expectation: the top-magnitude latents that fire on an absorbed word
carry that word's **own first letter** more than they carry other letters
(d_true − d_other > 0), i.e. the absorbed letter rides on the word's dominant
(splitting) latents. **NOTE — scope:** this is a *reconstruction-space* causal
attribution, not a model-behavior intervention. The stronger Chanin-style test
(patch the ablated reconstruction back into the residual stream, run the model
forward, measure the drop in the model's own spelling behavior) is the
registered **follow-up** (needs a spelling-prompt format) and is out of scope
for this round.

## Registered predictions

Cell statistic per SAE = absorption rate; compare arch means over 5 seeds
(report mean ± seed SD, and a 10k-seed bootstrap CI on the difference).

- **P1 (PRIMARY):** mean absorption rate (L1) − mean absorption rate (TopK)
  **> 0** at the primary θ = 0, with the seed-bootstrap 95% CI on the
  difference excluding 0. **FALSIFIED** if the CI excludes 0 in the *other*
  direction (TopK > L1); **inconclusive** if the CI straddles 0. (Effect-size
  expectation, not a bar: L1 materially higher, per the L1-splitting
  literature; but the *registered* test is the sign + CI.)
  **P1 robustness (registered):** the sign of (L1 − TopK) must be **stable
  across the θ grid** {0, 0.01, 0.05, 0.1}. If the sign flips with θ, the
  effect is a thresholding artifact and P1 is **not** counted as confirmed
  regardless of the θ = 0 CI (reported, not hidden).
  **P1 matched-L0 gate (registered):** the scorer records each SAE's realized
  (held-out) L0 and gates P1 on **both arches' mean L0 falling in [28, 36]**
  (or the widened [24, 40], disclosed). If either mean L0 is outside, the arms
  are not matched and P1 is **not** confirmed (λ is frozen on one seed, so
  per-seed L0 drift is a real confound and must be surfaced, not assumed away).
- **P2 (causal validity, non-circular):** pooling the per-SAE mean
  (d_true − d_other) across the 10 SAEs, the 95% bootstrap CI is **> 0** — the
  magnitude-selected carriers letter-specifically carry the first letter. If
  the CI includes 0, the carriers are letter-agnostic and the "absorption" the
  metric flags is not carried by the word's dominant latents (a genuine
  negative about the mechanism).
- **P3 (detector recall, secondary):** the fixed pair detector
  (`real_analyze.py`, seeded, opportunity-normalized) is scored for **recall of
  the ground-truth main-L-latents**: the fraction of clean main-L-latents that
  appear in *any* detector-flagged pair (`involved_latents`), by architecture.
  Precision is reported descriptively — the detector also flags feature
  *splitting*, so low precision is expected and is **not** a failure.
  Descriptive; no pass/fail bar, but a locked reporting set.
- **Descriptive:** FVU/L0 Pareto (both arch at the matched point; plus a 2nd
  λ/k point if compute permits), probe accuracies per letter, per-arch
  absorption-rate distribution, θ-sensitivity curves, and the detector's
  opportunity-normalized flag rate and cluster counts.

Honest-scoring: scored by a frozen `analysis/analyze_round12.py`; failures
reported as registered; the report follows the outcome. If L1 does **not**
absorb more than TopK, that is the headline (and would itself revise the
project's L1-splitting intuition at real scale).

## Dev phase (before lock)

- **D1 (SMOKE):** the full pipeline on `pythia-70m` (tiny): extraction +
  word-set + probe + one L1 + one TopK SAE + the absorption metric (θ grid) +
  the **non-circular** causal test + detector recall, end to end on CPU,
  producing sane numbers. *(Done 2026-07-24: pipeline green; the non-circular
  P2 gives d_true≫d_other with magnitude-selected carriers; θ-sign machinery
  and P3 recall wired. Smoke P1 sign is the expected unmatched-L0 artifact,
  fixed by the run's λ-matching.)*
- **λ-calibration:** one L1 seed to pick λ for L0∈[28,36]; record in the lock.
- At lock: metric code, word-set rule, θ, probe protocol, seeds, and the
  scorer are frozen; the lock hash is recorded by amendment.

## Cost & ops

Acts held on GPU → SAE training ~5–10 min each; ~10 SAEs + eval ≈ 2–3 h on a
spot L4. Weights + eval indices → GCS with hashes (`ARTIFACT_MANIFEST`);
per-SAE absorption/causal/detector JSONs (small) → git. Spot-preemption-safe:
each SAE + its eval uploaded to GCS as it completes.

## Review provenance

Dual pre-lock external review (Gemini 2.5 Pro via Vertex; GPT-5.6 via
chatgpt.com pointed at the public repo). "Review" = LLM-assisted adversarial
review, not human peer review. Verdicts + applied changes archived in
`reviews/`; lock hash recorded by amendment.
