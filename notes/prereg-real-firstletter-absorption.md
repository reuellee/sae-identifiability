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
(H1). Secondary (P2, **descriptive**): the absorbed first letter is
*concentrated* in the word's dominant (magnitude-selected) latents — a
reconstruction-space attribution, **not** a causal-validity claim (the
model-behavior causal test is the deferred Chanin follow-up). Secondary (P3):
the label-free pair detector, scored **blind**, recalls the ground-truth
main-L-latents better than its opportunity baseline.

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
   0.05, 0.1} as a *descriptive, biased upper bracket* (θ > 0 is unmatched — see
   the θ-grid note under P1). L-latent = argmax_j s_j; require s_j ≥ 0.30 else
   "no clean L-latent" (that letter excluded from the SAE's score, disclosed;
   the count `n_letters_scored` is compared across arms — review Finding 3).
3. **Absorbed instance:** an L-word w is **absorbed** iff it is letter-present
   (probe > 0.5) AND the L-latent does **not** fire on w (act ≤ θ). (The model
   represents the first letter, but the dedicated latent misses it → absorbed
   elsewhere.)
4. **Absorption rate (per SAE):** over letters with a clean L-latent,
   `Σ 1[letter-present ∧ L-latent misses] / Σ 1[letter-present]` pooled over
   all L-words of all scored letters — the fraction of first-letter-present
   words the main L-latent fails on. Computed at the primary θ and at every
   grid θ.

## P2 attribution (FROZEN) — DESCRIPTIVE, reconstruction-space (not a causal bar)

**Scope, stated up front (two review rounds pushed on this).** The genuine
causal-validity test is the Chanin-style **model-behavior intervention** (patch
the ablated reconstruction back into the residual stream, run the model forward,
measure the drop in the model's own spelling behavior). That needs a
spelling-prompt format and is the registered **follow-up** — *out of scope for
this round*. What P2 reports here is a **reconstruction-space attribution**: of
the firing latents that carry an absorbed word, is its first letter concentrated
in the *dominant* (magnitude) latents? P2 is **descriptive** (no confirm/falsify
bar on the project's thesis), because a pure sign test here is near-tautological
(see below).

For each absorbed instance w, carriers = top **N_CARRIERS = 3** firing latents
by **activation magnitude** (chosen independently of the probe). Let Δ_top =
Σ_{j∈carriers} f_j·W_dec[:,j] and Δ_rand the same for a random equal-count set
of w's *other* firing latents. Two contrasts:

1. **Concentration (the falsifiable one):** cos(Δ_top, ŵ_L) − cos(Δ_rand, ŵ_L).
   Cosine **normalizes out magnitude**, so this is *not* the near-tautology
   "does the reconstruction contain the letter that is present" — it asks
   whether the letter is **concentrated in the dominant latents** vs diffuse.
   If the letter sits in the tail, cos(Δ_top,ŵ_L) < cos(Δ_rand,ŵ_L) and this is
   **negative**. Genuinely can fail.
2. **Specificity (descriptive only):** d_true − d_other, where d_true = ŵ_L·Δ_top
   and d_other = mean_{L'≠L} ŵ_{L'}·Δ_top. Reported for interpretability, but
   flagged **near-guaranteed positive** for any low-FVU SAE (w contains L, not
   L'), so it is *not* evidence for anything and carries no bar.

Space note: Δ is in the SAE's *normalized* residual space, ŵ in *raw* space;
normalization is a **scalar**, so the cosine (concentration) is exactly
invariant and the sign of the specificity contrast is valid. Breaks under
per-dimension normalization.

## Registered predictions

Cell statistic per SAE = absorption rate; compare arch means over 5 seeds
(report mean ± seed SD, and a 10k-seed bootstrap CI on the difference).

- **P1 (PRIMARY):** mean absorption rate (L1) − mean absorption rate (TopK)
  **> 0** at the primary **θ = 0** (the *only* matched-sparsity point), with the
  seed-bootstrap 95% CI on the difference excluding 0. **FALSIFIED** if the CI
  excludes 0 in the *other* direction (TopK > L1); **inconclusive** if it
  straddles 0. θ = 0 is deliberately **conservative for L1** (its soft acts make
  its L-latent fire a hair more often → *fewer* L1 misses), so L1 > TopK *here*
  is the strong result. (Effect-size expectation, not a bar: L1 materially
  higher, per the L1-splitting literature; but the *registered* test is sign + CI.)
  **P1 matched-L0 gate (registered):** the scorer records each SAE's realized
  (held-out) L0 and gates P1 on **both arches' mean L0 in [28, 36]** (or widened
  [24, 40], disclosed). If either mean L0 is outside, the arms are not matched
  and P1 is **not** confirmed (λ is frozen on one seed, so per-seed L0 drift is
  a real confound and must be surfaced).
  **P1 matched-letter robustness (registered):** because L1 splitting can make
  more letters fail the clean-L-latent test (s_j ≥ 0.30) — so the arms could be
  scored on *different* letter subsets — the scorer also reports `n_letters_scored`
  per arm **and** recomputes absorption on the **intersection** of letters both
  arms score cleanly. If the L1 > TopK sign does not survive on the common
  letters, P1 is a letter-subset artifact (reported).
  **θ grid — DESCRIPTIVE only (not a gate):** absorption at θ ∈ {0, 0.01, 0.05,
  0.1} is reported, but θ > 0 is **unmatched** (a positive threshold zeros more
  of L1's small acts than TopK's, dropping L1's effective L0 and *inflating* its
  absorption) — it is an **upper bracket biased toward L1**, so it can only
  *disconfirm* (a gap that shrinks/reverses as θ rises argues against L1). It is
  **not** a robustness confirmation. (Review Finding 2.)
- **P2 (attribution, DESCRIPTIVE — no bar on the thesis):** pooling the per-SAE
  **concentration** contrast (cos(Δ_top,ŵ_L) − cos(Δ_rand,ŵ_L)) across the 10
  SAEs, report the 95% bootstrap CI. **> 0** ⇒ the absorbed first letter is
  concentrated in the word's *dominant* latents (consistent with absorption into
  word-specific splitting latents); **≤ 0** ⇒ diffuse/tail. Reported by arch.
  The specificity contrast (d_true − d_other) is reported but carries no weight
  (near-guaranteed positive). Neither substitutes for the deferred Chanin
  model-behavior test. (Review Finding 1.)
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
  the P2 attribution (concentration + specificity) + detector recall + the
  intersection-matched-letter robustness, end to end on CPU, producing sane
  numbers. *(Done 2026-07-24: pipeline green. The magnitude-normalized
  concentration contrast (+0.065) is far below the near-tautological specificity
  (+0.49), confirming the normalization strips the guaranteed part; L0 gate,
  θ-bracket labels, n_letters comparison, and intersection machinery all wired.
  Smoke P1 is the expected unmatched-L0 artifact, gated off, fixed by λ-matching.)*
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
