# Pre-registration: real-model causal first-letter absorption, L1 vs TopK (round 12)

*Drafted 2026-07-24. The whole-repo review's single highest-value next
experiment: a confirmatory matched-seed L1-vs-TopK first-letter-absorption
comparison on real SAEs, with a ground-truth residual probe, a retention check
separating absorption from feature loss, and a descriptive reconstruction-space
attribution (P2).*

**STATUS: LOCKED 2026-07-24.** Frozen artifacts = the metric
(`experiments/real_firstletter.py`), the scorer (`analysis/analyze_round12.py`),
the word-set rule, θ=0, the probe protocol, 8 seeds, and the P1/P2/P3 predictions
below, as of commit `0722212` (this lock commit changes only status). Dual
pre-lock LLM-assisted adversarial review complete: **Gemini 2.5 Pro → LOCK-READY**
(2 rounds); **GPT via codex → 12 findings, all dispositioned** (`reviews/`), the
3 that let P1 mislead fixed and **re-verified by GPT as correctly implemented**
(loss-vs-absorption retention, ΔL0-difference gate, matched-letter verdict gate).
The only remaining pre-run free parameter is the L1 **λ**, fixed by the registered
calibration on the L4 (seed 0, L0∈[28,36]) and appended by **amendment +
`LOCK_LAM`** before the 16-SAE run. After lock, changes only by dated amendment.*

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
  by amendment (and passed to the scorer as `LOCK_LAM`) before the 16-SAE run.
- **Seeds:** **8 per architecture (0–7)**, identical corpus/token-order/
  init-scheme/width/step-budget (15k steps). 16 SAEs. Per seed s, L1 and TopK
  share the **same init** (torch seed s) and data order, so seed s is a genuine
  **matched pair** → P1 uses a *paired* per-seed bootstrap (review #7). 8 seeds
  puts the exact paired sign-test floor at 2/2⁸ ≈ 0.008 < 0.05.
- Weights → GCS; per-SAE stats (FVU on a **held-out** eval slab, L0, dead%).

## First-letter absorption metric (FROZEN definition)

Per SAE, per letter L ∈ {a…z} with ≥ 30 words:

1. **Residual probe (ground truth for "is the first letter present"):** a
   `class_weight="balanced"` logistic-regression probe on the **residual**
   activations predicting ℓ(t)==L (5-fold; out-of-fold predictions). A word w is
   **letter-present** iff its out-of-fold probe prob > 0.5. (Class-weighting, not
   resampling, handles the ℓ==L minority — the wording is exact per review #12.)
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
3. **Retention (separates absorption from LOSS — review Finding 1, CRITICAL):**
   a word's first letter must **survive in the SAE reconstruction** to count as
   absorbed. A *full-fit* letter probe is applied to the reconstruction
   x̂ (mapped back to raw residual space, x̂/scale + μ); `retained(w)` iff that
   probe > 0.5. Without this, an SAE that simply reconstructs the letter *worse*
   would score as "more absorbing" (feature loss masquerading as absorption).
4. **Absorbed instance:** an L-word w is **absorbed** iff it is letter-present
   (probe(x) > 0.5) **AND** letter-retained (probe(x̂) > 0.5) **AND** the L-latent
   does **not** fire on w (act ≤ θ). If letter-present but **not** retained, w is
   **lost** (reported as `loss_rate`, not absorption).
5. **Absorption rate (per SAE):** over letters with a clean L-latent,
   `Σ 1[present ∧ retained ∧ L-latent misses] / Σ 1[present]` pooled over all
   L-words of all scored letters. `loss_rate` = `Σ 1[present ∧ ¬retained ∧
   L-latent misses] / Σ 1[present]` is reported alongside so P1 is not confounded
   by feature loss. Both computed at the primary θ; absorption also at every grid θ.

*Note (review #6, disclosed): the main L-latent is selected by max selectivity
and misses are counted on the same words — a mild winner's-curse that
**under-estimates** absorption, and more so for L1 (more splitting → more argmax
overfit). This bias is **conservative for H1** (works against finding L1 > TopK),
so it is disclosed rather than corrected with a word split (which would shrink
the eligible-letter set and worsen the matched-letter check).*

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
by **activation magnitude** (chosen independently of the probe). *(N_CARRIERS = 3
is a frozen, declared choice — a small fixed set capturing the word's dominant
latents; because P2 is descriptive, the concentration value may be sensitive to
it, and that sensitivity is a caveat on the descriptive read, not a bar. Review
noted this as non-blocking.)* Let Δ_top =
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

Cell statistic per SAE = absorption rate; the primary comparison is a **paired
per-seed bootstrap** of (L1_seed_s − TopK_seed_s) over the 8 seed-pairs (10k
resamples), plus the per-seed diffs and the sign-consistency across pairs.

- **P1 (PRIMARY, SUGGESTIVE):** mean **paired** (L1 − TopK) absorption rate
  **> 0** at the primary **θ = 0** (the *only* matched-sparsity point), 95%
  bootstrap CI excluding 0. **FALSIFIED** if the CI excludes 0 in the *other*
  direction; **inconclusive** if it straddles 0. θ = 0 is deliberately
  **conservative for L1** (its soft acts make its L-latent fire a hair more
  often → *fewer* L1 misses), so L1 > TopK *here* is the strong result. Framed
  **SUGGESTIVE** (8 seeds is modest power; the exploratory scale is stated), and
  P1 "CONFIRM" additionally requires **all** of the gates below to pass:
  - **Config-conformance gate:** the frozen scorer verifies every `_fl.json` was
    produced with the registered θ=0, sel_min=0.30, n_carriers=3, model/layer,
    k=32 (TopK), and the locked λ (L1, constant across seeds). Any mismatch ⇒ not
    confirmed (review #2 — the frozen scorer must enforce the frozen design).
  - **Seed gate:** exactly the registered seed set {0…7} present in both arms
    (review #3 — no duplicated/absent/extra runs entering the bootstrap).
  - **Matched-L0 gate (tests MATCHING, not just a band):** |mean L0_L1 − mean
    L0_TopK| ≤ **3** **and** both means in [24, 40] near the target 32. A
    both-in-a-wide-band test (e.g. L1=28 vs TopK=36) would pass an 8-unit
    mismatch — so the gate is on the **difference** (review #4). Widening the
    band is a **lock-time amendment from calibration**, not an automatic post-hoc
    branch. L0 is the held-out `#{act>0}` from the trainer.
  - **Matched-letter gate (feeds the verdict):** absorption recomputed on the
    **intersection of letters clean in EVERY SAE of both arms** (true intersection,
    not a majority, so denominators match). If the L1 > TopK sign does **not**
    hold there, P1 is a letter-subset artifact and is **not** confirmed (review #5;
    `n_letters_scored` per arm is also reported).
  **θ grid — DESCRIPTIVE only (not a gate):** absorption at θ ∈ {0, 0.01, 0.05,
  0.1} is reported, but θ > 0 is **unmatched** (a positive threshold zeros more
  of L1's small acts than TopK's, dropping L1's effective L0 and *inflating* its
  absorption) — it is an **upper bracket biased toward L1**, so it can only
  *disconfirm* (a gap that shrinks/reverses as θ rises argues against L1). It is
  **not** a robustness confirmation. (Review Finding 2.)
- **P2 (attribution, DESCRIPTIVE — no bar on the thesis):** the per-SAE
  **concentration** contrast (cos(Δ_top,ŵ_L) − cos(Δ_rand,ŵ_L)) is bootstrapped
  **per architecture** (L1 and TopK separately, 95% CI). **> 0** ⇒ the absorbed
  first letter is concentrated in the word's *dominant* latents (consistent with
  absorption into word-specific splitting latents); **≤ 0** ⇒ diffuse/tail. The
  specificity contrast (d_true − d_other) is reported but carries no weight
  (near-guaranteed positive). *Caveat (review #9, #10): the probe direction is
  in-fold for the attributed word and the random-tail control is
  magnitude-asymmetric, so the concentration value may be inflated — it is
  descriptive only and does not substitute for the deferred Chanin model-behavior
  test.*
- **P3 (detector recall, secondary):** the fixed pair detector
  (`real_analyze.py`, seeded) is scored for **recall of the ground-truth
  main-L-latents** — the fraction of clean main-L-latents in *any* flagged pair
  (`involved_latents`) — **against the opportunity baseline** = |involved|/m (the
  chance recall if the flagged set were random), reported as an **enrichment**
  (recall / baseline) so a detector that flags most latents does not earn free
  recall (review #8). Precision (|main-L ∩ involved|/|involved|) is reported
  descriptively — the detector also flags feature *splitting*, so low precision
  is expected and is **not** a failure. Descriptive; no pass/fail bar.
- **Descriptive:** FVU/L0 Pareto (both arch at the matched point; plus a 2nd
  λ/k point if compute permits), probe accuracies per letter, per-arch
  absorption-rate distribution, θ-sensitivity curves, and the detector's
  opportunity-normalized flag rate and cluster counts.

Honest-scoring: scored by a frozen `analysis/analyze_round12.py`; failures
reported as registered; the report follows the outcome. If L1 does **not**
absorb more than TopK, that is the headline (and would itself revise the
project's L1-splitting intuition at real scale).

## Dev phase (before lock)

- **D1 (SMOKE):** the full pipeline on `pythia-70m` (tiny), 2 seeds/arch:
  extraction + word-set + probe + retention + absorption (θ grid) + loss_rate +
  P2 concentration (by arch) + detector recall/baseline + all P1 gates
  (conformance, seed, ΔL0, matched-letter paired) end to end on CPU.
  *(Done 2026-07-24: pipeline green. The retention check split TopK's raw 0.444
  into 0.222 absorbed + 0.222 lost and L1's 0.089 into 0.0 absorbed + 0.089 lost
  — exactly the loss-vs-absorption confound review #1 warned of, now separated.
  Concentration (+0.13, magnitude-normalized) ≪ near-tautological specificity
  (+1.3). All gates fire: conformance flags the 70m/L3 model, ΔL0 gate catches
  the unmatched L0, matched-letter sign correctly "does not hold", paired
  bootstrap reports per-seed diffs + sign floor. P1 gated off as expected;
  the real run on Pythia-1.4B is where conformance/L0 pass.)*
- **λ-calibration:** one L1 seed to pick λ for L0∈[28,36]; record in the lock.
- At lock: metric code, word-set rule, θ, probe protocol, seeds, and the
  scorer are frozen; the lock hash is recorded by amendment.

## Cost & ops

Acts held on GPU → SAE training ~5–10 min each; ~16 SAEs + calibration + eval
≈ 2–3 h on a spot L4. Weights + eval indices → GCS with hashes
(`ARTIFACT_MANIFEST`); per-SAE absorption/attribution/detector JSONs (small) →
git. Spot-preemption-safe: each SAE + its eval uploaded to GCS as it completes.

## Review provenance

Dual pre-lock external review (Gemini 2.5 Pro via Vertex; GPT-5.6 via
chatgpt.com pointed at the public repo). "Review" = LLM-assisted adversarial
review, not human peer review. Verdicts + applied changes archived in
`reviews/`; lock hash recorded by amendment.
