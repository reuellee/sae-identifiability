# Pre-lock Review: Round 15 (SAE Identifiability)

Here is the adversarial pre-lock review of the Round 15 cross-validation harness and evaluator.

## 🚨 P1 (Fatal: Invalidates the round or crashes at scale)

### 1. Missing mean-centering (decoder bias subtraction) in the JumpReLU encoder
**File:** `experiments/gemmascope_crossval.py` (Line 211)
**Defect:** The pre-activation is computed as `pre = xb @ s["W_enc"] + s["b_enc"]`.
**Attack:** Gemma Scope JumpReLU SAEs are trained on mean-centered activations. The proper formulation explicitly subtracts the pre-encoder bias (which in these releases is `b_dec`) prior to the encoder projection: `pre = (xb - s["b_dec"]) @ s["W_enc"] + s["b_enc"]`.
**Consequence:** Without subtracting `b_dec`, the encoder receives out-of-distribution inputs that are massively offset by `b_dec @ W_enc`. The features will misfire, sparsity will be ruined, and reconstructions will be severely degraded. This will cause the SAEs to fail Gate 1 (`FVU <= 0.5`) loudly and crash the round.

## ⚠️ P2 (Weakens interpretation)

### 2. Spurious correlations from improper tie-handling in `spearman3`
**File:** `analysis/analyze_round15.py` (Lines 44-45)
**Defect:** The custom Spearman correlation function computes ranks via `argsort(argsort())`.
**Attack:** `argsort` uses a stable sort that assigns strictly distinct ranks (e.g., 0, 1, 2) even when the input values are identical. Because `rate_family` values are fractions with small denominators (often ~30-50 absorbed trials), exact numerical ties across the width series are probable. If a flat sequence like `[0.5, 0.5, 0.5]` occurs, it will be ranked `[0, 1, 2]`, falsely producing a perfect `+1.0` correlation instead of `0.0`.
**Consequence:** This will artificially inflate or distort the descriptive (D) monotonicity claim if rates plateau or tie.
**Fix:** Use `scipy.stats.spearmanr` (since `scipy` is installed on the VM) to handle tied ranks correctly with averages.

## 🛠️ P3 (Polish / Minor Ops)

### 3. Memory bandwidth thrashing from massive boolean array slicing
**File:** `experiments/gemmascope_crossval.py` (Line 173)
**Defect:** `sel = fires[yL == 1].mean(0) - fires[yL == 0].mean(0)`
**Attack:** `fires` is a ~6.3 GB boolean array at width 262k. The slice `fires[yL == 0]` allocates a new ~6.26 GB boolean array, computes the mean, and drops it. Doing this sequentially for ~26 letters inside the loop forces ~160 GB of memory copying.
**Consequence:** While this won't OOM the 32GB `e2-standard-8` VM, it will severely bottleneck CPU and memory bandwidth, extending the scoring runtime significantly.
**Fix:** Compute the negated mean without allocation: `(fires.sum(0) - fires[yL == 1].sum(0)) / (yL == 0).sum()`.

### 4. Fluctuation noise in pooled letter bootstrap (ratio estimator)
**File:** `analysis/analyze_round15.py` (Lines 28-30, 143-144)
**Defect:** For P3, letters clean in all 3 widths append 3 values to `d3[L]`, while letters clean in only 1 width append 1 value. `boot_ci_letters` resamples letters (keys) and flattens them into `vals`.
**Attack:** The total denominator `len(vals)` fluctuates randomly across bootstrap iterations depending on how many 3-value vs 1-value letters are drawn.
**Consequence:** While statistically valid as a pooled ratio estimator, this introduces mathematically unnecessary variance to the CI.

### 5. Missing programmatic enforcement of Gate 3
**File:** `analysis/analyze_round15.py` (Gates section)
**Defect:** The prereg states: "Words gate. >= 30 words/letter for >= 20 letters."
**Attack:** While the `gemmascope_crossval.py` script correctly filters out letters with `< 30` words, the `analyze_round15.py` evaluator never explicitly checks if the *resulting* number of letters is `>= 20`.
**Consequence:** If the subsample yields fewer than 20 valid letters, the script will silently proceed instead of loudly failing the gate.
