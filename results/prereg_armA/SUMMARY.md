# Arm A results: both registered hypotheses fail — trained absorption is *leaky*

**Run:** 2026-07-22, 256 SAEs (one L4 session, ~1.6 h ≈ $1.4), `experiments/
prereg_bimodality_armA.py` at pre-results commit `cfd3e09` (analysis choices
locked before any result existed). Scored by `analysis/analyze_prereg_armA.py`
against the thresholds in `notes/prereg-bimodality-estimator.md`.

## Verdicts against the registered metrics

| # | Registered prediction | Outcome |
|---|---|---|
| M1 (H1 no-go) | binarized signature dist. invariant to ρ (excess TV < 0.02, CI incl. 0) | **Effectively falsified.** Excess TV 0.0215, boot95 [0.0210, 0.0220] — small only because diluted by the background signature space. In-plane excess TV **0.044 ≈ the theoretical maximum leak** (P_HOST·Δρ = 0.045), and the direct mechanism diagnostic TV(host-only sigs vs host+child sigs) = **0.9999**: the binarized code separates the two sub-populations essentially perfectly. |
| M2 (H2 estimator) | 2-comp GMM weight recovers ρ (mean err < 0.02, r > 0.9) at σ=0 | **FALSIFIED** at σ=0: mean err 0.483, r 0.234. (But see M4: works at σ=0.1.) |
| M3 | estimator route ≫ binarized route | **INVERTED.** Binarized signature counting wins at every ρ: err 0.0007/0.020/0.019/0.012 at ρ=0.02/0.05/0.10/0.20 vs GMM 0.39–0.56. |
| M4 (H3 SNR) | finite σ* above which estimator degrades monotonically | **Premise wrong — relationship is inverted.** Estimator fails at σ=0 (err 0.46), works at σ=0.1 (err **0.0075**, n=14), and absorption itself disappears at σ ≥ 0.2 (0/16 absorbed; SAEs go faithful, child cos 0.80–0.90). |
| M5 (H4 multi-child) | per-child recovery degrades with m | **Untestable as configured:** at m ≥ 2 (child rates 0.04–0.16) 0/16 runs formed a mono-composite — children were simply erased (cos_child ≈ 0.05, cos_parent ≈ 0.98). |

Decision rule applied: **"H1 falsified → re-open §2"** of
`notes/label-free-frequency-identifiability.md`.

## The mechanism: gated ("leaky") absorption

The §2/§3 idealization assumed one composite latent `w=(v_p+v_c)/√2` serving
BOTH sub-populations (activations 1/√2 vs √2). Trained absorbed SAEs do
something else — exactly what this project's own `theory/theory_merged.py`
absorbed branch predicts (in-plane pair {≈0°, ≈46°}):

- a **parent-aligned latent** handles host-only events;
- the **composite latent is encoder-gated**: it fires on **100%** of
  host+child events (mean act ≈ 1.28) and on only 3–20% of host-only events
  at mean act ≈ **0.01** — i.e. ~zero.

So the *binarized* code is a near-perfect child detector (cond TV ≈ 1.0), and
counting signatures — ρ̂ = P(composite fires)/P(composite or parent fires) —
recovers ρ to ≤ 0.02 everywhere. Meanwhile the *magnitude* distribution inside
the composite is unimodal-plus-a-zero-smear at σ=0, so the registered mixture
estimator fits noise (hence err ≈ 0.5). At σ=0.1 the noise leaks enough
host-only mass through the gate to re-create a second mode, and the GMM weight
tracks ρ strikingly well (err 0.0075) — a calibration we can't fully account
for from the retained summary stats (raw histograms not kept; follow-up).

Even the capacity-forced X block (m_lat=31, built to realize §2's shared-
composite regime) refused the idealization: cond TV 0.998 — gating, not
sharing, again.

## Additional findings (exploratory, disclosed)

1. **Absorption formation is trainability-limited in ρ**: absorbed fraction
   rises 4/16 → 8/16 → 11/16 → 14/16 across ρ = 0.02→0.20 (σ=0). Non-absorbed
   runs are **child-erased** (cos_parent ≈ 0.98, cos_child ≈ 0.15), not
   faithful — at these joint rates (0.5–5% of events) SGD often never builds
   the composite. Echoes the round-2 A-oracle trainability regimes.
2. **Activation noise σ ≥ 0.2 destroys absorption in favor of faithful child
   latents** (cos_child 0.80–0.90 at σ=0.2). Noise-as-remedy is a new,
   pre-registerable direction.

## Caveats

- ρ̂_bin uses the oracle-identified (parent, composite) latent pair, per Arm A
  scope (oracle for identification/scoring only — same allowance the GMM route
  received). The label-free **pair-identification** problem is now the entire
  remaining gap; note the pair has a distinctive geometric signature (unit-norm
  decoders ≈ 45° apart with near-disjoint firing) that a label-free scan could
  target.
- Confirmatory cells have reduced n after the registered absorption filter
  (disclosed above; worst: ρ=0.02, n=4).
- §2's algebra remains correct *for its stated model*; what failed is
  transfer — trained SAEs do not realize that model, even when capacity-forced.
