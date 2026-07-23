# Deep-research sweep 2026-07-23: OrtSAE prediction status + round-9 novelty

*Method: 100-agent fan-out web research (5 search angles → source fetch →
per-claim 3-vote adversarial verification), run from the local session.
All claims below survived ≥2/3 adversarial verification against primary
sources; confidence labels are the harness's. This replaces the blocked
attempt recorded in `GEMINI_DEEP_RESEARCH.md` (2026-07-21).*

## Part A — the §7.1b coherence-penalty prediction vs OrtSAE

1. **OrtSAE (arXiv:2509.22033) cannot test the overdose prediction: it has
   no γ ablation.** γ = 0.25 in every experiment (GPT-2, Gemma-2-2B,
   Llama-3-8B); the only hyperparameter ablation (App. C) varies chunk count
   K, and the loss-frequency variant deliberately holds effective γ constant.
   [high]
2. **Its single operating point is consistent with the bounded-effect half
   of the no-go.** Absorption shrinks ~65% vs BatchTopK but is strictly
   nonzero everywhere (0.05–0.095 vs 0.148–0.220), and stays 3–4× above
   Matryoshka's (0.015–0.022). C2R (arXiv:2606.30609) independently
   confirms the residual floor and outperforms OrtSAE. [high]
3. **The predicted non-monotonic overdose pattern appears in the only
   direct penalty-strength sweep found: MetaSAEs (arXiv:2604.03436).**
   Intermediate decomposability-penalty weights (λ₂ ∈ {0.1, 0.3}) are
   Pareto-optimal (−7.5% mean pairwise co-activation), while λ₂ = 1.0
   *worsens* co-activation (+2.6–3.8%) at severe reconstruction cost.
   [medium — single-author preprint, one model/layer, decomposability
   penalty ≠ literally a pairwise coherence penalty, |φ| ≠ SAEBench
   absorption]
4. Ecosystem-wide: no size/sparsity configuration eliminates absorption
   (Gemma Scope sweep; L1 and TopK on Qwen2-0.5B/Llama-3.2-1B) [high];
   SASA (arXiv:2606.06333) leaves an 18.3%/11.9% floor on Mistral-7B
   [medium].

**Consequence for report §7.1b:** the OrtSAE-ablation test is *unrunnable
on the public record* — score the prediction as untested there, cite the
single-point consistency, and cite MetaSAEs as the closest existing
penalty-strength sweep, matching the predicted shape in a neighboring
penalty family. The strong-form test (γ sweep on a fixed absorption metric)
remains an open external experiment.

## Part B — round-9 (gating-corrected ρ̂) novelty

5. **The leak phenomenon is prior art, well documented.** Full absorption:
   the parent latent's encoder learns ≈ (parent ∧ ¬child) — gates off
   exactly when the child co-fires (Chanin et al. arXiv:2409.14507 App.
   A.3). Partial absorption: the parent latent fires weakly on joint tokens
   (Feature Hedging, arXiv:2505.11756 — which also documents that narrow
   Matryoshka prefix levels trade absorption for hedging). Both regimes
   bias binarized co-activation counting. [high]
6. **No prior estimator correction found.** Exhaustive full-text search of
   the absorption line found zero mixture/EM/latent-variable or other
   statistical corrections applied to binarized co-activation counts; the
   only proposed handlings are architectural (Matryoshka, snap loss) or
   metric-threshold tweaks (τ relaxations that concededly miss partial
   absorption). Round 9's estimator correction **appears unclaimed**.
   [medium — absence-of-evidence claim; keep the usual "to our current
   knowledge" hedge]
7. Nothing relevant published strictly after 2026-07-21; the newest
   relevant works are June 2026 (C2R, SASA). [high]

**Consequence for round 9:** proceed; novelty language stays hedged per
claim 6's confidence. Cite 2409.14507 App. A.3 + 2505.11756 as the
mechanism's prior art in the theory note (already partially cited via
report §16).
