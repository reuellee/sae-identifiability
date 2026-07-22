# Capacity rerun (m ∈ {32, 34, 40}): the reviewer was right — one slot of headroom and SGD finds the triple

288 runs (2 cells × 6 ε/ε\* × 3 widths × 8 seeds), round-2 trainer/event model,
one L4 session (`m33.log`, 357 s). Pre-registered K1–K3 in
`experiments/capacity_m33_rerun.py` (commit `0cba6b2`, before results).

## Verdicts

| # | Prediction | Outcome |
|---|---|---|
| K1 | triples (parent + child + composite latents coexisting) appear at m ≥ 33 for ε > 0 in a majority of runs | **CONFIRMED**: m=34 → 55/80 ε>0 runs (69%); m=40 → 41/80 (51%); combined 60%. At m=32: architecturally blocked, and indeed absent. |
| K2 | m=32 control stays triple-free | **CONFIRMED**: 0/96. |
| K3 | functional child transition shifts earlier with headroom | **CONFIRMED, strongly**: φ_func crosses 67.5° between 0.5–0.75·ε\* at m=32 (matching round-2's 0.58–0.70), between 0.25–0.5·ε\* at m=34, and *below* 0.25·ε\* at m=40 (φ ≈ 92–100° from 0.25·ε\* up — near-faithful child representation almost as soon as ε > 0). |

Full grid in `m33_runs.csv` (per-seed rows; dispersion visible per cell).

## Reading

**The withdrawn round-1 claim is now refuted in its original form, not just
unsupported.** "Optimization dynamics select absorption even without capacity
scarcity" is the opposite of what happens: give SGD one latent of genuine
headroom (m = 34 = 30 bg + 3 + 1 spare) and it finds the redundant triple —
the unconstrained population optimum — in most ε > 0 runs, and the practical
transition moves toward "child represented whenever ε > 0."

This *sharpens* the paper's story rather than weakening it:

- Theorem 2's capacity-limited two-latent competition is the operative
  mechanism for absorption, full stop. The round-1/2 quantitative agreement
  (λq scaling collapse, 0.58–0.70·ε\* midpoint) is the m = 32 constrained
  regime behaving as the theory says — and that is the *realistic* regime,
  since real SAEs cannot afford a latent per feature combination.
- The §14 spare-capacity composition result (m = 1536, triples everywhere) is
  now continuous with the toy model: composition begins at literally one
  spare slot.

**Wrinkle, disclosed:** the triple *rate* dips from 69% (m=34) to 51% (m=40)
even as φ_func gets *more* faithful. Cause: with more spare latents,
feature-splitting spreads the child/composite directions over several
sub-threshold (ρ < 0.8) latents, failing the round-1 triple criterion while
the functional child metric stays faithful — the same splitting behavior the
pair-ID null condition surfaced. The criterion, not the phenomenon, degrades
with width. One spurious ε=0 "triple" at m=40 (1/16 ε=0 runs across widths)
is a splitting artifact of the same kind.

Report §8's correction note is replaced by this result; see also
`reviews/RESPONSE_GPT-5.6.md`.
