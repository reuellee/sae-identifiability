# Round 10: absorption under TopK SAEs — registered scoring (a largely negative round)

Prereg `notes/prereg-topk-absorption.md`, **lock `f2e92fc`** (bars, metric,
arms, scorer frozen; nothing changed after lock). Run: 960 fresh-seed runs
(arm A 768 + arm B 96 + arm C 96), one spot-L4 session (us-central1-a, ~8 min
wall), weights saved. Scored by the frozen `analysis/analyze_round10.py`.
Activation-aware binary child-recovery metric (signed cos ≥ 0.6, fires on
child-solo ≥ 0.5, child-selective ≥ 0.3 — no geometry/missingness loopholes).

**This round mostly did not go as predicted, and the negative result is the
finding.** The theory (2-atom oracle propositions) is verified and unchanged;
what failed is its *SGD realization* in the capacity-limited arm, and the
practical hypothesis that TopK would resist absorption.

## Registered verdicts (as scored)

| Prediction | Verdict |
|---|---|
| **P1** (two-atom crossover + q-scaling, arm A m=2 k=1) | **INCONCLUSIVE** — recovery never crosses 0.5 at any ε (max 0.21–0.25); ε_mid right-censored both q. |
| **P2** (two-atom capacity collapse, arm A m=2) | **FALSIFIED** — in every predesignated cell k=2 does **not** recover (recovery ≤ 0.08, gaps ≈ 0); SGD stays out of the faithful basin regardless of budget. |
| **P3** (overcomplete escape, arm B m=16 TopK) | **PARTIAL** — TopK recovers the child (0.62–0.83) but not cleanly ≥ 0.75 at every ε (0.62 at ε=0.10). |
| **P4** (TopK resists L1 absorption, descriptive) | **REFUTED (direction inverted)** — L1 recovers **1.00** at every ε; TopK 0.62–0.83; ΔTopK−L1 = −0.17 … −0.38. |

## What actually happened (diagnostics)

- **Arm A (m=2) is a degenerate SGD regime, not a clean absorption test.**
  Reconstruction stays high (rec ≈ 0.42–0.51 on the pair events — a clean
  faithful or absorbed 2-atom dictionary would give ≈ 0), and the two atoms
  are **non-selective**: the child-side latent fires on parent-solo events at
  ≈ 0.58–0.63, almost as often as on child-solo (≈ 0.67), so
  child-selectivity (fire|csolo − fire|psolo ≥ 0.3) is essentially never met.
  SGD on m=2 does not converge to the clean {v_p, v_c} or {v_p, d_comp}
  dictionaries the theorem compares; it smears. So P1/P2 are the
  **SGD-reachability gap** the pre-lock reviewers flagged — the 2-atom
  *global-optimum* propositions (M0-verified) are not the SGD outcome in this
  tiny, ~55%-zero-token regime. The theorem is not refuted; its SGD
  realization in m=2 is.
- **Overcomplete SAEs (m=16) recover the child cleanly — both architectures.**
  Arm C (L1): rec ≈ 0.01–0.02, child atom cos 0.98–0.99, fires on child-solo
  1.00, on parent-solo **0.00** → recovery **1.00** at every ε including 0.05.
  Arm B (TopK k=1): rec ≈ 0.04–0.14, child atom cos 0.77–0.93, fires on
  child-solo 0.96–1.00 but on parent-solo 0.13–0.29 → recovery 0.62–0.83. The
  free dedicated child atom forms in both (the overcomplete "escape" of §6),
  confirming the *qualitative* prediction that width lets the child be
  recovered for ε > 0.
- **The hard TopK budget slightly *degrades* rare-feature recovery.** The
  child atom must win the top-1 slot to fire, so it receives fewer training
  signals and stays less clean (cos 0.77–0.93 vs L1's 0.99; nonzero
  parent-solo firing), lowering recovery below L1's. This is the opposite of
  "TopK resists absorption."

## Corrected takeaways (honest, and they refine the program)

1. **Dictionary width, not the per-token budget, is what lets the child be
   recovered here.** m=16 recovers; m=2 is degenerate. The binding "capacity"
   is spare atoms for a dedicated child latent — realized by width.
2. **In isolation (no background), neither L1 nor TopK absorbs an ε > 0
   child.** L1 recovers it perfectly. So the L1 absorption documented in the
   earlier rounds is **driven by background competition** (the composite is
   "good enough" and the child atom competes with reconstructing many
   background features), **not by child rarity alone.** This is a genuine
   refinement of the project's own L1 absorption story and a concrete next
   experiment (L1-vs-TopK head-to-head *with* background, where L1 does
   absorb).
3. **The "TopK resists L1 absorption" hypothesis is refuted in this setup;** if
   anything the hard budget hurts clean recovery of rare features. Whether
   TopK helps or hurts in the background regime (where absorption is real) is
   now the open question.

## Scope and honesty

Registered predictions were locked and scored mechanically; three of four
failed and are reported as such. The m=2 degeneracy and the
isolated-L1-doesn't-absorb finding are **post-hoc diagnoses** of *why* they
failed — they do not change the verdicts. The theorem (2-atom oracle
propositions) stands; its SGD realization and the practical hypothesis did
not. Queued follow-ups: (a) a non-degenerate capacity-limited test (larger d,
fewer zero tokens, or an explicit 2-atom-allocation constraint) to test the
2q crossover at SGD level; (b) the L1-vs-TopK head-to-head **with background**,
the regime where absorption actually occurs.

**The decisive next experiment (b), sharpened** (results-stage review,
`reviews/ROUND10_RESULTS_REVIEW_GEMINI-2.5-PRO_2026-07-24.md`): the original
"TopK resists L1 absorption" hypothesis was meaningful only *with* background,
where L1 absorption is real — and this round tested it where it isn't. The
head-to-head with background disambiguates two live hypotheses: (i) TopK's hard
budget keeps *hurting* rare-code recovery (fewer training signals → worse than
L1); or (ii) the hard budget *forces a clean choice* — an atom must represent
the child or the background, it cannot form a "good-enough" composite that
smears a bit of each (the hypothesized L1 absorption mechanism) → TopK more
identifiable. Disambiguating these directly advances the north-star's
*identifiable codes* stage. Verdict of the results-stage review: **ACCEPT** —
honest, correctly-scoped negative round.

## Data

`r10_runs.csv` (960 rows: child_recovered + absorbed + full activation
diagnostics — signed cos, conditional fire rates, composite presence, φ
secondary), weights per arm, `r10.log`. Scoring: `analysis/analyze_round10.py`
(frozen at lock).
