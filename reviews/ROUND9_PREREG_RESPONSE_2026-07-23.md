# Round-9 prereg: response to pre-lock external reviews (2026-07-23)

Reviews: `ROUND9_PREREG_REVIEW_GEMINI-2.5-PRO_2026-07-23.md` (MINOR
REVISION) and `ROUND9_PREREG_REVIEW_GPT-5.6_2026-07-23.md` (MAJOR
REVISION). GPT reviewed the pre-Gemini-fix draft, so some findings overlap;
dispositions below cover the union. Revised documents:
`theory/gating_corrected_rho.md` + `notes/prereg-gating-corrected-rho.md`
(this commit).

## Gemini required changes

1. State the conditional-independence assumption → **applied, sharpened**:
   with F1–F2 exact the J/S class terms need no independence at all (theory
   §3); only background cells did — and those now use general q_ij (see
   GPT #2), so independence is nowhere assumed.
2. Analyze B∧11 contamination of ρ̂_D → **applied** (theory §4–§5: h_B
   share; π_B, q₁₁-B, B∧11 mass are registered diagnostics).
3. Deterministic ρ̂_count → **applied** (lower-rate rule, tie → comp), and
   superseded in role by GPT #7 (ρ̂_C is the comparator).
4. Add ρ = 0.1 RC cell → **applied** (Q = 0.04/P0 = 0.36).
5. Register per-cell exclusion rates → **applied**, plus a
   minimum-inclusion rule (≥ 16/24) per GPT #8.

Adopted suggestions: P4 falsify bar 0.20 → 0.10 (then reworked per GPT
#10); θ-sensitivity descriptive readout; tie-rule arbitrariness noted;
seeds 8 → 24/cell (also standing-constraint compliance; addresses GPT #12).

## GPT-5.6 required changes (§8 numbering)

1. Narrow estimand to oracle-located, criterion-qualified pairs →
   **applied** throughout both documents; non-tested items listed in the
   prereg scope.
2. q_ij background algebra, independence as tested option → **applied**
   (theory §2–§3; q_ij measured in D1 + confirmatory diagnostics).
3. Identifiability deficit scoped to the idealized model → **applied**
   (theory §3).
4. ρ̂_D error decomposition (δ_J, δ_S, h_B); consistency = background-free
   theorem → **applied** (theory §4); operational estimator labeled
   potentially biased; M/O endpoint split introduced.
5. Latent-scale comparability → **applied** as precondition S1: both
   harnesses renormalize decoder columns to unit norm every step
   (`BSAE.renorm()`; the real-data loop's per-step normalize), so raw
   activation = reconstruction contribution; asserted in eval code
   (tolerance 1e-3) rather than assumed.
6. D1 before lock; F4 = hypothesis → **applied** (theory §2 F4 relabeled;
   prereg pre-lock phase: M0 + D1 + D2 complete and committed before lock,
   with an explicit F4-failure escape that revises rather than registers).
7. Counting-baseline repair → **applied**: ρ̂_C = P(b_comp)/P(either) is
   the registered leak-bias comparator; rarity-based ρ̂_count kept as the
   descriptive incumbent with its min(ρ, 1−ρ) target and ρ ≤ 0.5 scope
   stated; P3 uses ρ̂_C with a −0.05 margin and lock-time predefined cells.
8. Frozen exclusions/matching + minimum inclusion → **applied**
   (qualification formulas, argmax tie rule, no-rerun rule, the
   no-exclusion-on-diagnostics clause, ≥ 16/24 scoreability).
9. Frozen edge cases → **applied** (nan, never smoothed; implemented in
   `analysis/rho_estimators_lib.py`, verified in M0).
10. P4 redefined → **applied** (per-seed δ_J and δ_S, ≥ 100-token class-11
    denominators, cell medians, weighted misclassification mass reported;
    bars ≤ 0.05 / > 0.10 per cell median).
11. Bars from an error budget + endpoint split → **applied structurally**
    (M and O endpoints registered; [AT-LOCK] bars to be filled from D1's
    measured budget in the lock commit — targets 0.03/0.07 mechanism,
    0.05/0.15 operational pending that budget).
12. Power → **applied** via 24 seeds/cell (all cells, both harnesses).
13. Eval details + dev→lock sequence + stress cells → **applied** (frozen
    eval-details paragraph; explicit M0/D1/D2-before-lock sequence;
    σ = 0.05 SC cells and ρ = 0.1 RC cell added).

Adopted optional suggestions: signed bias/MAE/RMSE + all seed-level
estimates; four estimator variants via M/O endpoints + auto-orientation
descriptive (incl. unordered-pair reporting); h_B per run; θ-sweep;
analytic Monte Carlo check (`analysis/mc_check_rho_formulas.py` — M0, all
32 checks pass, committed with this response).

Not adopted: none of the required items. The multiplicity suggestion
(aggregate primary endpoint) was adopted as an acknowledgment of the
severe every-cell/any-cell stance rather than a change of endpoint —
per-cell verdicts remain primary, matching project precedent.
