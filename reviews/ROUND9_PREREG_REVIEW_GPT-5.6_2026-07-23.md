# Round-9 prereg external review — GPT-5.6 Sol (High), chatgpt.com, 2026-07-23

*Reviewed documents: `theory/gating_corrected_rho.md` + `notes/prereg-gating-corrected-rho.md`
(draft state after the Gemini-2.5-Pro review's changes were applied — the
8-seed→24-seed change and the Gemini required fixes were NOT yet in the copy
GPT reviewed; overlapping findings are noted in the response doc). Verbatim
transcript below (math notation as rendered by the page-text extraction;
`[ ... ]` blocks are the reviewer's LaTeX).*

---

## Verdict

MAJOR REVISION — do not lock the documents in their current form.

The core idea is promising. The exclusive-ratio odds formula is correct under
its intended idealization, and a dominance partition can recover ρ exactly in
a background-free model with exact gating and perfect tokenwise dominance.

The drafts, however, currently promote that ideal result into a broader
consistency claim that has not been established. The principal problems are:

1. The background pattern equations silently assume conditional independence.
2. The claimed consistency of ρ̂_D ignores background-active tokens.
3. Aggregate activation gaps do not establish the required tokenwise
   dominance property.
4. Comparing raw activations across latents requires a common per-latent
   scale, not merely cross-harness rank robustness.
5. The preregistration says "detected pairs" but uses oracle pair location
   and oracle orientation.
6. The "rarer latent" counting baseline does not estimate ρ when ρ>0.5, even
   under perfectly clean gating.
7. Exclusions can selectively remove difficult seeds without a
   minimum-inclusion rule.
8. Eight seeds per cell are insufficient to establish a cellwise MAE bound of
   0.05 with useful precision.
9. The pass/falsification bars are not derived from the theory's admitted
   error terms.

## 1. Algebraic audit

### 1.1 Pattern-cell equations: unstated background independence

Let q_ij = P(b_par=i, b_comp=j | B). Under exact a1=g0=1, the generally valid
equations are P10 = r_S(1−a0) + r_B·q10; P01 = r_J(1−g1) + r_B·q01;
P11 = r_J·g1 + r_S·a0 + r_B·q11; P00 = r_B·q00.

The draft instead substitutes q10 = b_p(1−b_c), q01 = b_c(1−b_p),
q11 = b_p·b_c, which assumes the two latent-fire indicators are conditionally
independent on background tokens. That assumption is neither stated nor
naturally guaranteed. SAE latents can cofire on background because of shared
residual structure, encoder correlations, common noise, or overlapping
decoder directions. b_p and b_c are marginal rates and do not determine any
of the four background pattern cells without q11.

Required correction: use q_ij in the theory. Conditional independence may be
presented as an optional simplification that must be checked empirically, not
built into the base equations.

A useful nuance: equal background marginals b_p = b_c imply q10 = q01 even
without independence (q10 = b_p − q11, q01 = b_c − q11). That is sufficient
for symmetric contamination of ρ̂_X, but not for the stronger product
formulas.

### 1.2 "Exactly one equation short" is true only in the idealized model

With no background firing, P00 identifies T = r_J + r_S; the two independent
active-pattern proportions then constrain (ρ, a0, g1), leaving a
one-dimensional deficit. With unknown background patterns, q10/q01/q11 add
unidentified quantities — the system is more severely underidentified unless
the background distribution is known, separately measured, or restricted.
"Any corrected estimator must import exactly one more constraint" must be
scoped to the exact-gating, background-free (or known-background) model.
C-dom is not simply "one more equation": it imports token-level activation
information and a strong separability assumption — a different observation
model.

### 1.3 Exclusive-ratio estimator: formula correct, scope overstated

The odds algebra and the ρ=0.1 / a0=0.16 / g1=0 worked example (ρ̂_X ≈
0.1168) are correct. Missing qualifications: no background contamination;
exact F1–F2 (not merely ≥0.998); 0<ρ<1; n01+n10>0. "Exact iff a0=g1" is only
generically true under those conditions; at a0=g1=1 both exclusive cells
vanish and the estimator is undefined. With arbitrary background the
starting point should be
ρ̂_X → [r_J(1−g1) + r_B·q01] / [r_J(1−g1) + r_S(1−a0) + r_B(q01+q10)].

### 1.4 Dominance estimator: ideal consistency valid, stated claim not

Define δ_J = P(a_comp ≤ a_par | J,11) (ties count as J-misclassification
under the registered tie rule) and δ_S = P(a_comp > a_par | S,11). Under
exact F1–F2, ignoring background:

  ρ̂_D → ρ(1−g1·δ_J) + (1−ρ)·a0·δ_S
  bias(ρ̂_D) = (1−ρ)·a0·δ_S − ρ·g1·δ_J.

Three consequences: (i) perfect tokenwise dominance ⇒ exact, regardless of
leak symmetry; (ii) average activation separation is insufficient — what
matters is the inversion probability within each oracle class's 11 cell;
(iii) J and S errors can cancel — a nearly unbiased ρ̂_D does not by itself
prove C-dom.

With background, let π_B = P(a_comp > a_par | B,11). Then

  ρ̂_D → [r_J(1−g1·δ_J) + r_S·a0·δ_S + r_B(q01 + q11·π_B)]
         / [r_J + r_S + r_B(1−q00)]

— a mixture of the parent-event estimate and the background assignment share
h_B = (q01 + q11·π_B)/(1−q00). Consistent for ρ only if background-active
mass vanishes, is removed, or happens to balance. F1–F2 + F4 alone are NOT
sufficient. The consistency sentence must be restricted to a background-free
theorem; the operational all-token estimator should be described as
potentially biased.

### 1.5 The evidence stated for F4 does not yet establish F4

Reported magnitudes are averages; a mean gap of 1.2 vs 0.15 does not prove
a_comp > a_par on nearly every J/11 token — a minority of severe inversions
can coexist with a large mean gap. Token-level inversion rates are required.
D1 is designed to obtain exactly those measurements: complete and commit D1,
determine whether tokenwise C-dom is supported, and only then lock a theory
note that describes C-dom as an empirical structural fact. Until then F4 is
a hypothesis motivated by aggregate magnitudes.

### 1.6 Background-contamination formula: missing factor and wrong scope

For ρ̂_X with q01=q10=q: ρ̂_X → (1−λ)·A/(A+C) + λ/2 with
λ = 2·r_B·q/(A+C+2·r_B·q) — the draft's stated weight omits the factor 2.
The analysis applies to ρ̂_X only; for ρ̂_D contamination pulls toward h_B
(must be measured), and for ρ̂_count the background limit is yet another
quantity. The (0.32+0.036)/(0.4+0.070) arithmetic is correct but the
attribution is not identified from one marginal excess rate; measure the
components directly. "Implies a larger noise-fire denominator" (Arm A σ=0.1)
is too strong — consistent-with, not implied.

### 1.7 "Bias at ρ=0.5 would falsify the picture" is overstated

Asymmetric q01≠q10, π_B≠1/2, systematic tie assignment, unequal inversion
masses, F1/F2 violations, per-latent scaling, and selection effects can all
bias ρ=0.5. Describe as evidence of an omitted asymmetry, not something the
model categorically cannot produce.

## 2. A major baseline error at ρ>0.5

In the clean-gating limit P(b_comp|parent)=ρ, P(b_par|parent)=1−ρ, so the
rarity-based estimator converges to **min(ρ, 1−ρ)**, not ρ. At ρ=0.7 it
estimates 0.3 under perfect gating. Consequences: count-estimator error
there is not "leak inflation"; P3 can be won trivially; oracle orientation
does not repair it; rarity-based auto-orientation cannot distinguish ρ from
1−ρ. Required: register the oracle-comp counting baseline
P(b_comp)/P(b_par ∨ b_comp) — which targets ρ under clean gating — as the
leak-bias comparator, and treat the deployed rarity-based count as an
operational baseline valid only under a registered ρ ≤ 0.5 restriction.

## 3. Raw activation comparison is not automatically scale-free

Immune to common global rescaling, not to separate per-latent rescaling
(activation ×c, decoder column ÷c leaves reconstruction unchanged). Raw
activation ordering is meaningful only if a common feature scale is enforced
— e.g. unit-norm decoder columns (verify on saved weights), or compare
scale-invariant reconstruction contributions a_k·‖d_k‖. Also θ=0.05 is an
absolute constant: say "no new harness-tuned constant beyond the detector's
fixed threshold," not "no constants."

## 4. Experimental-design problems

4.1 Claim–experiment mismatch: rename to oracle-located,
criterion-qualified absorbed pairs (or actually run the frozen detector);
round 9 does not test detection, FPs/FNs, rate-window effects, end-to-end
label-free operation, or automatic orientation.
4.2 Exclusion rules permit survivorship bias: freeze exact absorption
criteria, matching order, multiple-candidate handling, failed-run treatment;
NO exclusions based on F1/F2/dominance/estimator behavior; minimum
included-seed rule per cell; report formation probability on all attempts.
4.3 Development must finish before lock: D1/D2 committed → code + scoring
frozen → lock → confirmatory. "D-phase may fix implementation bugs" after
lock is a discretion channel.
4.4 Freeze evaluation details: eval-token count; train/eval independence
(shared activation corpus disclosed); nominal vs realized ρ as target;
injection RNG; which components vary per seed; tie representation; failed
runs; exact oracle matching algorithm.
4.5 Missing stress cells: real-harness ρ=0.1 (or narrow claim to
ρ∈[0.3,0.7]); synthetic σ=0.05 (transition regime where leakage already
emerges).

## 5. Estimator-definition edge cases

5.1 Zero denominators: freeze handling (report undefined, no pseudocounts).
5.2 Ties break exact orientation equivariance (swapped estimate = 1−ρ̂ only
with zero ties); equivariance is not robustness — 1−ρ is still wrong for a
directed estimand unless output is treated as the unordered pair.
5.3 Pooled inversion rates can hide failure: define P4 separately for J/11
and S/11, per seed, with minimum denominators; the theory-relevant quantity
is the weighted misclassification mass (1−ρ)a0·δ_S − ρ·g1·δ_J.
5.4 Rate-window/selection behavior absent for the descriptive
auto-orientation readout: specify rarity split, ties, independence from eval
set, scoring at ρ>0.5.

## 6. Assessment of registered predictions

P1/P2 can pass through cancellation (J/S inversions, background bias, F1/F2
errors, scaling artifacts): add an oracle-parent-only (B-excluded)
mechanism diagnostic alongside the operational endpoint. P3 invalid at
ρ>0.5 as written; predefine eligible cells theoretically, use the
oracle-comp baseline, require a meaningful margin or paired comparison.
P4 thresholds poorly connected to the near-deterministic mechanism (20%
inversion is already a strong rejection); use separate class rates,
seed-level aggregation, minimum denominators, weighted mass. P5's
symmetry-window conditioning selects on confirmatory outcomes; condition
explicitly on background-exclusive contamination ≈ symmetric, or predict
from measured (a0, g1, q01, q10).

## 7. Statistical power and falsification bars

7.1 Eight seeds: t7 95% half-width ≈ 0.042 at seed-level SD 0.05 — an
observed MAE 0.05 is compatible with materially larger population MAE;
bootstrap reps do not add observations. Increase seeds or downgrade claim.
7.2 Bars not calibrated to theory: derive an error budget from F1/F2 miss
rates, class-specific weighted inversion masses, background mass and h_B,
finite-sample error, training-seed variation; two-endpoint design
(mechanism endpoint on oracle J∪S tokens with a materially tighter
falsification bar than 0.15; operational endpoint on all tokens).
7.3 "Every cell" multiplicity asymmetry: acknowledge, or predeclare a
primary aggregate over nontrivial cells with per-cell heterogeneity checks.

## 8. Required changes before lock (13)

1. Narrow the estimand/claim to oracle-located, criterion-qualified pairs.
2. Correct background algebra to q_ij; independence = tested option.
3. Restrict the identifiability statement to the idealized model.
4. Add the formal error decomposition for ρ̂_D (δ_J, δ_S, h_B); remove the
   unconditional consistency claim.
5. Resolve latent-scale comparability (document/assert decoder norms or use
   a_k·‖d_k‖).
6. Complete D1 before lock; F4 is a hypothesis until token-level inversion
   rates are committed.
7. Repair the counting baseline (oracle-comp for P3; rarity baseline scoped
   to ρ≤0.5).
8. Freeze all exclusions and oracle matching + minimum included-seed rule.
9. Freeze edge-case behavior.
10. Redefine P4 (class-separated, seed-level, minimum denominators, weighted
    mass).
11. Recalibrate the bars from an explicit error budget; add
    mechanism/operational endpoints.
12. Address power (raise primary seed counts or downgrade to pilot).
13. Freeze evaluation details and the development→lock sequence; add
    real ρ=0.1 and synthetic σ=0.05 or narrow the claim.

## 9. Optional suggestions

Signed bias + RMSE + all seed-level estimates; four estimator variants
(all-token, oracle-B-excluded, oracle-orientation, auto-orientation); report
empirical h_B by seed and cell; frozen descriptive θ-sensitivity sweep;
report unordered {ρ̂, 1−ρ̂} for auto-orientation; small analytic Monte
Carlo check of the corrected formulas before the GPU run.

## Final judgment

Real failure mode identified; dominance rule is a credible correction;
exclusive-ratio derivation sound in its narrow model. The documents
overstate what has been proved and preregister an experiment that could pass
without validating the advertised claim. Most consequential: the invalid
rarity baseline above ρ=0.5, the unmodeled background contribution to ρ̂_D,
unverified comparability of raw latent activations, and exclusion/power
weaknesses. **Verdict: MAJOR REVISION.** Lock only after the required
changes are incorporated.
