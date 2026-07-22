# The no-go theorem in general dimension: derivation, scope, and a new critical ratio

*Companion to §7.1b of report.md. Every numbered claim below is computationally verified (sympy/exact scans, not a proof assistant) by
`general_no_go_check.py` (checks A–E, all passing); the verification uses a code path
independent of the earlier event-loss enumeration, and reproduces its numbers.*

## Setup

Ambient dimension d. Generative model: two orthonormal feature directions a_p, a_c
(the pair plane Π) plus arbitrary background features orthogonal to Π. Events in Π:
joint x_Π = a_p + a_c (prob q), parent-solo a_p (prob p₀), child-solo a_c (prob ε),
none (else); background activity is arbitrary but supported orthogonal to Π.

SAE: dictionary D = [d₁ … d_m], unit-norm columns, m ≤ d, nonnegative codes,
population objective

L_β(D) = E_x[ min_{f≥0} ‖x − Df‖² + λ‖f‖₁ ] + β·P(D),
P(D) = Σ_{i<j} g(⟨d_i, d_j⟩),

for **any** g with g(0) = min g (Frobenius: g(t)=t²; abs-cos: g(t)=|t|; OrtSAE-style
penalties are handled a fortiori below). Let O = {D : columns orthonormal} (nonempty
since m ≤ d).

## Lemma 1 (penalty constancy and its consequences)

**(i)** P is constant on O, equal to its global minimum (m choose 2)·g(0).
*Proof:* every pairwise inner product is 0 on O, and g(0) minimizes g. ∎

**(ii)** For all D_A, D_B ∈ O and all β ≥ 0:
L_β(D_A) − L_β(D_B) = L₀(D_A) − L₀(D_B).
The penalty — any penalty of this class, at any strength — is invisible to
comparisons within O.

**(iii)** If additionally g(t) > g(0) for t ≠ 0, then as β → ∞ every limit point of
global minimizers of L_β lies in O, and minimizes L₀ within O. *Proof sketch:* the
domain (product of unit spheres) is compact and L₀, P are continuous; a limit point
D̄ of minimizers D_β must satisfy P(D̄) = min P (else, comparing against any D ∈ O,
β(P(D_β) − minP) → ∞ while the L₀ terms stay bounded), and P(D̄) = min P forces
pairwise orthogonality; optimality of L₀ within O follows from (ii). ∎

**(iv) A fortiori for chunked penalties.** OrtSAE-style penalties apply g only to
within-chunk pairs, so their constancy class strictly contains O; every consequence
below holds unchanged (and the chunking adds a second, cruder failure mode: the
pair of interest is co-penalized only when it lands in one chunk — probability
≈ (chunk size − 1)/(m − 1), ~23% in our GPU configuration — verified empirically
in the round-4 OrtSAE-style experiment).

## Lemma 2 (exact closed form on O — the workhorse)

For D ∈ O the nonnegative lasso decouples per coordinate, and

**L₀(D) = E‖x‖² − Σᵢ E[ (⟨dᵢ, x⟩ − λ/2)₊² ].**

*Proof:* with orthonormal columns, ‖x − Df‖² = ‖x‖² − 2Σᵢ fᵢ⟨dᵢ,x⟩ + Σᵢ fᵢ², so the
objective separates; minimizing fᵢ² − 2fᵢc + λfᵢ over fᵢ ≥ 0 gives fᵢ* = (c − λ/2)₊
and optimal value −(c − λ/2)₊². ∎  (Check A: matches direct active-set enumeration
on random orthonormal dictionaries to 2×10⁻¹⁵.)

**The one-line mechanism of absorption under orthogonality.** The gain function
(c − λ/2)₊² is *superadditive under energy concentration*: an event of norm r
reconstructed by ONE aligned column gains (r − λ/2)², while split across two
columns at 45° it gains 2(r/√2 − λ/2)² — smaller by (√2 − 1)λr − λ²/4 > 0 (leading order in λ).
The soft-threshold tax λ/2 is paid **once per active coordinate**. Absorption is the
objective's preference for paying it once on frequent co-occurrence events; no
function of decoder angles can see this, because it lives in the codes.

## Lemma 3 (reduction to the in-plane frame family)

Consider dictionaries in O whose columns split into: two columns spanning Π, and
m − 2 columns orthogonal to Π (class O₂ — contains both the faithful and the
anti-rotated candidates; rotating the in-plane pair preserves global orthonormality).
For D ∈ O₂, Lemma 2 splits L₀ into (in-plane part) + (background part), and the
background part is IDENTICAL across O₂ members sharing the same background columns:
background events project to 0 on the in-plane columns and identically on the rest.
Hence the competition within O₂ reduces exactly to the one-parameter family
T(θ) = in-plane loss of the frame {u(θ), u(θ+90°)}, computable in closed form from
Lemma 2.

**Scope, stated honestly.** (a) Configurations with *partially* in-plane columns
(mixing Π with background directions) are not covered by the reduction; check C
probes them — 3000-step random-rotation searches over the full orthogonal group
(exact finite-atom model, both sides of the boundary) never beat the O₂ optimum.
Not a proof; a systematic search that found nothing. (b) m > d (overcomplete SAEs):
O is empty and Lemma 1(iii) instead forces minimal-coherence frames; the reduction
applies when non-pair latents carry no in-plane mass at the optimum, which is what
trained SAEs empirically do in our runs (in-plane inventory shows exactly two
in-plane latents). The general overcomplete case is open.

## Theorem (general no-go, with its domain of validity)

Within O₂ (any d, any m ≤ d, any background structure, any penalty of the stated
class, any β including ∞): the penalized objective prefers an anti-rotated absorbed
frame over every faithful frame **iff ε < ε\*\*_∞(λ, q, p₀)**, where ε\*\*_∞ is the
crossover of T(θ) between the anti basin (θ ≈ −45°) and the faithful basin (θ ≈ 0°).

**Closed-form anchor (pure strategies θ = −45° vs θ = 0°):**

ε₀(λ, q, p₀) = λ·[ (2−√2)q − (√2−1)p₀ + λ(p₀−q)/4 ] / [ 1/2 − (2−√2)λ/2 ]

with the basin-optimized ε\*\*_∞ ≈ 2% above ε₀ (check B: at λ=q=p₀=0.2,
ε₀ = 0.0155, ε\*\*_∞ = 0.0159 — reproducing the §7.1b value through an independent
implementation). ε\*\*_∞ → 0 as λ → 0 (check D), and ε\*\*_∞ ∝ λ to leading order.

## New corollary: the critical occurrence ratio p₀/q = √2

The general derivation exposes a dependence the 2D pair-comparison hid: setting
ε₀ = 0 gives a **critical parent-solo rate**

**p₀\*(λ, q) = q·[(2−√2) − λ/4] / [(√2−1) − λ/4], with p₀\*/q → √2 as λ → 0.**

- **p₀ < p₀\*** (co-occurrence-dominated hierarchies): the no-go bites in full —
  ε\*\*_∞ > 0, no penalty of any form or strength rescues children below it, the
  anti-rotated evasion wins. (Our GPU experiments used p₀ = q, ratio 1 < √2:
  this regime.)
- **p₀ > p₀\***: the anti frame's Achilles heel — it reconstructs parent-solo
  events 2-sparsely, paying the λ/2 tax twice — becomes decisive, and the faithful
  frame wins the orthogonal competition **even at ε = 0**. Finite-β spot check
  (λ=0.2, q=0.2, p₀=0.35, ε=0.002): the global optimum flips to faithful already
  at β = β\* and sharpens toward {0°, 90°} as β grows, while the p₀=0.2 control
  stays anti-rotated at β = 64β\*. In this regime coherence penalties genuinely
  work — including at the ε = 0 wall, where they resolve Theorem 1's ontological
  tie in the faithful direction.

Verified numerically across p₀ (check E: ε\*\*_∞ vanishes between p₀ = 0.29 and
0.31 at λ = 0.2, q = 0.2; formula predicts p₀\* = 0.294).

**Interpretation and a falsifiable prediction.** Whether decoder-orthogonality
penalties can fix absorption is not a universal yes or no — it is governed by the
occurrence statistics of the hierarchy: penalties fail exactly for hierarchies
whose parent rarely appears without the child (p₀ < √2·q·(1+O(λ))) and work
otherwise. Pre-registered GPU prediction: at p₀ = 0.35, q = 0.2, a Frobenius
penalty at β ≥ β\* should reliably produce faithful children even at ε ~ 10⁻³,
in direct contrast to the p₀ = 0.2 runs of round 3. This also sharpens the
practical reading of §7.1b: the earlier blanket "never eliminates it" statement
is correct for p₀ ≤ √2 q (which includes all GPU configurations reported) and is
now annotated with its domain.

## Relation to the event-weighted remedy (§12)

The remedy result is unchanged and complementary: inverse-density weighting
eliminates the transition for all p₀ (its boundary condition is independent of
both ε and q), needs no orthogonality constraint at all, and removes the absorbed
basin rather than merely re-ranking optima.

## Verification suite summary (general_no_go_check.py)

| check | claim | result |
|---|---|---|
| A | Lemma 2 closed form vs direct enumeration | max err 1.8×10⁻¹⁵, PASS |
| B | boundary via independent path | 0.0159 = prior value, PASS |
| C | no mixed plane/background config beats O₂ | gap 0 both sides, PASS |
| D | ε\*\*_∞(λ) → 0 as λ → 0 | monotone, PASS |
| E | critical ratio p₀\* ≈ √2·q(1+O(λ)) | vanishes at 0.29–0.31 vs predicted 0.294, PASS |
| — | finite-β flip in the p₀ > p₀\* regime | faithful from β = β\* at p₀=0.35, ε=0.002 |

## Round-5 GPU test of the corollary (scored post-hoc, predictions above unchanged)

96 batched runs (`experiments/sae_round5.py`, `results/round5/`): p₀ ∈ {0.20, 0.35} ×
β ∈ {0, 1, 4}β\* × ε ∈ {0.002, 0.01} × 8 seeds.

- **P1 CONFIRMED**: p₀ = 0.35, β ≥ β\*: majority faithful at every cell, including
  5/8 (φ_med 72°) at the hardest point (β\*, ε = 0.002); 8/8 at (4β\*, 0.01).
- **P2 CONFIRMED**: p₀ = 0.35, β = 0: 0/8 faithful (φ ≈ 46°) — the rescue is the
  penalty's, not the occurrence statistics' alone.
- **P3 PARTIAL**: the p₀ = 0.20 control stays absorbed at the median at β\*
  (φ_med 45.6° vs 72.3° at p₀ = 0.35 — the regime contrast in the predicted
  direction), but at 4β\* more seeds escape to faithful (5/8 at ε = 0.002) than
  the equilibrium theory allows, replicating round 3's high-β anomaly. This is a
  genuine dynamics-vs-objective gap: at strong penalties SGD's basin selection
  favors the faithful frame more than global optimality predicts. The corollary's
  equilibrium claim stands (verified exactly); its translation to SGD basin
  fractions is quantitative only at moderate β.
