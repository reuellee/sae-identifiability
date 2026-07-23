# Adversarial pre-lock review: Round 10 TopK absorption

## Verdict: **MAJOR REVISION — DO NOT LOCK**

The two-atom calculations are sound for the stated oracle-coded, two-atom dictionary model. The preregistered experiment, however, uses an overcomplete `m=16` learned SAE. In that setting, a simple three-atom solution destroys the claimed `ε*=2q` tradeoff. Consequently, Arm M does not currently instantiate the theorem it is supposed to confirm.

There are also serious scoring loopholes: the primary statistic can be satisfied by an unused decoder atom, undefined outcomes are selectively excluded, and the ε=0.05 interpretation clause permits a preregistered empirical failure to be linguistically downgraded.

## Blocking issue 1: the theorem assumes two atoms, but Arm M has sixteen

The theory optimizes over dictionaries with exactly two pair atoms: faithful `{v_p,v_c}` versus absorbed `{v_p,d_comp}`, and the verifier’s global scan is explicitly limited to two-atom dictionaries. Arm M instead uses `m=16` with only six low-rate background features.

With three available atoms, define

[
U_3={v_p,\ v_c,\ d_{\rm comp}},\qquad
d_{\rm comp}=(v_p+v_c)/\sqrt2.
]

Under `k=1`:

* parent-solo is reconstructed exactly by (v_p);
* child-solo is reconstructed exactly by (v_c);
* joint is reconstructed exactly by (\sqrt2,d_{\rm comp}).

Thus

[
L_{k=1}(U_3,x)=0
]

for every modeled event and every value of (p,q,\epsilon). This strictly dominates both two-atom candidates whenever their loss is positive. There is then **no global `ε*=2q` crossover at all**.

I numerically checked this counterexample using the verifier’s own nonnegative sparse-coding formulation; all three event losses are zero to numerical precision.

This is fatal to T1. The one-slot activation budget does not force absorption when the dictionary has enough separate atoms for parent, child, and joint prototypes. The relevant capacity is not only per-token `k`; dictionary width or effective pair-atom allocation also matters.

### Required correction

Use one of these designs before locking:

1. **Exact-theorem arm:** `n_bg=0`, `m=2`, with precisely two trainable decoder atoms. This tests the two-atom crossover and `k=1` versus `k=2` result.

2. **General overcomplete arm:** retain `m=16`, but explicitly treat it as a test of whether SGD nevertheless finds absorption despite the existence of a zero-loss three-atom solution. It cannot be presented as confirmation of the current global-optimum theory.

The strongest design would include both and sweep effective pair width through at least 2 and 3. A result that changes sharply between two and three available pair atoms would expose the currently omitted dictionary-width mechanism.

## Blocking issue 2: the theory uses oracle sparse coding, not the trained TopK encoder

The theoretical loss minimizes over all atom subsets and all nonnegative coefficients:

[
\min_{|S|\leq\kappa}\min_{f\geq0}|x-U_Sf|^2.
]

That is an oracle nonnegative sparse-coding objective. A trained TopK SAE instead selects atoms using learned encoder preactivations and uses the corresponding encoded magnitudes. The oracle objective is a lower bound on what the learned encoder can achieve, not an exact description of its loss landscape.

The note acknowledges that encoder and gating effects may shift the transition, but this is more than a possible numerical shift. The subset-selection mechanism and coefficient parameterization can change which dictionaries are reachable and stationary.

### Required correction

Rename the theoretical object precisely, for example:

> “Two-atom oracle-coded nonnegative (k)-sparse dictionary model.”

Then register diagnostics connecting the trained encoder to that model:

* frequency with which the intended pair atoms actually occupy the TopK slots;
* conditional gate occupancy on joint, parent-solo, and child-solo events;
* achieved reconstruction loss versus oracle NNLS loss for the learned decoder;
* encoder-selected atom versus oracle-best atom agreement.

Without these, a failed capacity-collapse cell cannot be cleanly diagnosed as SGD reachability rather than encoder/gating mismatch.

## Blocking issue 3: `phi_child` can pass using a dead or unused atom

`phi_child` is selected solely from decoder geometry. The preregistration does not require the selected latent to fire on child-solo examples, contribute to reconstruction, or even be alive. An unused decoder column near (v_c) can therefore yield `phi_child ≥ 75°` while the operational representation still absorbs the child.

This is especially dangerous with `m=16`, where spare or dead columns are plausible. The measurement can confirm geometric availability without demonstrating functional child recovery.

The auxiliary `cos_child` also uses absolute cosine. Under nonnegative codes, a column near (-v_c) is not equivalent to (v_c), despite having absolute cosine near one.

### Required correction

Make the primary metric activation-aware. A child-recovered latent should satisfy all of:

* positive signed alignment with (v_c);
* nontrivial firing or TopK-selection rate on child-solo events;
* measurable reconstruction contribution on child-solo events;
* preferably child selectivity relative to parent-solo events.

Freeze explicit thresholds before running. Report geometric `phi_child` as a secondary diagnostic, not as the sole primary endpoint.

## Blocking issue 4: outcome-dependent missingness can manufacture a pass

Up to 8 of 24 seeds may have undefined `phi_child` and be discarded while the median is computed over the remaining seeds. Undefinedness means that no qualifying in-plane child-side latent was found—precisely an outcome likely to correlate with failed recovery. Excluding those seeds is therefore not neutral missing-data handling.

A cell with 16 favorable defined seeds and 8 failed or pathological undefined seeds can pass as faithful. The “unscoreable below 16” rule only protects against more extreme missingness.

### Required correction

For primary directional tests, use conservative scoring:

* when testing faithful recovery, impute undefined seeds as non-faithful;
* when testing absorption, impute undefined seeds in the direction least favorable to the claim; or
* replace the windowed statistic with an always-defined activation-aware score.

At minimum, the defined-seed rate must itself have a pass bar. Merely “disclosing” undefined seeds does not close the selection loophole.

## Blocking issue 5: the ε=0.05 clause changes the empirical verdict

T2 predicts faithful recovery for **every** listed nonzero ε. It then says that a failure confined to ε=0.05 will be called an “SGD-reachability limit,” while only larger or repeated failures count as genuine falsification of the SGD-level claim.

It is legitimate to say such a result does not refute the mathematical global-optimum statement. It is not legitimate to exempt it from falsifying the registered trained-SAE prediction.

### Required wording

A localized ε=0.05 failure must be reported as:

> **T2 FALSIFIED**, localized to the smallest ε and diagnosed as evidence of an SGD/encoder reachability limit; the two-atom global-optimum proposition is unaffected.

Diagnosis may qualify scientific interpretation. It must not alter the registered verdict.

## Blocking issue 6: T2 can become “vacuous” without blocking confirmation

The matched `k=1` cell is only a reported precondition. If it does not absorb, the collapse comparison is declared vacuous. But demonstrating absorption under tight capacity is part of the headline claim. A run in which `k=1` is already faithful cannot confirm a capacity-driven collapse.

### Required correction

Specify that any vacuous primary cell prevents a T2 confirmation. The result may be “inconclusive/not instantiated,” but not a pass. The overall headline also cannot be confirmed unless tight-budget absorption is demonstrated in predesignated cells.

## Major theory corrections

### “Any dictionary containing the composite is strictly worse” is false

With more than two atoms, `{v_p,v_c,d_comp}` has zero loss at both `k=1` and `k=2`, while containing the composite direction. At `k=2`, the composite may remain as a redundant atom at no reconstruction cost. The theory’s statement is valid only for the specific two-atom absorbed dictionary `{v_p,d_comp}`, not for arbitrary dictionaries containing a composite.

This distinction matters directly because the experiment uses `m=16`. Define absorption as **absence of a functional child-specific representation**, not mere presence of a composite atom. Register how runs containing both child and composite latents will be classified.

### The faithful frame is not the unique `k=2` optimum

Any two rays whose nonnegative cone contains both (v_p) and (v_c) can reconstruct all three events exactly. For example, atoms at approximately (-30^\circ) and (120^\circ) have zero event loss under `k=2`, despite neither being the exact faithful frame.

The note partly acknowledges “wide-cone neighbours,” but then repeatedly equates zero-loss optimality with faithful child recovery. The theorem establishes existence of zero-loss non-absorbed solutions, not unique recovery of (v_c).

### ε=0 is non-identifiability, not impossibility of recovery

When ε=0, the distribution still contains parent-solo (v_p) and joint (v_p+v_c), so an algorithm using the assumed support structure can obtain (v_c=(v_p+v_c)-v_p). What fails is unique identification by the unconstrained reconstruction objective, not information-theoretic recoverability “under any architecture.” Also, the faithful dictionary remains a global optimum at ε=0; it is not impossible for training to recover it.

Replace “cannot recover” with “is not uniquely identified by this reconstruction objective without additional inductive assumptions.”

## Verifier weaknesses

The verifier exits successfully, but several advertised conclusions are not actually checked:

* “all unit-norm dictionaries” is a 145-point angular grid, not an exhaustive or analytic search;
* the listed intermediate optimum angles are printed but mostly not asserted;
* the `k=2` global-minimum check is nearly tautological because the grid contains the faithful frame with known zero loss;
* it never tests `m=3`, which immediately exposes the zero-loss `k=1` counterexample;
* it verifies an oracle decoder objective, not the learned TopK encoder.

Calling M0 a “proof” is therefore inappropriate. The elementary two-atom inequalities can be presented as an analytic proposition, with the code described as numerical regression tests.

Add mandatory tests for:

1. the three-atom zero-loss counterexample;
2. redundant child-plus-composite solutions;
3. wide-cone `k=2` optima;
4. exact assertions for every angle/table value claimed in the note;
5. the distinction between exactly two atoms and arbitrary `m`.

## Remaining preregistration ambiguities

### T1 is not fully falsifiable on the current grid

For `q=0.2`, the falsification upper bound is `1.3·2q = 0.52`, but the ε grid ends at 0.40. A right-censored `ε_mid > 0.40` cannot distinguish a transition at 0.45 from one above 0.52. Extend that row to at least ε=0.60 or register the result as inconclusive rather than non-falsified.

### Pass, falsify, and gray-zone outcomes need explicit names

T2 passes at `≥75°` and falsifies at `≤55°`; gaps pass at `≥15°` and falsify at `≤5°`. T1 and T3 have similar guard regions. State explicitly that intermediate values are **inconclusive**, and define how one inconclusive cell affects each claim and the overall verdict.

### “Rises with k” is not scored

T3 says φ rises with k, but its bars only test endpoints and whether any point reaches the faithful band. A strongly nonmonotonic curve could pass. Either register a monotonicity statistic—such as a rank correlation or bounded downward-step rule—or remove monotonic rise from the confirmatory wording.

### The λ arm does not test λ-independence

Adding `lam=0.1` adds an L1 term to the TopK objective. Dependence on that term would not contradict a theory about pure TopK with no L1 term. Describe this arm as robustness to an auxiliary L1 penalty, not as a test of λ-independence.

## Minimum acceptable pre-lock revision

1. Add a genuinely exact `m=2`, no-background primary arm—or rewrite the theory and claim to include dictionary width.
2. Reclassify the current `m=16` arm as an overcomplete SGD-behavior test.
3. Replace geometry-only scoring with an activation-aware child-recovery metric.
4. Eliminate outcome-dependent exclusion from primary passes.
5. Preserve `FALSIFIED` for any registered T2 bar failure, including isolated ε=0.05 failures.
6. Make vacuous primary cells block confirmation.
7. Extend the `q=0.2` ε grid beyond 0.52.
8. Correct the composite, uniqueness, and ε=0 overclaims.
9. Add the three-atom counterexample to the verifier and sharply scope every theorem to its atom-count assumptions.
10. Freeze and commit the actual scorer before lock; the prose alone is insufficient to establish that the stated rules are executable.

Until these changes are made, a confirmatory run could produce almost any outcome and still fail to answer the stated theoretical question.

