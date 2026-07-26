# Response to `GEMINI_round14_2026-07-26.md`

Adjudication of the adversarial review of the two unpushed round-14 commits
(`d2f32fa`, `f798687`). Reviews in this project are never auto-adopted; each finding
is accepted, rejected, or scoped here, with the evidence that decided it.

The review's verdict was **DO NOT PUSH**, on two findings it marked [BLOCKING]. One of
those is quantitatively wrong and is rejected with evidence. The other — marked only
[FIX-BEFORE-PUSH] by the reviewer — is the most valuable thing in the review, is
accepted, and is the reason the summary needed amending before publication rather
than after.

**Standing constraint on every item below:** `analysis/round14_carrier.py` is the
frozen scorer that produced the registered results. It is not edited in response to
this review. Where a finding identifies something that should be done differently,
the fix lands in a successor, not retroactively in the instrument.

---

## Rejected

### F3.1 / F5.1 [BLOCKING → REJECTED on magnitude; mechanism real, effect nil]

**Claim.** `c = F[rows, idx] * a[idx]` is 0 for inactive latents and negative for
active latents with negative alignment, so on any trial where *every* active latent is
negatively aligned, `argmax` returns the first inactive index. The reviewer asserts
this is "highly common" for a random direction (~50% of alignments negative), that it
"artificially concentrates the null carrier on a single inactive feature, inflating
`share_null` to the massive 34.0%", and that P2's $-0.199$ difference "is entirely a
mathematical artifact of this tie-break bug".

**The mechanism is real.** `np.argmax` does return the first maximal index, and an
inactive latent scoring exactly 0 does beat a negative score. The code would be better
with an explicit active-only mask, and the successor will have one.

**The magnitude claim is false**, three ways:

1. *Combinatorially.* The fall-through requires **all** active non-family latents to be
   negatively aligned simultaneously. These SAEs run at matched $L_0 = 32$, and on an
   absorbed trial the family is silent by definition, so ~32 non-family latents are
   active. At the reviewer's own ~50% figure that is ~$2^{-32}$ per trial, not "highly
   common".
2. *From the frozen results.* If the fall-through drove the letter-direction statistic,
   the modal carrier $\kappa$ would pile up at the lowest non-family indices. Across
   the 180 scored cells, $\kappa$ has median index 5246 (**44.7% of the way through the
   dictionary**), minimum 27, **178 distinct values in 180 cells**, and only 2 cells
   below index 100. That is the opposite of the predicted signature.
3. *By direct measurement.* `analysis/round14_validity_selftest.py` builds worlds with
   the same sparsity structure and measures the fraction of trials whose selected
   carrier does not fire: **0.0000** for the letter direction and **0.0000** for the
   random-direction null, in both the carrier and diffuse worlds. D1 measures the same
   quantity on the real weights (reported in `results_round14_validity.txt`).

**Consequence for the verdict.** The reviewer's stated ground for DO NOT PUSH does not
hold. It was still worth the check: the reasoning was specific and falsifiable, which
is what a useful review looks like even when it is wrong.

---

## Accepted

### F4.1 [reviewer: FIX-BEFORE-PUSH → ACCEPTED; the most important finding in the review]

**Claim.** The negative conclusion cannot distinguish representational loss from
*compositional* absorption, because `conc_A` measures the share held by the **global
modal** carrier $\kappa$, averaged over all absorbed trials. Since $\kappa$ is the top
contributor on only 14.1% of trials, its mean share is small by construction. If each
trial's letter mass is picked up by a *different* composite latent, every trial is
individually concentrated and P4 still reports a tiny number.

**Accepted in full.** This is correct, it is a real gap between what the statistic
computes and what the summary concluded from it, and I did not see it myself. Reading
`_share_of(F[rows_A], align, kappa)` in the frozen scorer confirms the fixed global
$\kappa$ is applied to every row of $A$, while the control comparator uses the single
best *family* latent on $C$ — the two sides are not the same estimator.

The claim "absorbed trials are diffuse" therefore does not follow from P4 as computed.
D2 in `analysis/round14_validity.py` recomputes concentration **per trial**, using each
trial's own top non-family latent, which is the statistic the claim actually requires.
The instrument was validated against synthetic ground truth *before* its real output
was read: it separates a carrier world from a diffuse world by 0.578 (0.874 vs 0.296).

### F2.1, F2.2 [ACCEPTED — confirms the call already made]

The review independently reaches the same diagnosis of P1's selection-on-outcome flaw,
and agrees that leaving the registered verdict as CONFIRMED while annotating it as
uninformative is the correct prereg practice rather than a violation. No change; this
is what `f798687` already does.

### F2.3 [ACCEPTED — an improvement on my own proposed fix]

The summary proposed selecting $\kappa$ on held-out or control trials and comparing on
disjoint $A$ trials. The reviewer notes selecting on $C$ risks picking latents
irrelevant to $A$, and that the standard fix is **sample splitting within $A$** —
choose the modal carrier on half of $A$, evaluate on the disjoint half against $C$.
That is better and is adopted as the successor's design.

### F1.1, F1.2, F3.2, F3.3 [ACCEPTED, no action]

Honesty, tone, the per-SAE-then-bootstrap collapse, and the handling of the
$|A| \ge 20$ power floor are endorsed. The bootstrap treats the SAE as the independent
unit, which is the point of the fix made at lock time.

### F5.2 [ACCEPTED, no action — frozen instrument]

The `if len(rows_C)` guard is indeed unreachable, since the caller already requires
`ctrl.sum() > 0`. Harmless dead code in the frozen scorer; not edited, for the reason
stated at the top. Noted for the successor.

---

## A finding the review did not make

Checking F3.1 surfaced a separate asymmetry in the P2 null that neither the review nor
the pre-registration caught. For the letter direction, the **family is excluded** before
the argmax — those are the most letter-selective latents. For a random direction,
**nothing is excluded**. So the null retains its most concentrated candidates while the
letter direction has had its removed by construction, which biases the comparison
toward "no carrier". D3 re-runs the null while dropping the top-$|\mathrm{fam}|$
latents by alignment with each random direction, making the exclusion symmetric.

This does not invalidate the registered P2 — the registered comparator is what it is,
and it is reported unchanged — but it means the size of the P2 gap should be read off
the symmetric version.

**Caveat on D3, stated because the self-test does not cover it.** The self-test builds
its "family" as five latents that *fire* on control trials; it does not give them
elevated alignment with $u$. The asymmetry D3 exists to measure is precisely that the
real family is the set of most letter-*selective* latents, i.e. the natural winners of
the letter-direction argmax, whereas the null's exclusion set is arbitrary. So the
self-test's finding that fair $\approx$ unfair in synthetic data is uninformative about
D3 — it never exercised the mechanism. D2 is validated by the self-test; **D3 is not**,
and its real-data number is reported as exploratory only. Note also that the exclusion
removes at most 32 of 16384 latents, so any effect comes from *which* latents are
dropped rather than how many.

---

## Outcome

*(Filled in from `results/real/results_round14_validity.txt` once the diagnostic
completes; the registered round-14 endpoints are unchanged regardless.)*
