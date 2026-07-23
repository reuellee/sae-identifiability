# Round-10 prereg: response to pre-lock external reviews (2026-07-24)

Reviews: `ROUND10_PREREG_REVIEW_GEMINI-2.5-PRO_2026-07-24.md` (MINOR — algebra
independently verified) and `ROUND10_PREREG_REVIEW_GPT-5.6_2026-07-24.md`
(MAJOR — the 3-atom counterexample + six scoring loopholes; GPT fetched the
public repo and ran the verifier itself). The GPT review substantially
improved the round: it exposed that the crossover is *capacity-limited* to a
two-atom dictionary, which reframed the whole round around the sharper,
practically-relevant headline — **overcomplete TopK resists the absorption L1
suffers**.

## GPT-5.6 blocking issues → dispositions (all applied)

1. **Theorem assumes 2 atoms, experiment had m=16 (3-atom escape).** →
   **applied**: added the exact **arm A (m=2, n_bg=0)** that instantiates the
   theorem; **reclassified the m=16 arm B as an overcomplete SGD-behaviour
   test** (P3), explicitly not a confirmation of the two-atom theory; added the
   3-atom zero-loss counterexample to the verifier; scoped every theory
   statement to its atom count (theory §6).
2. **Oracle sparse-coding ≠ trained encoder.** → **applied**: the theoretical
   object is now named "two-atom oracle-coded k-sparse dictionary model"
   (§2a); registered encoder/gate diagnostics (pair-atom slot occupancy,
   conditional firing on joint/parent-solo/child-solo, achieved-vs-oracle
   reconstruction) so a failed cell is diagnosable.
3. **φ_child could pass on a dead / −v_c atom.** → **applied**: the primary
   metric is now a **functional, activation-aware** child_recovered — signed
   cos ≥ 0.6, fires on child-solo ≥ 0.5, child-selective ≥ 0.3. φ_child is
   demoted to a secondary diagnostic. Signed (not absolute) cosine.
4. **Outcome-dependent missingness.** → **eliminated by construction**: the
   metric is binary and always defined, so there is no undefined outcome to
   exclude; recovery_rate is a mean over all 24 seeds.
5. **ε=0.05 clause softened a registered failure.** → **applied**: a localized
   ε=0.05 P2 failure is scored **FALSIFIED** (localized), *diagnosed* as an
   SGD/encoder reachability limit; the diagnosis qualifies interpretation, not
   the verdict (verified in the scorer test).
6. **Vacuous T2 cells could pass.** → **applied**: predesignated tight-budget
   cells must absorb at k=1 for P2 to confirm; a vacuous cell yields
   INCONCLUSIVE (not pass) — the headline requires demonstrated tight-budget
   absorption (verified in the scorer test).

## Major theory corrections (all applied)

- "Any dictionary containing the composite is strictly worse" was **false** →
  corrected; absorption redefined as **absence of a functional child
  representation** (§2b, §6); the ½ε penalty scoped to the specific two-atom
  absorbed dictionary.
- Faithful is **not the unique** k=2 optimum (wide-cone neighbours) →
  corrected; the κ≥2 statement is existence of zero-loss non-absorbed
  solutions, not unique v_c recovery (§6, verified).
- ε=0 is **non-identifiability**, not impossibility "under any architecture" →
  corrected (§7): v_c = joint − parent is recoverable given the support; the
  wall is non-uniqueness of the reconstruction objective.

## Verifier corrections (all applied)

Reframed as **numerical regression tests + analytic two-atom propositions**,
not "proof". Added: the 3-atom zero-loss counterexample; wide-cone k=2
optima; the redundant child+composite solution; exact assertions on every
claimed tilt angle; the 2-atom-vs-arbitrary-m scope check. All pass.

## Prereg ambiguities (all applied)

- T1 (P1) ε grid extended to **0.60** so the q=0.2 crossover is falsifiable
  past 1.3·2q=0.52; a right-censored ε_mid is scored **inconclusive**.
- Explicit pass/FALSIFIED/inconclusive naming and per-claim effect stated.
- The λ arm was already dropped (it tested robustness-to-L1, not
  λ-independence, per GPT); noted as future in theory §9.
- The scorer `analysis/analyze_round10.py` is **committed and frozen before
  lock** (loophole 10), and validated on fabricated ideal/vacuous/localized-
  failure CSVs.

## Gemini-2.5-Pro changes (from the earlier pass, retained)

SGD-vs-global-optimum caveat (now folded into P2/P3 framing), stochastic
background note (moot — the redesign uses n_bg=0 isolated arms), nonnegativity
geometric intuition (kept, §3).

Net: the round now has an exact-theorem arm (m=2), an honest overcomplete
SGD-behaviour arm (m=16 TopK), and a descriptive L1-vs-TopK head-to-head, all
scored by an activation-aware binary metric with no missingness or geometry
loopholes.
