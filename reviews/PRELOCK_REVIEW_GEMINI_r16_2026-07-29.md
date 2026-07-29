# Pre-lock Review: Round 16 (L0-axis)

**Reviewer:** Gemini 3.1 Pro (High)
**Date:** 2026-07-29
**Verdict:** REVISE-BEFORE-LOCK

The round design contains a structural tautology in its primary claim, a severe evaluator-vs-prereg mismatch that compromises the manipulation check, and a statistical survivor-bias in the secondary metric. These must be addressed before locking.

### 1. [P1 Severity] Endpoint Tautology & Verdict Blindness to D-Control (Design Flaw)
P1 claims "absorption falls as L0 rises" and uses the endpoint `rate_family(k16) - rate_family(k64) > 0` for confirmation. 
**The Flaw:** This prediction is mechanically guaranteed by the endpoint's construction. `rate_family` (absorption) requires that NO family latent fires (`~fires[np.ix_(Lw, fam)].any(axis=1)`). By definition, increasing L0 (from 16 to 64) forces more latents to fire across the board. If every latent has a higher probability of firing, the probability of an "all zeros" event drops mechanically. 

While the prereg acknowledges this ("firing probability scales with L0 by construction") and introduces the D-control (`fam_fire_absent`) to dissociate a specific slot effect from a general threshold shift, the **formal verdict completely ignores it**. 
In `analyze_round16.py`:
```python
    if lo1 > 0:
        p1 = "CONFIRMED (absorption falls as L0 rises -- D1's direction on Pythia)"
```
And in the prereg:
> "No verdict vocabulary attaches to the D-control; it constrains the interpretation section only."

This guarantees the script will print a headline "CONFIRMED" even if the D-control reveals the effect is 100% a mechanical artifact of the global threshold shift. 
**Fix:** The P1 `CONFIRMED` verdict must be mathematically gated by the D-control. If `fam_fire_absent` rises comparably to `fam_fire_present`, it is a threshold artifact and should not be labeled a confirmation of capacity dynamics.

### 2. [P1 Severity] Missing Absolute L0 Band Gate for L1 (Evaluator-vs-Prereg Mismatch)
**The Flaw:** The prereg strictly mandates an absolute L0 band check for L1 as part of the manipulation check (MC):
> "For L1 rows ... the realized held-out L0 must land in the cell's band (manipulation check) — fail-closed, as the config gate standard requires."

**The Mismatch:** `analyze_round16.py` fails to implement this. The MC logic only checks the *relative* ratio:
```python
            lo, hi = cell[(a, SMALLK)]["l0"], cell[(a, LARGEK)]["l0"]
            mc[a] = hi >= MC_RATIO * lo
```
There is no absolute band check (14-18 for k16, 56-72 for k64) in the MC. While Gate 3 contains a band check (`G3_BAND`), it explicitly *only gates P3*. This leaves P1 completely exposed: if L1 calibrates to L0=2 and L0=5, it satisfies the 2.5x ratio, passes the MC, and pollutes the P1 pooled estimate, directly violating the fail-closed mandate.
**Fix:** Implement the absolute band bounds in `analyze_round16.py`'s `mc` check or Gate 1.

### 3. [P2 Severity] Survivor Bias in P2 Family Size Contrast (Statistical Error)
**The Flaw:** P2 evaluates if "split families grow with L0" by contrasting the mean family size:
```python
    d2 = [fs[("l1", LARGEK, s)] - fs[("l1", SMALLK, s)] for s in range(8) ... ]
```
**The Statistical Error:** `mean_famsize(row)` computes an *unpaired mean* over whatever letters happen to be clean (`sel >= TAU`) in that specific cell. Because higher L0 allows more latents to cross the `TAU` threshold, the set of clean letters at k64 will systematically differ from the set at k16. 
You are comparing the mean of a small set of "easy" letters at k16 to a larger set of "harder" letters at k64. Any difference is confounded by this composition effect (survivor bias), making it impossible to tell if individual families actually grew.
**Fix:** P2 must compute the paired difference *per letter* for the intersection of letters that are clean in *both* the k16 and k64 cells for that seed, and then average those differences.

### 4. [P3 Severity] Gating Bypass for Empty Interior Cell (Gate Logic)
**The Flaw:** The prereg states regarding the interior cell:
> "Its identity is pinned by SHA256 in the frozen evaluator ... any mismatch fails conformance."

**The Error:** In `analyze_round16.py`, the gate logic is guarded by the presence of the cell:
```python
    if interior and int_ids != INTERIOR_PINS:
        viol.append(...)
```
If the download in `l4_r16.sh` fails silently and the JSON is empty, `interior` is falsey. The check is skipped, and Gate 1 passes. An absent cell silently passes conformance, violating the explicit "any mismatch" rule.
**Fix:** Remove the `if interior:` guard in Gate 1, or update the prereg to explicitly state that the interior cell's absence is non-fatal.

***
**VERDICT: REVISE-BEFORE-LOCK**
