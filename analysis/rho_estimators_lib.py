"""Round-9 estimator + diagnostic functions (numpy only, torch-free).

Shared by experiments/round9_rho_estimator.py (confirmatory), the D1
frozen-weight recompute, and the M0 analytic Monte Carlo check. Frozen
definitions per notes/prereg-gating-corrected-rho.md: undefined
denominators -> nan (never smoothed); dominance ties -> S; oracle
orientation is the caller's responsibility (pass par/comp correctly).
"""
THETA = 0.05

def rho_estimators(a_par, a_comp, theta=THETA):
    """All four estimators + pattern cells from raw activation vectors."""
    NAN = float("nan")
    bp = a_par > theta
    bc = a_comp > theta
    n11 = int((bp & bc).sum()); n10 = int((bp & ~bc).sum())
    n01 = int((~bp & bc).sum())
    either = n11 + n10 + n01
    # incumbent rarity-based count (descriptive; targets min(rho, 1-rho))
    lo_is_comp = bool(bc.mean() <= bp.mean()) if len(bp) else True  # tie -> comp
    lo_fires = bc if lo_is_comp else bp
    rho_count = float(lo_fires.sum() / either) if either else NAN
    # oracle-comp count: the leak-bias baseline
    rho_c = (n01 + n11) / either if either else NAN
    rho_x = n01 / (n01 + n10) if (n01 + n10) else NAN
    dom = a_comp > a_par                              # strict: ties -> S
    nJ = n01 + int((bp & bc & dom).sum())
    nS = n10 + int((bp & bc & ~dom).sum())
    rho_d = nJ / (nJ + nS) if (nJ + nS) else NAN
    ties = int((bp & bc & (a_comp == a_par)).sum())
    return dict(n11=n11, n10=n10, n01=n01,
                rho_count=round(rho_count, 5), rho_c=round(rho_c, 5),
                rho_x=round(rho_x, 5), rho_d=round(rho_d, 5), ties=ties,
                lo_is_comp=int(lo_is_comp))

def theta_sensitivity(a_par, a_comp):
    """Registered descriptive readout: rho_d at theta in {0.02, 0.10}."""
    return {f"rho_d_t{int(t*100):03d}": rho_estimators(a_par, a_comp, t)["rho_d"]
            for t in (0.02, 0.10)}

def oracle_diags(a_par, a_comp, mJ, mS, mB, theta=THETA):
    """Class-conditional fire rates, inversion rates, background pattern
    cells, h_B, realized rho (scoring only - uses oracle masks)."""
    bp = a_par > theta; bc = a_comp > theta
    def rate(mask, b):
        return round(float(b[mask].mean()), 5) if mask.any() else float("nan")
    both = bp & bc
    d = dict(a1=rate(mJ, bc), a0=rate(mS, bc), g0=rate(mS, bp), g1=rate(mJ, bp),
             b_p=rate(mB, bp), b_c=rate(mB, bc))
    j11 = mJ & both; s11 = mS & both; b11 = mB & both
    # class-11 denominators (prereg P4: delta defined only when >= 100)
    d["n_j11"] = int(j11.sum()); d["n_s11"] = int(s11.sum())
    d["delta_J"] = round(float((a_par[j11] >= a_comp[j11]).mean()), 5) if d["n_j11"] else float("nan")
    d["delta_S"] = round(float((a_comp[s11] > a_par[s11]).mean()), 5) if d["n_s11"] else float("nan")
    # background pattern cells (no independence assumed) + assignment share
    nB = max(int(mB.sum()), 1)
    q11 = float((mB & bp & bc).sum() / nB)
    q10 = float((mB & bp & ~bc).sum() / nB)
    q01 = float((mB & ~bp & bc).sum() / nB)
    d["q11_B"] = round(q11, 5); d["q10_B"] = round(q10, 5); d["q01_B"] = round(q01, 5)
    d["pi_B"] = round(float((a_comp[b11] > a_par[b11]).mean()), 5) if b11.any() else float("nan")
    active_B = q11 + q10 + q01
    pi = d["pi_B"] if b11.any() else 0.0
    d["h_B"] = round((q01 + q11 * pi) / active_B, 5) if active_B > 0 else float("nan")
    d["b11_frac"] = round(float(b11.sum() / max(both.sum() + (bp ^ bc).sum(), 1)), 5)
    # realized rho on this eval set (scoring target check)
    nJc, nSc = int(mJ.sum()), int(mS.sum())
    d["rho_real"] = round(nJc / (nJc + nSc), 5) if (nJc + nSc) else float("nan")
    return d
