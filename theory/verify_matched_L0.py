"""Verification for theory/matched_L0_invariance.md — matched-L0 comparison of
L1 vs TopK absorption in the 2D toy (written and run BEFORE round-13b
unblinding; no 13b results were read).

Model (2D toy of verify_absorption_theory.py): orthonormal v_p, v_c; events
joint x=v_p+v_c (prob q), parent-solo x=v_p (p), child-solo x=v_c (eps).
L1 SAE: per-event min_{f>=0} ||x-Df||^2 + lam*sum(f), unit-norm atoms.
TopK: oracle k-sparse nonneg coding, L_k(U,x)=min_{|S|<=k} min_{f>=0}||x-U_S f||^2.
"Achieved L0" of a config = expected number of STRICTLY POSITIVE coefficients
at the per-event optimum (ties broken toward the smaller support).

Checks (script exits nonzero if any FAIL):
  A. L1 per-event losses match the known closed forms AND per-event achieved
     supports are lam-independent (support quantization): faithful (2,1,1) on
     (joint,psolo,csolo), absorbed (1,1,1), triple (1,1,1), for lam in (0,sqrt2).
     => E[L0]_F - E[L0]_A = q exactly; lam cannot tune achieved L0 continuously.
  B. Symbolic: eps*_L1 re-derived == known formula; leading order (4-2sqrt2)lam q;
     lam_c := unique root of eps*_L1(lam)=2q in (0,sqrt2) equals
     8-4sqrt2-sqrt(92-64sqrt2) ~ 1.1224; eps*_L1 strictly increasing on (0,lam_c]
     and < 2q strictly inside; > 2q on (lam_c, sqrt2) (numeric spot).
  C. TopK regression: k=1 both configs achieve support 1 on every active event
     (same achieved L0), crossover at eps=2q; k=2 faithful zero-loss with
     supports (2,1,1) and beats absorbed for all eps>0.
  D. MAIN: matched-L0 fixed-point agreement. Grid over (lam<=lam_c, q, p, eps):
     let C_L1 = exact-KKT-preferred config in {F,A}; match TopK's budget k so
     TopK's achieved L0 equals L1's; then TopK's preferred config == C_L1.
  E. Failure of UNCONDITIONAL invariance: at lam in (lam_c, sqrt2) there are
     (q,eps) with eps in (2q, eps*_L1(lam)) where L1 prefers absorbed while
     TopK at the same achieved L0 (k=1) prefers faithful. Plus informational
     scan: absorbed still beats merged-single-atom and empty configs there.
  F. Overcomplete escape for L1 (oracle): the triple {v_p,v_c,d_comp} beats the
     2-atom absorbed dict by eps*(1/2-(1-sqrt2/2)lam) (>0 for lam<1/(2-sqrt2))
     and the 2-atom faithful dict by q*((2-sqrt2)lam-lam^2/4) (>0 for lam<
     4(2-sqrt2)); symbolic + numeric. Triple achieved L0 = q+p+eps = TopK
     k=1 achieved L0 with the same triple (which is zero-loss, regression).
  G. 3-atom global scan (lam=0.1, q=p=0.2, eps in {0.05,0.2}): grid over all
     3-atom unit dictionaries; global optimum equals the exact triple and is
     functionally child-recovering (an atom aligned with v_c serves csolo and
     does not fire on psolo).
  H. Feasibility of low-L0 matching for L1: for eps>2q NO lam in (0,lam_c]
     makes L1 prefer absorbed; for eps<2q there exists lam<lam_c that does.
Stdlib + numpy + sympy only.
"""
import math
import sys
from itertools import combinations

import numpy as np
import sympy as sp

RT2 = math.sqrt(2.0)
FAILS = []


def rec(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f" [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- exact solvers
def l1_event(x, D, lam):
    """Exact min_{f>=0} ||x-Df||^2 + lam*sum(f) over 2D unit atoms D, by
    active-set (KKT) enumeration, returning (loss, achieved support size).
    In 2D an optimal solution always exists with <=2 active atoms: at fixed
    reconstruction y=Df, minimizing sum(f) over {f>=0: Df=y} is an LP with two
    equality constraints, so a basic optimal solution has <=2 nonzeros."""
    nx = x[0] * x[0] + x[1] * x[1]
    bestL, bestS = nx, 0
    m = len(D)
    for i in range(m):
        d = D[i]
        f = d[0] * x[0] + d[1] * x[1] - lam / 2
        if f > 1e-12:
            L = nx - f * f  # unit atom: optimal single-atom value
            if L < bestL - 1e-12:
                bestL, bestS = L, 1
    for i in range(m):
        for j in range(i + 1, m):
            d1, d2 = D[i], D[j]
            g = d1[0] * d2[0] + d1[1] * d2[1]
            det = 1 - g * g
            if abs(det) < 1e-10:
                continue
            b1 = d1[0] * x[0] + d1[1] * x[1] - lam / 2
            b2 = d2[0] * x[0] + d2[1] * x[1] - lam / 2
            f1 = (b1 - g * b2) / det
            f2 = (b2 - g * b1) / det
            if f1 > 1e-12 and f2 > 1e-12:
                r0 = x[0] - f1 * d1[0] - f2 * d2[0]
                r1 = x[1] - f1 * d1[1] - f2 * d2[1]
                L = r0 * r0 + r1 * r1 + lam * (f1 + f2)
                if L < bestL - 1e-12:
                    bestL, bestS = L, 2
    return bestL, bestS


def l1_event_supports(x, D, lam):
    """Same as l1_event but also returns the active atom indices at the optimum."""
    nx = x[0] * x[0] + x[1] * x[1]
    best = (nx, ())
    m = len(D)
    for i in range(m):
        d = D[i]
        f = d[0] * x[0] + d[1] * x[1] - lam / 2
        if f > 1e-12:
            L = nx - f * f
            if L < best[0] - 1e-12:
                best = (L, (i,))
    for i in range(m):
        for j in range(i + 1, m):
            d1, d2 = D[i], D[j]
            g = d1[0] * d2[0] + d1[1] * d2[1]
            det = 1 - g * g
            if abs(det) < 1e-10:
                continue
            b1 = d1[0] * x[0] + d1[1] * x[1] - lam / 2
            b2 = d2[0] * x[0] + d2[1] * x[1] - lam / 2
            f1 = (b1 - g * b2) / det
            f2 = (b2 - g * b1) / det
            if f1 > 1e-12 and f2 > 1e-12:
                r0 = x[0] - f1 * d1[0] - f2 * d2[0]
                r1 = x[1] - f1 * d1[1] - f2 * d2[1]
                L = r0 * r0 + r1 * r1 + lam * (f1 + f2)
                if L < best[0] - 1e-12:
                    best = (L, (i, j))
    return best


def topk_event(x, D, k):
    """Oracle k-sparse NNLS loss + achieved support (ties -> smaller support).
    Projection onto a polyhedral cone in 2D lies on a face generated by <=2
    rays, so supports of size <=min(k,2) suffice."""
    nx = x[0] * x[0] + x[1] * x[1]
    bestL, bestS = nx, 0
    m = len(D)
    if k >= 1:
        for i in range(m):
            d = D[i]
            f = d[0] * x[0] + d[1] * x[1]
            if f > 1e-12:
                L = nx - f * f
                if L < bestL - 1e-12:
                    bestL, bestS = L, 1
    if k >= 2:
        for i in range(m):
            for j in range(i + 1, m):
                d1, d2 = D[i], D[j]
                g = d1[0] * d2[0] + d1[1] * d2[1]
                det = 1 - g * g
                if abs(det) < 1e-10:
                    continue
                b1 = d1[0] * x[0] + d1[1] * x[1]
                b2 = d2[0] * x[0] + d2[1] * x[1]
                f1 = (b1 - g * b2) / det
                f2 = (b2 - g * b1) / det
                if f1 > 1e-12 and f2 > 1e-12:
                    r0 = x[0] - f1 * d1[0] - f2 * d2[0]
                    r1 = x[1] - f1 * d1[1] - f2 * d2[1]
                    L = r0 * r0 + r1 * r1
                    if L < bestL - 1e-12:
                        bestL, bestS = L, 2
    return bestL, bestS


EV = {'joint': (1.0, 1.0), 'psolo': (1.0, 0.0), 'csolo': (0.0, 1.0)}
D_F = [(1.0, 0.0), (0.0, 1.0)]
D_A = [(1.0, 0.0), (1 / RT2, 1 / RT2)]
D_3 = [(1.0, 0.0), (0.0, 1.0), (1 / RT2, 1 / RT2)]


def l1_pop(D, lam, q, p, e):
    """(population loss, achieved E[L0]) for an L1 SAE with dictionary D."""
    L = S = 0.0
    for w, ev in [(q, 'joint'), (p, 'psolo'), (e, 'csolo')]:
        l, s = l1_event(EV[ev], D, lam)
        L += w * l
        S += w * s
    return L, S


def topk_pop(D, k, q, p, e):
    L = S = 0.0
    for w, ev in [(q, 'joint'), (p, 'psolo'), (e, 'csolo')]:
        l, s = topk_event(EV[ev], D, k)
        L += w * l
        S += w * s
    return L, S


def eps_star_num(lam, q):
    return lam * q * (8 - 4 * RT2 - lam) / (2 * (1 - (2 - RT2) * lam))


LAM_C_NUM = 8 - 4 * RT2 - math.sqrt(92 - 64 * RT2)  # ~1.12240

# =============================================================== A
print("=== A: L1 per-event losses (closed forms) and support quantization ===")
lam_grid = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.3, 1.38]
claims = {
    'F': ({'joint': lambda l: 2 * l - l * l / 2,
           'psolo': lambda l: l - l * l / 4,
           'csolo': lambda l: l - l * l / 4},
          {'joint': 2, 'psolo': 1, 'csolo': 1}, D_F),
    'A': ({'joint': lambda l: RT2 * l - l * l / 4,
           'psolo': lambda l: l - l * l / 4,
           'csolo': lambda l: 0.5 + RT2 * l / 2 - l * l / 4},
          {'joint': 1, 'psolo': 1, 'csolo': 1}, D_A),
    'triple': ({'joint': lambda l: RT2 * l - l * l / 4,
                'psolo': lambda l: l - l * l / 4,
                'csolo': lambda l: l - l * l / 4},
               {'joint': 1, 'psolo': 1, 'csolo': 1}, D_3),
}
for name, (cl, su, D) in claims.items():
    okL = okS = True
    bad = ""
    for ev, x in EV.items():
        for l in lam_grid:
            L, S = l1_event(x, D, l)
            if abs(L - cl[ev](l)) > 1e-9:
                okL, bad = False, f"{ev} lam={l} got {L:.6f} want {cl[ev](l):.6f}"
            if S != su[ev]:
                okS, bad = False, f"{ev} lam={l} support {S} want {su[ev]}"
    rec(f"A1 {name}: event losses match closed forms (lam in (0,1.38])", okL, bad)
    rec(f"A2 {name}: per-event supports lam-INDEPENDENT = "
        f"({su['joint']},{su['psolo']},{su['csolo']})", okS, bad)
# achieved-L0 gap = q, and constancy in lam
ok = True
for (q, p, e) in [(0.2, 0.2, 0.05), (0.1, 0.3, 0.15), (0.3, 0.05, 0.4)]:
    for l in lam_grid:
        _, SF = l1_pop(D_F, l, q, p, e)
        _, SA = l1_pop(D_A, l, q, p, e)
        if abs(SF - (2 * q + p + e)) > 1e-12 or abs(SA - (q + p + e)) > 1e-12:
            ok = False
rec("A3 achieved E[L0]: F = 2q+p+eps, A = q+p+eps for ALL lam => gap = q, "
    "not tunable by lam", ok)

# =============================================================== B
print("=== B: symbolic threshold, lam_c, ordering vs 2q ===")
lam_s, q_s, p_s, e_s = sp.symbols('lam q p eps', positive=True)
s2 = sp.sqrt(2)
LF_s = (q_s * (2 * lam_s - lam_s**2 / 2) + p_s * (lam_s - lam_s**2 / 4)
        + e_s * (lam_s - lam_s**2 / 4))
LA_s = (q_s * (s2 * lam_s - lam_s**2 / 4) + p_s * (lam_s - lam_s**2 / 4)
        + e_s * (sp.Rational(1, 2) + s2 * lam_s / 2 - lam_s**2 / 4))
eps_star_s = sp.simplify(sp.solve(sp.Eq(LF_s, LA_s), e_s)[0])
known = lam_s * q_s * (8 - 4 * s2 - lam_s) / (2 * (1 - (2 - s2) * lam_s))
rec("B1 eps*_L1 re-derived == known closed form (p-independent)",
    sp.simplify(eps_star_s - known) == 0)
lead = sp.limit(eps_star_s / (lam_s * q_s), lam_s, 0)
rec("B2 leading order eps*_L1 -> (4-2*sqrt2)*lam*q ~ 1.172 lam q",
    sp.simplify(lead - (4 - 2 * s2)) == 0, f"coeff={float(lead):.4f}")
roots = sp.solve(sp.Eq(known, 2 * q_s), lam_s)
lam_c_s = 8 - 4 * s2 - sp.sqrt(92 - 64 * s2)
ok = any(sp.simplify(r - lam_c_s) == 0 for r in roots)
rec("B3 lam_c = 8-4*sqrt2-sqrt(92-64*sqrt2) solves eps*_L1(lam)=2q", ok,
    f"lam_c={float(lam_c_s):.5f}")
rec("B3b lam_c in (0, sqrt2) (inside the formula's validity window)",
    0 < float(lam_c_s) < RT2, f"{float(lam_c_s):.5f} < {RT2:.5f}")
rec("B3c numeric constant matches", abs(float(lam_c_s) - LAM_C_NUM) < 1e-12)
# monotone increasing on (0, lam_c], strictly below 2q inside
ls = np.linspace(1e-4, LAM_C_NUM, 400)
vals = np.array([eps_star_num(l, 1.0) for l in ls])  # q=1 scale (eps*/q)
ok_mono = bool(np.all(np.diff(vals) > 0))
ok_below = bool(np.all(vals[:-1] < 2.0)) and abs(vals[-1] - 2.0) < 1e-9
rec("B4 eps*_L1/q strictly increasing on (0,lam_c], reaching 2 exactly at lam_c",
    ok_mono and ok_below, f"eps*(lam_c)/q={vals[-1]:.6f}")
rec("B4b eps*_L1 > 2q on (lam_c, sqrt2): spot lam=1.25",
    eps_star_num(1.25, 0.2) > 0.4, f"eps*={eps_star_num(1.25, 0.2):.4f} vs 2q=0.4")
# quantitative remark for moderate lam
rec("B5 moderate shrinkage: eps*_L1(0.5,q) ~ 0.652 q  (<< 2q)",
    abs(eps_star_num(0.5, 1.0) - 0.6517) < 5e-4,
    f"{eps_star_num(0.5, 1.0):.4f} q")

# =============================================================== C
print("=== C: TopK regression: supports and 2q crossover ===")
ok = True
for name, D in [('F', D_F), ('A', D_A)]:
    for ev in EV:
        _, S = topk_event(EV[ev], D, 1)
        if S != 1:
            ok = False
rec("C1 k=1: every active event uses support 1 in BOTH configs "
    "(achieved L0 = q+p+eps regardless of config)", ok)
ok = True
for q in (0.1, 0.2, 0.3):
    LF1, _ = topk_pop(D_F, 1, q, 0.2, 2 * q)
    LA1, _ = topk_pop(D_A, 1, q, 0.2, 2 * q)
    if abs(LF1 - LA1) > 1e-9:
        ok = False
    if not (topk_pop(D_A, 1, q, 0.2, q)[0] < topk_pop(D_F, 1, q, 0.2, q)[0]):
        ok = False
    if not (topk_pop(D_A, 1, q, 0.2, 3 * q)[0] > topk_pop(D_F, 1, q, 0.2, 3 * q)[0]):
        ok = False
rec("C2 k=1 crossover at eps = 2q (tie/below/above)", ok)
LFj, SFj = topk_event(EV['joint'], D_F, 2)
ok = abs(LFj) < 1e-12 and SFj == 2
for ev in ('psolo', 'csolo'):
    L, S = topk_event(EV[ev], D_F, 2)
    ok = ok and abs(L) < 1e-12 and S == 1
rec("C3 k=2 faithful: zero loss, supports (2,1,1) => achieved L0 = 2q+p+eps", ok)
ok = all(topk_pop(D_F, 2, 0.2, 0.2, e)[0] < topk_pop(D_A, 2, 0.2, 0.2, e)[0]
         for e in (0.01, 0.1, 0.4, 0.8))
rec("C4 k=2 faithful beats absorbed for all eps>0", ok)

# =============================================================== D
print("=== D: MAIN — matched-L0 fixed-point agreement for lam in (0, lam_c] ===")
lamD = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1]
qD = [0.05, 0.15, 0.25]
pD = [0.05, 0.2]
cells = agree = skipped = 0
bad = ""
for lam in lamD:
    for q in qD:
        es = eps_star_num(lam, q)
        for p in pD:
            eps_list = ([m * es for m in (0.3, 0.7, 0.95, 1.05, 1.5, 2.5)]
                        + [m * 2 * q for m in (0.5, 0.9, 1.1, 1.6)])
            for e in eps_list:
                if e < 1e-6 or q + p + e > 0.99:
                    continue
                if abs(e - 2 * q) < 1e-9 or abs(e - es) < 1e-9:
                    skipped += 1
                    continue
                LF, SF = l1_pop(D_F, lam, q, p, e)
                LA, SA = l1_pop(D_A, lam, q, p, e)
                if abs(LF - LA) < 1e-10:
                    skipped += 1
                    continue
                cells += 1
                c_l1 = 'A' if LA < LF else 'F'
                l1_L0 = SA if c_l1 == 'A' else SF
                # matched TopK budget: k with TopK achieved L0 == L1 achieved L0
                k = 1 if abs(l1_L0 - (q + p + e)) < 1e-9 else 2
                # sanity: TopK achieved L0 at budget k under ITS preferred config
                LFk, SFk = topk_pop(D_F, k, q, p, e)
                LAk, SAk = topk_pop(D_A, k, q, p, e)
                c_topk = 'A' if LAk < LFk else 'F'
                topk_L0 = SAk if c_topk == 'A' else SFk
                if c_topk == c_l1 and abs(topk_L0 - l1_L0) < 1e-9:
                    agree += 1
                elif not bad:
                    bad = (f"lam={lam} q={q} p={p} eps={e:.4f}: "
                           f"L1->{c_l1}(L0={l1_L0:.3f}) TopK(k={k})->{c_topk}"
                           f"(L0={topk_L0:.3f})")
rec(f"D1 decisions AGREE in all {cells} non-tie cells (skipped {skipped} ties)",
    agree == cells and cells > 300, bad or f"{agree}/{cells}")

# =============================================================== E
print("=== E: unconditional invariance FAILS for lam in (lam_c, sqrt2) ===")
lam, q, p, e = 1.25, 0.2, 0.2, 0.45  # 2q=0.4 < eps=0.45 < eps*_L1(1.25)=0.510
LF, SF = l1_pop(D_F, lam, q, p, e)
LA, SA = l1_pop(D_A, lam, q, p, e)
LF1, SF1 = topk_pop(D_F, 1, q, p, e)
LA1, SA1 = topk_pop(D_A, 1, q, p, e)
rec("E1 L1 prefers ABSORBED (eps < eps*_L1(1.25) = "
    f"{eps_star_num(lam, q):.4f})", LA < LF, f"LA={LA:.4f} < LF={LF:.4f}")
rec("E2 TopK k=1 prefers FAITHFUL (eps > 2q)", LF1 < LA1,
    f"LF1={LF1:.4f} < LA1={LA1:.4f}")
rec("E3 both operate at the SAME achieved L0 = q+p+eps",
    abs(SA - (q + p + e)) < 1e-12 and abs(SF1 - (q + p + e)) < 1e-12
    and abs(SA1 - (q + p + e)) < 1e-12,
    f"L1-A: {SA:.3f}, TopK-F: {SF1:.3f}")
# informational: absorbed still beats 1-atom merged and empty configs there
best1 = (np.inf, None)
for a in np.arange(-90, 181, 0.25):
    t = math.radians(a)
    L, _ = l1_pop([(math.cos(t), math.sin(t))], lam, q, p, e)
    if L < best1[0]:
        best1 = (L, a)
L_empty = q * 2 + p + e
rec("E4 disagreement point not vacuous: absorbed beats best single-atom "
    "and empty dicts", LA < best1[0] and LA < L_empty,
    f"LA={LA:.4f}, best1={best1[0]:.4f}@{best1[1]:.0f}deg, empty={L_empty:.4f}")
# informational 2-atom global scan at the disagreement point
bestg = (np.inf, None)
angs = np.arange(-90, 181, 1.0)
for i, a1 in enumerate(angs):
    t1 = math.radians(a1)
    d1 = (math.cos(t1), math.sin(t1))
    for a2 in angs[i + 1:]:
        t2 = math.radians(a2)
        L, _ = l1_pop([d1, (math.cos(t2), math.sin(t2))], lam, q, p, e)
        if L < bestg[0]:
            bestg = (L, (a1, a2))
print(f"  INFO  global 2-atom optimum at lam=1.25: L={bestg[0]:.4f} at "
      f"{bestg[1]} deg (absorbed candidate L={LA:.4f})")

# =============================================================== F
print("=== F: overcomplete escape — free third atom, both architectures ===")
L3_s = (q_s * (s2 * lam_s - lam_s**2 / 4) + p_s * (lam_s - lam_s**2 / 4)
        + e_s * (lam_s - lam_s**2 / 4))
mA = sp.simplify(LA_s - L3_s)
claimA_m = e_s * (sp.Rational(1, 2) - (1 - s2 / 2) * lam_s)
rec("F1 symbolic: L_absorbed2 - L_triple = eps*(1/2 - (1-sqrt2/2)*lam)",
    sp.simplify(mA - claimA_m) == 0)
rec("F1b margin > 0 for all 0<lam<1/(2-sqrt2)~1.707 (so on the whole "
    "validity window lam<sqrt2)",
    all(eps * (0.5 - (1 - RT2 / 2) * l) > 0
        for l in [0.01, 0.5, 1.0, 1.4] for eps in [0.01, 0.5]))
mF = sp.simplify(LF_s - L3_s)
claimF_m = q_s * ((2 - s2) * lam_s - lam_s**2 / 4)
rec("F2 symbolic: L_faithful2 - L_triple = q*((2-sqrt2)*lam - lam^2/4) > 0 "
    "for 0<lam<4(2-sqrt2)", sp.simplify(mF - claimF_m) == 0)
# numeric: triple losses+supports already checked in A; achieved L0:
ok = True
for l in [0.05, 0.2, 0.5, 1.0]:
    _, S3 = l1_pop(D_3, l, 0.2, 0.2, 0.1)
    if abs(S3 - (0.2 + 0.2 + 0.1)) > 1e-12:
        ok = False
rec("F3 L1 triple achieved E[L0] = q+p+eps (composite serves joint ALONE; "
    "child atom fires only on csolo)", ok)
ok = True
for ev in EV:
    L, S = topk_event(EV[ev], D_3, 1)
    if abs(L) > 1e-12 or S != 1:
        ok = False
rec("F4 TopK triple k=1: zero loss, support 1 per event (regression; same "
    "achieved L0 as L1 triple)", ok)

# =============================================================== G
print("=== G: 3-atom global scan — overcomplete L1 optimum is the triple and "
      "is child-recovering ===")
lam, q, p = 0.1, 0.2, 0.2
angsG = np.arange(-45.0, 135.1, 2.5)  # includes 0, 45, 90 exactly
dirs = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in angsG]
for e in (0.05, 0.2):
    L_triple, _ = l1_pop(D_3, lam, q, p, e)
    best = (np.inf, None)
    for i, j, k in combinations(range(len(angsG)), 3):
        D = [dirs[i], dirs[j], dirs[k]]
        L = (q * l1_event(EV['joint'], D, lam)[0]
             + p * l1_event(EV['psolo'], D, lam)[0]
             + e * l1_event(EV['csolo'], D, lam)[0])
        if L < best[0]:
            best = (L, (angsG[i], angsG[j], angsG[k]))
    rec(f"G1 eps={e}: grid-global 3-atom optimum == exact triple loss",
        abs(best[0] - L_triple) < 1e-9,
        f"global={best[0]:.6f}@{best[1]} vs triple={L_triple:.6f}")
    # functional child recovery at the scan optimum
    Dopt = [dirs[list(angsG).index(a)] for a in best[1]] if best[1] else None
    _, sup_c = l1_event_supports(EV['csolo'], Dopt, lam)
    _, sup_p = l1_event_supports(EV['psolo'], Dopt, lam)
    okrec = any(Dopt[i][1] > 0.9 and i not in sup_p for i in sup_c)
    rec(f"G2 eps={e}: optimum is functionally child-recovering "
        "(csolo served by a v_c-aligned atom that is silent on psolo)", okrec,
        f"csolo support angles={[best[1][i] for i in sup_c]}")

# =============================================================== H
print("=== H: feasibility of the low-L0 operating point for L1 ===")
q, p = 0.2, 0.1
e_hi = 0.45  # > 2q
ok = True
for lam in np.linspace(0.02, LAM_C_NUM - 1e-3, 60):
    LF, _ = l1_pop(D_F, lam, q, p, e_hi)
    LA, _ = l1_pop(D_A, lam, q, p, e_hi)
    if LA < LF:
        ok = False
rec("H1 eps=0.45 > 2q=0.4: NO lam in (0,lam_c] makes L1 prefer absorbed "
    "(low-L0 matching infeasible; L1 sits at E[L0]=2q+p+eps)", ok)
ok = True
for e_lo, lam in [(0.30, 1.05), (0.10, 0.40), (0.02, 0.10)]:
    LF, _ = l1_pop(D_F, lam, q, p, e_lo)
    LA, _ = l1_pop(D_A, lam, q, p, e_lo)
    if not LA < LF:
        ok = False
        print(f"    (eps={e_lo}, lam={lam}: expected absorbed preferred)")
rec("H2 eps < 2q: suitable lam < lam_c DOES put L1 at the low-L0 (absorbed) "
    "point (eps,lam) in {(0.30,1.05),(0.10,0.40),(0.02,0.10)}", ok)

print()
if FAILS:
    print("MATCHED-L0 VERIFICATION FAILED:")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("verify_matched_L0: ALL CHECKS PASS. Summary: L1 achieved L0 is "
      "support-quantized (gap = q, lam-independent); eps*_L1(lam) increases to "
      "2q exactly at lam_c = 8-4*sqrt2-sqrt(92-64*sqrt2) ~ 1.1224; matched-L0 "
      "decisions coincide for all lam <= lam_c, and provably disagree for "
      "lam in (lam_c, sqrt2); with a free third atom BOTH architectures "
      "escape absorption at the oracle level.")
