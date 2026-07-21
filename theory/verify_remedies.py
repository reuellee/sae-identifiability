"""Symbolic derivation + verification of two provable absorption remedies.

Remedy A — decoder-coherence penalty: L_pen(D) = L(D) + beta * sum_{i<j} <d_i,d_j>^2.
  Claim: faithful beats absorbed for ALL eps >= 0 iff beta > beta* = 2q[(2-s2)lam - lam^2/4],
  and beats the redundant triple iff beta > beta*/2. Verified symbolically, plus a
  numeric global scan over all 2-latent dictionaries with the penalty.

Remedy B — Matryoshka (shared-code, nested prefixes {1}, {1,2}):
  Per-sample loss: ||x - f1 u1||^2 + ||x - f1 u1 - f2 u2||^2 + lam*(f1+f2), f >= 0.
  Derive closed-form event losses for faithful/absorbed dicts by KKT active-set
  enumeration; solve for the Matryoshka absorption boundary eps*_mat(lam, q);
  compare to vanilla eps*. Also verify the DECOUPLED (per-prefix optimal coding)
  variant leaves the boundary unchanged.
"""
from sympy import symbols, sqrt, Rational, simplify, solve, Matrix, factor, expand, nsimplify
import numpy as np

lam, q, p, eps, beta = symbols('lam q p eps beta', nonnegative=True)
s2 = sqrt(2)

# ---------------------------------------------------- shared KKT machinery
def min_nonneg(L, fvars, lam_vals):
    """Enumerate KKT active sets for min over f>=0 of quadratic+linear L.
    Returns list of (expr, coef_exprs); numeric min taken at eval time."""
    cases = [(L.subs({f: 0 for f in fvars}), [])]
    from itertools import combinations
    for r in range(1, len(fvars) + 1):
        for act in combinations(fvars, r):
            Ls = L.subs({f: 0 for f in fvars if f not in act})
            sol = solve([Ls.diff(f) for f in act], list(act), dict=True)
            for s in sol:
                cases.append((Ls.subs(s), [s[f] for f in act]))
    return cases

def eval_min(cases, subs_num):
    best = np.inf
    for L, coefs in cases:
        ok = True
        for c in coefs:
            v = complex(c.subs(subs_num))
            if abs(v.imag) > 1e-12 or v.real < -1e-12:
                ok = False; break
        if ok:
            best = min(best, float(L.subs(subs_num)))
    return best

f1, f2 = symbols('f1 f2', real=True)
X = {'joint': Matrix([1, 1]), 'psolo': Matrix([1, 0]), 'csolo': Matrix([0, 1])}
U_F = Matrix([[1, 0], [0, 1]])
U_A = Matrix([[1, 1/s2], [0, 1/s2]])

# ---------------------------------------------------- Remedy A: symbolic thresholds
print("=== Remedy A: coherence penalty ===")
LF = (p + eps + 2*q) * (lam - lam**2/4)
LA = q*(s2*lam - lam**2/4) + p*(lam - lam**2/4) + eps*(Rational(1,2) + s2*lam/2 - lam**2/4)
LT = q*(s2*lam - lam**2/4) + p*(lam - lam**2/4) + eps*(lam - lam**2/4)   # triple
# penalties: F:0, A: beta*1/2, T: beta*(1/2+1/2)
diff_FA = simplify((LF) - (LA + beta/2))          # faithful wins iff < 0
beta_star = solve(diff_FA.subs(eps, 0), beta)[0]
print("beta* (kills absorbed at eps=0; worst case over eps) =", factor(beta_star))
claimed = 2*q*((2 - s2)*lam - lam**2/4)
print("matches claimed 2q[(2-s2)lam - lam^2/4]:", simplify(beta_star - claimed) == 0)
diff_FT = simplify(LF - (LT + beta))
beta_T = solve(diff_FT, beta)[0]
print("beta needed vs triple =", factor(beta_T), "= beta*/2:",
      simplify(beta_T - beta_star/2) == 0)
# eps-dependence check: d(diff_FA)/d(eps) < 0 so eps=0 is worst case
print("d(L_F - L_A)/d(eps) =", simplify(diff_FA.diff(eps)), "(negative => eps=0 worst case)")

# numeric global scan with penalty
def pop_loss_pen(t1, t2, lam_v, q_v, p_v, e_v, beta_v):
    U = np.array([[np.cos(t1), np.cos(t2)], [np.sin(t1), np.sin(t2)]])
    tot = beta_v * (U[:, 0] @ U[:, 1]) ** 2
    for w, x in [(q_v, np.array([1., 1.])), (p_v, np.array([1., 0.])), (e_v, np.array([0., 1.]))]:
        if w == 0: continue
        best = x @ x
        for act in [(0,), (1,), (0, 1)]:
            Ua = U[:, list(act)]
            A_ = Ua.T @ Ua
            if len(act) == 2 and abs(np.linalg.det(A_)) < 1e-10: continue
            f = np.linalg.solve(A_, Ua.T @ x - lam_v / 2)
            if (f >= -1e-12).all():
                r = x - Ua @ f
                best = min(best, r @ r + lam_v * f.sum())
        tot += w * best
    return tot

lam_v, q_v, p_v = 0.2, 0.2, 0.2
bstar = float(beta_star.subs({lam: lam_v, q: q_v}))
print(f"beta* at lam={lam_v}, q={q_v}: {bstar:.5f}")
ts = np.deg2rad(np.linspace(0, 95, 96))
for beta_v in [0.0, 0.5 * bstar, 1.2 * bstar, 3 * bstar]:
    for e_v in [0.0, 0.01]:
        best = (np.inf, None)
        for t1 in ts:
            for t2 in ts:
                if t2 <= t1 + 1e-9: continue
                L = pop_loss_pen(t1, t2, lam_v, q_v, p_v, e_v, beta_v)
                if L < best[0]: best = (L, (np.degrees(t1), np.degrees(t2)))
        print(f"  beta={beta_v:.4f} eps={e_v}: global min at angles="
              f"({best[1][0]:.0f},{best[1][1]:.0f})deg")

# ---------------------------------------------------- Remedy B: Matryoshka
print("\n=== Remedy B: Matryoshka (shared code) ===")
def mat_loss_cases(U):
    u1, u2 = U[:, 0], U[:, 1]
    out = {}
    for ev, x in X.items():
        r1 = x - f1 * u1                      # prefix {1} residual
        r2 = x - f1 * u1 - f2 * u2            # full residual
        L = r1.dot(r1) + r2.dot(r2) + lam * (f1 + f2)
        out[ev] = min_nonneg(L, [f1, f2], None)
    return out

# closed forms via evaluation + fit: get exact expressions by solving symbolically
def mat_event_expr(U, ev):
    u1, u2 = U[:, 0], U[:, 1]
    x = X[ev]
    r1 = x - f1 * u1
    r2 = x - f1 * u1 - f2 * u2
    L = r1.dot(r1) + r2.dot(r2) + lam * (f1 + f2)
    cases = min_nonneg(L, [f1, f2], None)
    # pick the interior (both-active) solution when valid, else best boundary —
    # determine symbolically which case wins for small lam by numeric probe, then
    # return that case's expression.
    probe = {lam: 0.07}
    vals = []
    for i, (Le, coefs) in enumerate(cases):
        ok = all(abs(complex(c.subs(probe)).imag) < 1e-12 and
                 complex(c.subs(probe)).real >= -1e-12 for c in coefs)
        if ok: vals.append((float(Le.subs(probe)), i))
    best_i = min(vals)[1]
    return simplify(expand(cases[best_i][0]))

exprs = {}
for name, U in [('F', U_F), ('A', U_A)]:
    for ev in X:
        e_ = mat_event_expr(U, ev)
        exprs[(name, ev)] = e_
        print(f"  {name}/{ev}: {e_}")

LmF = q*exprs[('F','joint')] + p*exprs[('F','psolo')] + eps*exprs[('F','csolo')]
LmA = q*exprs[('A','joint')] + p*exprs[('A','psolo')] + eps*exprs[('A','csolo')]
dm = simplify(LmF - LmA)
print("  L_F^mat - L_A^mat =", factor(dm))
sols = solve(dm, eps)
print("  eps*_mat =", [simplify(s) for s in sols])
es_van = lam*q*(8 - 4*s2 - lam) / (2*(1 - (2 - s2)*lam))
for lv in (0.05, 0.1, 0.2, 0.3):
    for s in sols:
        e_m = float(s.subs({lam: lv, q: 0.2, p: 0.2}))
        e_v = float(es_van.subs({lam: lv, q: 0.2}))
        print(f"    lam={lv}: eps*_mat={e_m:.5f}  vs vanilla eps*={e_v:.5f}  ratio={e_m/e_v:.3f}")

# sanity: decoupled matryoshka (independent prefix coding) must leave boundary unchanged
print("\n  decoupled check: prefix term uses its own g1 (not f1)")
g1 = symbols('g1', real=True)
def dec_event(U, ev):
    u1, u2 = U[:, 0], U[:, 1]
    x = X[ev]
    Lp = (x - g1*u1).dot(x - g1*u1) + lam*g1          # prefix, own code
    Lf = (x - f1*u1 - f2*u2).dot(x - f1*u1 - f2*u2) + lam*(f1+f2)
    cp = min_nonneg(Lp, [g1], None)
    cf = min_nonneg(Lf, [f1, f2], None)
    probe = {lam: 0.07}
    vp = min(float(L.subs(probe)) for L, cs in cp
             if all(complex(c.subs(probe)).real >= -1e-12 for c in cs))
    vf = min(float(L.subs(probe)) for L, cs in cf
             if all(complex(c.subs(probe)).real >= -1e-12 for c in cs))
    return vp, vf
tot = {}
for name, U in [('F', U_F), ('A', U_A)]:
    t = 0.0
    for ev, w in [('joint', 0.2), ('psolo', 0.2), ('csolo', 0.012)]:
        vp, vf = dec_event(U, ev)
        t += w * (vp + vf)
    tot[name] = t
print(f"  decoupled at lam=0.07,eps=0.012 (vanilla eps*={float(es_van.subs({lam:0.07,q:0.2})):.4f}):"
      f" L_F={tot['F']:.5f} L_A={tot['A']:.5f} "
      f"-> {'ABSORBED still wins' if tot['A'] < tot['F'] else 'faithful wins'}")
