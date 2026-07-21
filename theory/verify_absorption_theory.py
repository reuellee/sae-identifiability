"""Symbolic + numeric verification of the feature-absorption phase boundary.

Model: parent/child features a_p, a_c orthonormal. Work in their 2D span,
coordinates (parent_coord, child_coord).
Events: joint x=(1,1) w.p. q; parent-solo x=(1,0) w.p. p; child-solo x=(0,1) w.p. eps.
SAE loss per sample: min_{f>=0} ||x - D f||^2 + lam * sum(f), D unit columns.

Dictionaries:
  Faithful: u1=(1,0), u2=(0,1)
  Absorbed: u1=(1,0), u2=(1,1)/sqrt(2)

Checks:
  1. Symbolic: optimal nonneg 2-latent code for each (event, dictionary),
     via KKT case enumeration; confirm closed-form event losses.
  2. Confirm L_F - L_A and the threshold eps*(lam, q).
  3. Numeric global scan over ALL unit-norm 2-latent dictionaries
     (angles t1, t2) to confirm faithful/absorbed are the competing optima
     and the global-optimum switchover matches eps*.
"""
import numpy as np
from sympy import (symbols, sqrt, Rational, simplify, solve, Piecewise,
                   Matrix, lambdify, nsimplify, factor, together)

lam, q, p, eps, g, h = symbols('lam q p eps g h', nonnegative=True)

def event_loss_sym(x, U):
    """Exact min over f>=0 of ||x - U f||^2 + lam*(f1+f2), by KKT case enumeration."""
    f1, f2 = symbols('f1 f2', real=True)
    best = None
    cases = []
    # Case (0,0)
    cases.append((x.dot(x), []))
    # Case (f1>0, f2=0) and (0, f2>0)
    for j in range(2):
        u = U[:, j]
        fj = symbols('fj', real=True)
        r = x - fj * u
        L = r.dot(r) + lam * fj
        sol = solve(L.diff(fj), fj)
        for s in sol:
            cases.append((L.subs(fj, s), [s]))  # validity (s>=0) checked at eval time
    # Case both > 0
    r = x - g * U[:, 0] - h * U[:, 1]
    L = r.dot(r) + lam * (g + h)
    sol = solve([L.diff(g), L.diff(h)], [g, h], dict=True)
    for s in sol:
        cases.append((L.subs(s), [s[g], s[h]]))
    return cases

def min_loss_at(cases, lam_v):
    best = np.inf
    for L, coefs in cases:
        vals = [complex(c.subs(lam, lam_v)) for c in coefs]
        if any(abs(v.imag) > 1e-12 or v.real < -1e-12 for v in vals):
            continue
        Lv = complex(L.subs(lam, lam_v))
        best = min(best, Lv.real)
    return best

s2 = sqrt(2)
U_F = Matrix([[1, 0], [0, 1]])
U_A = Matrix([[1, 1/s2], [0, 1/s2]])
X = {'joint': Matrix([1, 1]), 'psolo': Matrix([1, 0]), 'csolo': Matrix([0, 1])}

# Claimed closed forms (derived by hand)
claimed = {
    ('F', 'joint'): 2*lam - lam**2/2,
    ('F', 'psolo'): lam - lam**2/4,
    ('F', 'csolo'): lam - lam**2/4,
    ('A', 'joint'): s2*lam - lam**2/4,
    ('A', 'psolo'): lam - lam**2/4,
    ('A', 'csolo'): Rational(1, 2) + s2*lam/2 - lam**2/4,
}

print("=== Check 1: event losses (numeric eval of exact KKT enumeration vs claimed) ===")
ok = True
for name, U in [('F', U_F), ('A', U_A)]:
    for ev, x in X.items():
        cases = event_loss_sym(x, U)
        for lam_v in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            exact = min_loss_at(cases, lam_v)
            cl = float(claimed[(name, ev)].subs(lam, lam_v))
            if abs(exact - cl) > 1e-9:
                print(f"  MISMATCH {name}/{ev} lam={lam_v}: exact={exact:.6f} claimed={cl:.6f}")
                ok = False
print("  all closed forms match" if ok else "  CLOSED FORMS WRONG")

print("=== Check 2: threshold formula ===")
L_F = (p + eps + 2*q) * (lam - lam**2/4)
LFj = q*(2*lam - lam**2/2) + p*(lam - lam**2/4) + eps*(lam - lam**2/4)
assert simplify(L_F - LFj) == 0
L_A = q*(s2*lam - lam**2/4) + p*(lam - lam**2/4) + eps*(Rational(1,2) + s2*lam/2 - lam**2/4)
diff = simplify(L_F - L_A)
print("  L_F - L_A =", factor(diff))
eps_star = solve(diff, eps)[0]
eps_star = simplify(eps_star)
print("  eps* =", eps_star)
claimed_thr = q*((2 - s2)*lam - lam**2/4) / (Rational(1,2) - (2 - s2)*lam/2)
print("  matches claimed:", simplify(eps_star - claimed_thr) == 0)

print("=== Check 3: numeric global scan over all unit-norm 2-latent dictionaries ===")
def pop_loss_num(t1, t2, lam_v, q_v, p_v, e_v):
    U = np.array([[np.cos(t1), np.cos(t2)], [np.sin(t1), np.sin(t2)]])
    tot = 0.0
    for w, x in [(q_v, np.array([1., 1.])), (p_v, np.array([1., 0.])), (e_v, np.array([0., 1.]))]:
        # nonneg lasso in 2D by fine grid + polish
        best = x @ x
        # active-set exact solves
        for act in [(0,), (1,), (0, 1)]:
            Ua = U[:, list(act)]
            # min ||x-Ua f||^2 + lam*1'f  ->  f = (Ua'Ua)^-1 (Ua'x - lam/2)
            A_ = Ua.T @ Ua
            try:
                f = np.linalg.solve(A_, Ua.T @ x - lam_v / 2)
            except np.linalg.LinAlgError:
                continue
            if (f >= -1e-12).all():
                r = x - Ua @ f
                best = min(best, r @ r + lam_v * f.sum())
        tot += w * best
    return tot

rng = np.random.default_rng(0)
q_v, p_v = 0.2, 0.2
lam_v = 0.1
es = float(eps_star.subs({q: q_v, lam: lam_v}))
print(f"  predicted eps* at q={q_v}, lam={lam_v}: {es:.5f}")
t_faith = (0.0, np.pi/2)
t_abs = (0.0, np.pi/4)
for e_v in [0.0, es*0.5, es*0.9, es*1.1, es*2, es*4]:
    # global scan
    ts = np.linspace(0, np.pi, 181)
    best = (np.inf, None)
    for t1 in ts:
        for t2 in ts:
            if t2 <= t1: continue
            L = pop_loss_num(t1, t2, lam_v, q_v, p_v, e_v)
            if L < best[0]: best = (L, (t1, t2))
    Lf = pop_loss_num(*t_faith, lam_v, q_v, p_v, e_v)
    La = pop_loss_num(*t_abs, lam_v, q_v, p_v, e_v)
    (t1b, t2b) = best[1]
    winner = 'ABSORBED' if La < Lf else 'FAITHFUL'
    print(f"  eps={e_v:.5f}: L_faith={Lf:.5f} L_abs={La:.5f} pairwise-winner={winner} | "
          f"global-min L={best[0]:.5f} at angles=({np.degrees(t1b):.1f},{np.degrees(t2b):.1f})deg")
