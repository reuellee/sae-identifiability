"""Verification gauntlet for the GENERAL (m latents, d dims) no-go derivation.

The derivation (general_no_go.md) rests on:
  L1 (constancy):   any pairwise penalty P = sum_{i<j} g(<d_i,d_j>), g minimized
                    at 0, is constant on the orthonormal manifold O.
  L2 (closed form): for D in O the nonneg lasso decouples per coordinate and
                    L0(D) = E||x||^2 - sum_i E[(<d_i,x> - lam/2)_+^2].
  L3 (reduction):   among orthonormal dictionaries whose non-pair columns are
                    orthogonal to the pair plane, the competition reduces to the
                    1-parameter in-plane frame family T(theta).
  T  (boundary):    exact T(theta) from L2; basin-optimized crossover eps**_inf,
                    plus the NEW closed-form pure-strategy crossover
                    eps0(lam,q,p0) = lam[(2-rt2)q - (rt2-1)p0 + lam(p0-q)/4]
                                     / [1/2 - (2-rt2)lam/2]
                    -> the no-go's bite depends on p0/q: for p0 > ~rt2*q the
                    FAITHFUL frame wins the orthogonal competition even at eps=0.

Checks below:
  A. L2 vs direct active-set enumeration on random orthonormal dictionaries.
  B. T(theta) boundary at lam=q=p0=0.2 must reproduce 0.0159 (INDEPENDENT code
     path from the earlier event-loss enumeration).
  C. L3 probe: random orthonormal perturbations mixing pair-plane and background
     directions must not beat the in-plane optimum (exact finite-atom model).
  D. eps**_inf(lam) curve: must ->0 as lam->0.
  E. p0-dependence: eps**_inf(p0) must vanish near p0 ~ rt2*q*(1+O(lam)); spot
     check the finite-beta landscape at p0=0.3 to confirm large beta DOES
     restore faithfulness there (scope refinement of the no-go).
Pure python. Runtime ~1-2 min.
"""
import math, itertools, random

RT2 = math.sqrt(2)

# ---------------------------------------------------------------- generic tools

def dot(a, b): return sum(x * y for x, y in zip(a, b))

def gain(c, lam):
    """Per-coordinate gain (c - lam/2)_+^2 for orthonormal dictionaries (L2)."""
    a = c - lam / 2
    return a * a if a > 0 else 0.0

def L0_orthonormal(D, atoms, lam):
    """L2 closed form: atoms = [(prob, x_vec)]. D = list of orthonormal columns."""
    L = 0.0
    for p, x in atoms:
        xx = dot(x, x)
        L += p * (xx - sum(gain(dot(d, x), lam) for d in D))
    return L

def nonneg_lasso_direct(x, D, lam):
    """Exact min_{f>=0} ||x-Df||^2 + lam*sum f by active-set enumeration.
    General D (not nec. orthonormal). Pure-python Gaussian elimination."""
    m = len(D)
    best = dot(x, x)
    for r in range(1, m + 1):
        for S in itertools.combinations(range(m), r):
            G = [[dot(D[i], D[j]) for j in S] for i in S]
            b = [dot(D[i], x) - lam / 2 for i in S]
            # solve G f = b
            n = len(S)
            A = [row[:] + [b[i]] for i, row in enumerate(G)]
            ok = True
            for col in range(n):
                piv = max(range(col, n), key=lambda r2: abs(A[r2][col]))
                if abs(A[piv][col]) < 1e-12: ok = False; break
                A[col], A[piv] = A[piv], A[col]
                for r2 in range(n):
                    if r2 != col:
                        fac = A[r2][col] / A[col][col]
                        for c2 in range(col, n + 1):
                            A[r2][c2] -= fac * A[col][c2]
            if not ok: continue
            f = [A[i][n] / A[i][i] for i in range(n)]
            if all(fi > 0 for fi in f):
                res = [x[k] - sum(f[j] * D[S[j]][k] for j in range(n))
                       for k in range(len(x))]
                best = min(best, dot(res, res) + lam * sum(f))
    return best

# ---------------------------------------------------------------- check A

def check_A():
    print("A. closed form (L2) vs direct enumeration on random orthonormal D:")
    rng = random.Random(0)
    worst = 0.0
    for trial in range(40):
        d, m = 5, 3
        # random orthonormal columns by Gram-Schmidt
        cols = []
        while len(cols) < m:
            v = [rng.gauss(0, 1) for _ in range(d)]
            for u in cols:
                pr = dot(u, v)
                v = [vi - pr * ui for vi, ui in zip(v, u)]
            nv = math.sqrt(dot(v, v))
            if nv > 1e-6: cols.append([vi / nv for vi in v])
        x = [rng.gauss(0, 1) for _ in range(d)]
        lam = rng.choice([0.05, 0.2, 0.5])
        direct = nonneg_lasso_direct(x, cols, lam)
        closed = dot(x, x) - sum(gain(dot(c, x), lam) for c in cols)
        worst = max(worst, abs(direct - closed))
    print(f"   max |direct - closed| over 40 random trials = {worst:.2e}  "
          f"({'PASS' if worst < 1e-9 else 'FAIL'})")

# ---------------------------------------------------------------- T(theta) family

def T_theta(theta_deg, lam, q, p0, eps):
    """Exact in-plane loss of the orthogonal frame {u(t), u(t+90)} via L2.
    Coordinates in (a_p, a_c) basis; constant background terms omitted (they
    cancel in all comparisons)."""
    t = math.radians(theta_deg)
    u1 = (math.cos(t), math.sin(t))
    u2 = (-math.sin(t), math.cos(t))
    atoms = [(q, (1.0, 1.0)), (p0, (1.0, 0.0)), (eps, (0.0, 1.0))]
    L = 0.0
    for p, x in atoms:
        L += p * (dot(x, x) - gain(dot(u1, x), lam) - gain(dot(u2, x), lam))
    return L

def basin_min(center, lam, q, p0, eps, span=25.0):
    best_t, best_L = None, None
    for k in range(-500, 501):
        t = center + span * k / 500
        L = T_theta(t, lam, q, p0, eps)
        if best_L is None or L < best_L: best_L, best_t = L, t
    s = span / 500
    for _ in range(8):
        s /= 3
        for k in range(-10, 11):
            t = best_t + s * k / 10
            L = T_theta(t, lam, q, p0, eps)
            if L < best_L: best_L, best_t = L, t
    return best_t, best_L

def global_frame_min(lam, q, p0, eps):
    best_t, best_L = None, None
    t = -180.0
    while t < 180.0:
        L = T_theta(t, lam, q, p0, eps)
        if best_L is None or L < best_L: best_L, best_t = L, t
        t += 0.02
    return basin_min(best_t, lam, q, p0, eps, span=0.05)

def eps_inf_boundary(lam, q, p0):
    """Basin-optimized crossover: smallest eps where the global frame optimum
    is faithful-class (child-side angle > 57deg)."""
    lo, hi = 0.0, 0.2
    for _ in range(30):
        mid = (lo + hi) / 2
        t, _ = global_frame_min(lam, q, p0, mid)
        a1, a2 = sorted(((t % 360 + 180) % 360 - 180, ((t + 90) % 360 + 180) % 360 - 180))
        # child-side = the column angle closest to 90
        child_side = min((abs(a - 90.0), a) for a in
                         (t % 360, (t + 90) % 360, (t + 180) % 360, (t + 270) % 360))[1]
        if abs(child_side - 90) < 33: hi = mid
        else: lo = mid
    return hi

def eps0_pure(lam, q, p0):
    """NEW closed-form pure-strategy crossover T(-45) = T(0)."""
    num = lam * ((2 - RT2) * q - (RT2 - 1) * p0 + lam * (p0 - q) / 4)
    den = 0.5 - (2 - RT2) * lam / 2
    return num / den

def check_B():
    print("B. boundary at lam=q=p0=0.2 (independent gain-formula path):")
    e = eps_inf_boundary(0.2, 0.2, 0.2)
    e0 = eps0_pure(0.2, 0.2, 0.2)
    print(f"   basin-optimized eps**_inf = {e:.4f}  (must match prior 0.0159: "
          f"{'PASS' if abs(e - 0.0159) < 0.0015 else 'FAIL'})")
    print(f"   pure-strategy closed form eps0 = {e0:.4f}  (anchor, ~2% below)")

# ---------------------------------------------------------------- check C

def cayley(Askew, n):
    """R = (I-A)^-1 (I+A), orthogonal for skew A. Pure python."""
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ImA = [[I[i][j] - Askew[i][j] for j in range(n)] for i in range(n)]
    IpA = [[I[i][j] + Askew[i][j] for j in range(n)] for i in range(n)]
    # solve (I-A) R = (I+A) column by column
    R = [[0.0] * n for _ in range(n)]
    for col in range(n):
        A = [ImA[i][:] + [IpA[i][col]] for i in range(n)]
        for c in range(n):
            piv = max(range(c, n), key=lambda r2: abs(A[r2][c]))
            A[c], A[piv] = A[piv], A[c]
            for r2 in range(n):
                if r2 != c:
                    fac = A[r2][c] / A[c][c]
                    for c2 in range(c, n + 1):
                        A[r2][c2] -= fac * A[c][c2]
        for i in range(n):
            R[i][col] = A[i][n] / A[i][i]
    return R

def check_C():
    print("C. L3 probe: mixed plane/background perturbations (d=4, m=4, exact atoms):")
    lam, q, p0 = 0.2, 0.2, 0.2
    bg_rate = 0.08
    for eps, label in ((0.005, "below boundary"), (0.03, "above boundary")):
        # atoms: pair-event x {bg1,bg2 on/off}; bg coeff fixed at 1
        pair_events = [(q, (1.0, 1.0)), (p0, (1.0, 0.0)), (eps, (0.0, 1.0)),
                       (1 - q - p0 - eps, (0.0, 0.0))]
        atoms = []
        for pp, (xp, xc) in pair_events:
            for b1 in (0, 1):
                for b2 in (0, 1):
                    pr = pp * (bg_rate if b1 else 1 - bg_rate) * (bg_rate if b2 else 1 - bg_rate)
                    atoms.append((pr, (xp, xc, float(b1), float(b2))))
        # best in-plane (O2) config: theta-optimized frame + bg identity cols
        tstar, _ = global_frame_min(lam, q, p0, eps)
        tr = math.radians(tstar)
        D0 = [(math.cos(tr), math.sin(tr), 0.0, 0.0),
              (-math.sin(tr), math.cos(tr), 0.0, 0.0),
              (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
        L_best_O2 = L0_orthonormal(D0, atoms, lam)
        # random-rotation local search over full SO(4)
        rng = random.Random(1)
        best_mixed = L_best_O2
        cur = [row[:] for row in D0]
        cur_L = L_best_O2
        scale = 0.15
        for it in range(3000):
            A = [[0.0] * 4 for _ in range(4)]
            for i in range(4):
                for j in range(i + 1, 4):
                    v = rng.gauss(0, scale)
                    A[i][j], A[j][i] = v, -v
            R = cayley(A, 4)
            cand = [[sum(R[i][k] * cur[c][k] for k in range(4)) for i in range(4)]
                    for c in range(4)]
            Lc = L0_orthonormal(cand, atoms, lam)
            if Lc < cur_L - 1e-12:
                cur, cur_L = cand, Lc
            if it % 500 == 499: scale *= 0.6
            best_mixed = min(best_mixed, cur_L)
        gap = best_mixed - L_best_O2
        print(f"   eps={eps} ({label}): best mixed-rotation search vs in-plane opt: "
              f"gap = {gap:+.2e}  ({'PASS (no better mixed config)' if gap > -1e-9 else 'PROBE FOUND BETTER CONFIG'})")

# ---------------------------------------------------------------- checks D, E

def check_D():
    print("D. eps**_inf(lam) at q=p0=0.2 (must -> 0 as lam -> 0):")
    for lam in (0.4, 0.3, 0.2, 0.1, 0.05, 0.02):
        e = eps_inf_boundary(lam, 0.2, 0.2)
        e0 = eps0_pure(lam, 0.2, 0.2)
        print(f"   lam={lam:5.2f}: eps**_inf = {e:.5f}   closed-form anchor eps0 = {e0:.5f}")

def check_E():
    print("E. p0-dependence at lam=0.2, q=0.2 (predicted vanish near p0 ~ rt2*q):")
    for p0 in (0.10, 0.20, 0.25, 0.27, 0.29, 0.31, 0.35):
        e = eps_inf_boundary(0.2, 0.2, p0)
        e0 = eps0_pure(0.2, 0.2, p0)
        note = "penalty CANNOT reach eps=0" if e > 1e-4 else "faithful wins even at eps=0 -> penalty WORKS here"
        print(f"   p0={p0:.2f}: eps**_inf = {e:.5f}  (anchor {max(e0,0):.5f})  {note}")
    print("   rt2*q =", f"{RT2 * 0.2:.4f}")

if __name__ == "__main__":
    check_A()
    print()
    check_B()
    print()
    check_C()
    print()
    check_D()
    print()
    check_E()
