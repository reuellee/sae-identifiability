"""Exact analysis: Matryoshka needs breadth (multi-child) to beat absorption.

Feature space: orthonormal a_p, a_c1, a_c2 (3D coords).
Events: {p,c1} w.p. q; {p,c2} w.p. q; {p} w.p. p0; {c1} w.p. eps; {c2} w.p. eps.
Candidate dictionaries (unit columns):
  FAITH3 : {a_p, a_c1, a_c2}
  ABS2   : {m1, m2},  m_i = (a_p + a_ci)/sqrt(2)
  ABS2+P : {a_p, m1, m2}
Objectives (per sample, code f >= 0 shared across nested prefixes):
  vanilla:    ||x - D f||^2 + lam ||f||_1
  matryoshka: sum over prefixes s in S of ||x - D_{:s} f_{:s}||^2 + lam ||f||_1
              prefix sizes: {1, m} (and {1,2,m} for 3-latent dicts) — report best
              ORDERING and best prefix scheme per dictionary (SGD would find it).
Exact per-event solve: convex QP in f >= 0 -> enumerate active sets (<= 2^3).
Also: single-child sanity replication, and coherence-penalty check in 2-child model.
"""
import numpy as np, itertools, math

def solve_event(x, U, prefixes, lam):
    """min over f>=0 of sum_s ||x - U[:, :s] f[:s]||^2 + lam*sum(f). Exact via
    active sets. Quadratic form: sum_s (f_a' G_s f_a - 2 b_s' f_a) + const."""
    m = U.shape[1]
    best = None
    for act in itertools.chain.from_iterable(
            itertools.combinations(range(m), r) for r in range(m + 1)):
        a = list(act)
        # build quadratic over active coords
        G = np.zeros((len(a), len(a))); b = np.zeros(len(a)); c0 = 0.0
        for s in prefixes:
            Us = U[:, :s]
            idx = [i for i, j in enumerate(a) if j < s]
            cols = [a[i] for i in idx]
            c0 += x @ x
            if not cols:
                continue
            Ua = Us[:, [j for j in cols]]
            Gi = Ua.T @ Ua
            bi = Ua.T @ x
            for r_, i_ in enumerate(idx):
                b[i_] += bi[r_]
                for r2, i2 in enumerate(idx):
                    G[i_, i2] += Gi[r_, r2]
        if a:
            try:
                f = np.linalg.solve(2 * G, 2 * b - lam)
            except np.linalg.LinAlgError:
                continue
            if (f < -1e-11).any():
                continue
            val = c0 - 2 * b @ f + f @ G @ f + lam * f.sum()
        else:
            val = c0
        if best is None or val < best:
            best = val
    return best

def total_loss(U, events, lam, prefix_scheme):
    return sum(w * solve_event(x, U, prefix_scheme, lam) for w, x in events)

def best_arrangement(cols, events, lam, matryoshka):
    """min over column orderings and (if matryoshka) prefix schemes."""
    m = len(cols)
    best = np.inf; arg = None
    schemes = [[m]] if not matryoshka else \
              ([[1, m]] if m == 2 else [[1, m], [2, m], [1, 2, m]])
    for perm in itertools.permutations(range(m)):
        U = np.stack([cols[i] for i in perm], axis=1)
        for sch in schemes:
            L = total_loss(U, events, lam, sch)
            if L < best:
                best, arg = L, (perm, tuple(sch))
    return best, arg

e = np.eye(3)
a_p, a_c1, a_c2 = e[0], e[1], e[2]
m1 = (a_p + a_c1) / math.sqrt(2)
m2 = (a_p + a_c2) / math.sqrt(2)
DICTS = {
    "FAITH3": [a_p, a_c1, a_c2],
    "ABS2": [m1, m2],
    "ABS2+P": [a_p, m1, m2],
}

def events2(q, p0, eps):
    return [(q, a_p + a_c1), (q, a_p + a_c2), (p0, a_p),
            (eps, a_c1), (eps, a_c2)]

q, p0 = 0.2, 0.2
print("=== two-child model:  L per dictionary (best arrangement) ===")
print(f"{'lam':>5} {'eps':>6} | {'objective':>10} | " +
      " ".join(f"{k:>10}" for k in DICTS) + "   winner")
for lam in (0.1, 0.2):
    for eps in (0.0, 0.01, 0.03):
        ev = events2(q, p0, eps)
        for mat in (False, True):
            vals = {}
            args = {}
            for k, cols in DICTS.items():
                vals[k], args[k] = best_arrangement(cols, ev, lam, mat)
            win = min(vals, key=vals.get)
            tag = "matryoshka" if mat else "vanilla"
            print(f"{lam:>5} {eps:>6} | {tag:>10} | " +
                  " ".join(f"{vals[k]:>10.5f}" for k in DICTS) +
                  f"   {win}  {args[win]}")

print("\n=== single-child replication (sanity vs symbolic results) ===")
def events1(q, p0, eps):
    return [(2*q, a_p + a_c1), (p0, a_p), (eps, a_c1)]
D1 = {"FAITH2": [a_p, a_c1], "ABS1": [(a_p + a_c1) / math.sqrt(2), a_p]}
for lam in (0.1,):
    for eps in (0.0, 0.01, 0.03, 0.06):
        ev = events1(q, p0, eps)   # 2q joint mass to match total pair mass
        for mat in (False, True):
            vals = {k: best_arrangement(c, ev, lam, mat)[0] for k, c in D1.items()}
            win = min(vals, key=vals.get)
            print(f"lam={lam} eps={eps} {'mat' if mat else 'van'}: "
                  + " ".join(f"{k}={v:.5f}" for k, v in vals.items()) + f" -> {win}")

print("\n=== coherence penalty in two-child model (vanilla objective + beta*Gram^2) ===")
def pen(cols, beta):
    U = np.stack(cols, axis=1)
    Gr = U.T @ U - np.eye(U.shape[1])
    return beta * (Gr ** 2).sum() / 2
for lam in (0.2,):
    bstar = lam * q * (8 - 4 * math.sqrt(2) - lam) / 2
    for beta in (0.0, bstar, 3 * bstar):
        for eps in (0.0, 0.01):
            ev = events2(q, p0, eps)
            vals = {k: best_arrangement(c, ev, lam, False)[0] + pen(c, beta)
                    for k, c in DICTS.items()}
            win = min(vals, key=vals.get)
            print(f"lam={lam} beta={beta:.4f} eps={eps}: " +
                  " ".join(f"{k}={v:.5f}" for k, v in vals.items()) + f" -> {win}")
