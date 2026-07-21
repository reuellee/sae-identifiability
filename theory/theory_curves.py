"""Compute theory curves for the absorption experiment.

For each (lam, q): sweep eps on a fine grid; find the GLOBAL optimum over all
unit-norm 2-latent dictionaries (angles t1, t2) of the population SAE loss in
the pair plane; record the child-side angle phi = max(t1, t2) in degrees.
Also record the pairwise threshold eps*(lam, q).
Output: theory_curves.csv
"""
import numpy as np, csv, math

def pop_loss(t1, t2, lam, q, p, e):
    U = np.array([[np.cos(t1), np.cos(t2)], [np.sin(t1), np.sin(t2)]])
    tot = 0.0
    for w, x in [(q, np.array([1., 1.])), (p, np.array([1., 0.])), (e, np.array([0., 1.]))]:
        if w == 0.0:
            continue
        best = x @ x
        for act in [(0,), (1,), (0, 1)]:
            Ua = U[:, list(act)]
            A_ = Ua.T @ Ua
            det = np.linalg.det(A_) if len(act) == 2 else A_[0, 0]
            if abs(det) < 1e-10:
                continue
            f = np.linalg.solve(A_, Ua.T @ x - lam / 2)
            if (f >= -1e-12).all():
                r = x - Ua @ f
                best = min(best, r @ r + lam * f.sum())
        tot += w * best
    return tot

def global_opt_angle(lam, q, p, e):
    ts1 = np.deg2rad(np.arange(-15, 30, 1.0))
    ts2 = np.deg2rad(np.arange(30, 100, 1.0))
    best = (np.inf, 0, 0)
    for t1 in ts1:
        for t2 in ts2:
            L = pop_loss(t1, t2, lam, q, p, e)
            if L < best[0]:
                best = (L, t1, t2)
    # local refine at 0.1 deg
    _, b1, b2 = best
    ts1 = b1 + np.deg2rad(np.arange(-1.5, 1.5, 0.1))
    ts2 = b2 + np.deg2rad(np.arange(-1.5, 1.5, 0.1))
    for t1 in ts1:
        for t2 in ts2:
            L = pop_loss(t1, t2, lam, q, p, e)
            if L < best[0]:
                best = (L, t1, t2)
    return np.degrees(max(best[1], best[2]))

def eps_star(lam, q):
    s2 = math.sqrt(2)
    return lam * q * (8 - 4 * s2 - lam) / (2 * (1 - (2 - s2) * lam))

p = 0.2
rows = []
for q, lam in [(0.2, 0.05), (0.2, 0.1), (0.2, 0.2), (0.2, 0.3), (0.1, 0.2)]:
    es = eps_star(lam, q)
    eps_grid = sorted(set(
        list(np.linspace(0, 0.2, 41)) + list(np.linspace(0, 3 * es, 31))))
    for e in eps_grid:
        phi = global_opt_angle(lam, q, p, e)
        rows.append(dict(q=q, lam=lam, eps=round(float(e), 6),
                         phi_theory=round(float(phi), 2),
                         eps_star=round(es, 6)))
    print(f"lam={lam} q={q} eps*={es:.4f} done")

with open("theory_curves.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote theory_curves.csv", len(rows), "rows")
