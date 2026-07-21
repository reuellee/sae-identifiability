"""Exact 2D theory EXTENDED to variable latent count, with coherence penalty.

Round-1 theory compared only fixed 2-latent dictionaries ({p,c} faithful vs
{p,comp} absorbed) and derived beta* as their crossover. Experiment C1 refuted
the prediction: at beta >= beta* SGD keeps phi~40deg with ONE in-plane latent
(rho~0.98, delocalization ruled out). Hypothesis: the penalty is evaded by
MERGING — a single compromise latent between parent (0deg) and composite (45deg)
pays zero pairwise penalty. This script computes the exact population loss of
every 1- and 2-latent in-plane dictionary (0.25deg grid + local refinement) and
reports the global optimum config vs (eps, beta), to compare against C1/C2 data.

Pure python (no numpy) — runs anywhere. Events: joint q: x=a_p+a_c;
parent-solo p0: x=a_p; child-solo eps: x=a_c. lam=q=p0=0.2 (Experiment C setup).
"""
import math, itertools

LAM, Q, P0 = 0.2, 0.2, 0.2
BSTAR = LAM * Q * (8 - 4 * math.sqrt(2) - LAM) / 2

def dirv(deg):
    r = math.radians(deg)
    return (math.cos(r), math.sin(r))

def dot(a, b): return a[0] * b[0] + a[1] * b[1]

def event_loss(x, D, lam):
    """min over f>=0 of ||x - D f||^2 + lam*sum(f), by active-set enumeration."""
    best = dot(x, x)                                   # f = 0
    m = len(D)
    for r in range(1, m + 1):
        for S in itertools.combinations(range(m), r):
            if r == 1:
                d = D[S[0]]
                f0 = dot(d, x) - lam / 2
                if f0 > 0:
                    rx = (x[0] - f0 * d[0], x[1] - f0 * d[1])
                    best = min(best, dot(rx, rx) + lam * f0)
            else:
                d1, d2 = D[S[0]], D[S[1]]
                g = dot(d1, d2)
                det = 1 - g * g
                if abs(det) < 1e-12: continue
                b1, b2 = dot(d1, x) - lam / 2, dot(d2, x) - lam / 2
                f1 = (b1 - g * b2) / det
                f2 = (b2 - g * b1) / det
                if f1 > 0 and f2 > 0:
                    rx = (x[0] - f1 * d1[0] - f2 * d2[0],
                          x[1] - f1 * d1[1] - f2 * d2[1])
                    best = min(best, dot(rx, rx) + lam * (f1 + f2))
    return best

def total_loss(angles, eps, beta):
    D = [dirv(a) for a in angles]
    events = [(Q, (1.0, 1.0)), (P0, (1.0, 0.0)), (eps, (0.0, 1.0))]
    L = sum(pr * event_loss(x, D, LAM) for pr, x in events)
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            L += beta * dot(D[i], D[j]) ** 2
    return L

def refine(f, lo_hi_list, steps=3, factor=5.0):
    """Coordinate-refine an argmin over a list of (val, lo, hi) angle params."""
    vals = [v for v, _, _ in lo_hi_list]
    span = [(hi - lo) for _, lo, hi in lo_hi_list]
    for _ in range(steps):
        span = [s / factor for s in span]
        for i in range(len(vals)):
            grid = [vals[i] + span[i] * (k / 10.0 - 1) for k in range(21)]
            vals[i] = min(grid, key=lambda g: f([*vals[:i], g, *vals[i+1:]]))
    return vals

def global_opt(eps, beta):
    # 1-latent scan (full range)
    g1 = min((a / 4 for a in range(-360, 541)),
             key=lambda a: total_loss([a], eps, beta))
    g1 = refine(lambda v: total_loss(v, eps, beta), [(g1, g1 - 1, g1 + 1)])
    L1 = total_loss(g1, eps, beta)
    # 2-latent scan (coarse 2deg, FULL range incl. negative/anti-rotated) + refine
    best2, L2 = None, float("inf")
    for a1 in range(-90, 136, 2):
        for a2 in range(a1 + 2, 136, 2):
            L = total_loss([a1, a2], eps, beta)
            if L < L2: L2, best2 = L, [a1, a2]
    best2 = refine(lambda v: total_loss(v, eps, beta),
                   [(best2[0], 0, 0), (best2[1], 0, 0)])
    L2 = total_loss(best2, eps, beta)
    if L1 <= L2 + 1e-9:
        return 1, [round(a, 1) for a in g1], L1, L2 - L1
    return 2, [round(a, 1) for a in best2], L2, L1 - L2

def compare_sgd_configs(eps, beta):
    """Exact losses of the three configurations SGD actually finds."""
    anti = refine(lambda v: total_loss(v, eps, beta), [(-37, 0, 0), (45, 0, 0)])
    merged = refine(lambda v: total_loss(v, eps, beta), [(34, 0, 0)])
    faith = refine(lambda v: total_loss(v, eps, beta), [(5, 0, 0), (85, 0, 0)])
    return (total_loss(anti, eps, beta), [round(a,1) for a in anti],
            total_loss(merged, eps, beta), [round(a,1) for a in merged],
            total_loss(faith, eps, beta), [round(a,1) for a in faith])

print(f"lam={LAM} q={Q} p0={P0}  beta* (2-latent crossover) = {BSTAR:.4f}")
print(f"{'eps':>5} {'beta/b*':>7} | {'n_lat':>5} {'angles':>16} {'margin':>8} | C1 experiment")
expt = {  # (eps, beta_mult) -> (phi_child_mean, n_inplane_mean) from results_remedies.csv
    (0.00, 0.0): (45.3, 2.0), (0.00, 0.5): (45.0, 2.0), (0.00, 1.0): (39.9, 1.5),
    (0.00, 2.0): (39.6, 1.5), (0.00, 4.0): (39.5, 1.5),
    (0.01, 0.0): (45.8, 2.0), (0.01, 0.5): (54.6, 2.0), (0.01, 1.0): (40.4, 1.5),
    (0.01, 2.0): (40.2, 1.5), (0.01, 4.0): (65.9, 2.0),
    (0.02, 0.0): (46.6, 2.0), (0.02, 0.5): (72.7, 2.0), (0.02, 1.0): (78.4, 2.0),
    (0.02, 2.0): (65.7, 2.0), (0.02, 4.0): (66.8, 2.0),
}
for eps in (0.0, 0.01, 0.02):
    for bm in (0.0, 0.5, 1.0, 2.0, 4.0):
        n, ang, L, margin = global_opt(eps, bm * BSTAR)
        e = expt.get((round(eps, 2), bm), ("?", "?"))
        print(f"{eps:5.2f} {bm:7.1f} | {n:5d} {str(ang):>16} {margin:8.4f} | "
              f"phi={e[0]} n_inplane={e[1]}")

print()
print("=== exact losses of the three configs SGD finds (basin comparison) ===")
print(f"{'eps':>5} {'beta/b*':>7} | {'anti(-37,45)':>22} {'merged(34)':>16} {'faithful(5,85)':>20}")
for eps in (0.0, 0.01, 0.02):
    for bm in (1.0, 2.0, 4.0):
        La, aa, Lm, am, Lf, af = compare_sgd_configs(eps, bm * BSTAR)
        win = min((La, 'anti'), (Lm, 'merged'), (Lf, 'faithful'))[1]
        print(f"{eps:5.2f} {bm:7.1f} | {La:.4f} @{str(aa):>14} {Lm:.4f} @{str(am):>7} "
              f"{Lf:.4f} @{str(af):>14}  -> {win}")
