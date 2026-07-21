"""Constructive remedy: event-weighted reconstruction losses (exact 2D analysis).

The no-go theorem (no_go_theorem.py) shows pairwise-coherence penalties cannot
fix absorption: the faithful and anti-rotated frames are indistinguishable to
any function of decoder inner products. The discriminating information is in the
CODES per event class. This file analyzes the remedy family that uses it:
weight the per-sample loss by w(x) depending on the event.

CLOSED FORM (from the verified event-loss table, Theorem 2):
  weighted loss difference
    L_w(faithful) - L_w(absorbed)
      = w_j*q*[(2-sqrt2)lam - lam^2/4] - w_c*eps*[1/2 - (2-sqrt2)lam/2]
  so absorption is preferred iff  eps < (w_j/w_c) * eps*(lam, q).
  The boundary simply scales by the weight ratio.

INVERSE-DENSITY WEIGHTING (w(event) = 1/P(event)): w_j/w_c = eps/q, and eps
cancels out of its own boundary condition:
  absorbed preferred  iff  C(lam) := lam*(8-4*sqrt2-lam) / (2*(1-(2-sqrt2)*lam)) > 1
  -- INDEPENDENT of eps and q. Solving C(lam)=1:
  lam_crit = [(12-6*sqrt2) - sqrt((12-6*sqrt2)^2 - 8)] / 2  ~ 0.714.
  For all lam < lam_crit, the faithful dictionary beats the absorbed one for
  EVERY eps > 0: the phase transition is eliminated (boundary pushed to 0+).
  At eps = 0 exactly, Theorem 1 still applies (no data distinguishes the
  ontologies) -- the remedy is optimal in the sense that it fixes everything
  that is information-theoretically fixable.

This script verifies all of the above numerically (global scans, all 1- and
2-latent dictionaries), and checks what no closed-form comparison can see:
whether the absorbed/anti basin SURVIVES AS A LOCAL TRAP under weighting.
Pure python, no dependencies.
"""
import math, itertools

LAM, Q, P0 = 0.2, 0.2, 0.2
RT2 = math.sqrt(2)

def eps_star(lam, q):
    return lam * q * (8 - 4 * RT2 - lam) / (2 * (1 - (2 - RT2) * lam))

def lam_crit():
    b = 12 - 6 * RT2
    return (b - math.sqrt(b * b - 8)) / 2

def dirv(deg):
    r = math.radians(deg)
    return (math.cos(r), math.sin(r))

def dot(a, b): return a[0] * b[0] + a[1] * b[1]

def event_loss(x, D, lam):
    best = dot(x, x)
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
                g = dot(d1, d2); det = 1 - g * g
                if abs(det) < 1e-12: continue
                b1, b2 = dot(d1, x) - lam / 2, dot(d2, x) - lam / 2
                f1 = (b1 - g * b2) / det; f2 = (b2 - g * b1) / det
                if f1 > 0 and f2 > 0:
                    rx = (x[0] - f1 * d1[0] - f2 * d2[0],
                          x[1] - f1 * d1[1] - f2 * d2[1])
                    best = min(best, dot(rx, rx) + lam * (f1 + f2))
    return best

def weighted_loss(angles, eps, weights):
    """weights = (w_joint, w_psolo, w_csolo); probabilities still (Q, P0, eps)."""
    D = [dirv(a) for a in angles]
    wj, wp, wc = weights
    events = [(Q * wj, (1.0, 1.0)), (P0 * wp, (1.0, 0.0)), (eps * wc, (0.0, 1.0))]
    return sum(pr * event_loss(x, D, LAM) for pr, x in events)

def inv_density_weights(eps):
    return (1.0 / Q, 1.0 / P0, 1.0 / eps if eps > 0 else 0.0)

def global_opt(eps, weights, step=1.5):
    best, bL = None, None
    a1 = -135.0
    while a1 < 135.0:
        a2 = a1 + step
        while a2 < a1 + 180.0 and a2 < 226.0:
            L = weighted_loss([a1, a2], eps, weights)
            if bL is None or L < bL: bL, best = L, [a1, a2]
            a2 += step
        a1 += step
    # 1-latent configs too
    t = -180.0
    n_lat = 2
    while t < 180.0:
        L = weighted_loss([t], eps, weights)
        if L < bL: bL, best, n_lat = L, [t], 1
        t += 0.5
    # refine (single + joint moves)
    v = best
    s = step
    for _ in range(10):
        if len(v) == 2:
            moves = ([(s * k / 10, 0) for k in range(-10, 11)]
                     + [(0, s * k / 10) for k in range(-10, 11)]
                     + [(s * k / 10, s * k / 10) for k in range(-10, 11)])
            for da, db in moves:
                cand = [v[0] + da, v[1] + db]
                L = weighted_loss(cand, eps, weights)
                if L < bL: bL, v = L, cand
        else:
            for k in range(-10, 11):
                cand = [v[0] + s * k / 10]
                L = weighted_loss(cand, eps, weights)
                if L < bL: bL, v = L, cand
        s /= 3
    return v, bL, len(v)

def classify(v):
    if len(v) == 1: return "merged/single"
    a1, a2 = sorted(v)
    return "faithful" if a2 > 57 else "absorbed/anti"

def absorbed_is_local_min(eps, weights):
    """Is there still a local min near the absorbed config [0,45] under weighting?
    Local search seeded there; trap = converges to an absorbed-class config that
    beats its neighborhood."""
    v = [0.0, 45.0]
    L = weighted_loss(v, eps, weights)
    s = 3.0
    for _ in range(10):
        moves = ([(s * k / 10, 0) for k in range(-10, 11)]
                 + [(0, s * k / 10) for k in range(-10, 11)]
                 + [(s * k / 10, s * k / 10) for k in range(-10, 11)])
        for da, db in moves:
            cand = [v[0] + da, v[1] + db]
            Lc = weighted_loss(cand, eps, weights)
            if Lc < L: L, v = Lc, cand
        s /= 3
    return classify(v), [round(a, 2) for a in v]

if __name__ == "__main__":
    print(f"lam={LAM} q={Q} p0={P0}   vanilla eps* = {eps_star(LAM, Q):.4f}")
    print(f"lam_crit (closed form) = {lam_crit():.4f}  "
          f"(inverse-density weighting eliminates the transition for lam < lam_crit)")
    C = LAM * (8 - 4 * RT2 - LAM) / (2 * (1 - (2 - RT2) * LAM))
    print(f"C(lam={LAM}) = {C:.4f}  ({'<1: transition ELIMINATED' if C < 1 else '>1: transition survives'})")
    print()
    print("Global optimum under INVERSE-DENSITY weighting, eps sweep")
    print("(prediction: faithful for every eps > 0, however small):")
    for eps in (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02):
        v, L, n = global_opt(eps, inv_density_weights(eps))
        print(f"  eps={eps:.4f}: opt {classify(v):>14} at "
              f"{[round(a,2) for a in v]}  (n_latents={n})")
    print()
    print("Control: same scan UNWEIGHTED (w=1,1,1) -- transition should persist near eps*~0.049:")
    for eps in (0.001, 0.01, 0.03, 0.0486, 0.06):
        v, L, n = global_opt(eps, (1.0, 1.0, 1.0))
        print(f"  eps={eps:.4f}: opt {classify(v):>14} at {[round(a,2) for a in v]}")
    print()
    print("TRAP CHECK under inverse-density weighting: does an absorbed local min survive?")
    for eps in (0.0005, 0.001, 0.005, 0.01, 0.02):
        cls, v = absorbed_is_local_min(eps, inv_density_weights(eps))
        print(f"  eps={eps:.4f}: local search from absorbed converges to "
              f"{cls:>14} at {v}  ({'TRAP SURVIVES' if cls != 'faithful' else 'trap ELIMINATED'})")
    print()
    print("Boundary scaling law check: eps*_w = (w_j/w_c) * eps* for a few fixed ratios:")
    for ratio in (0.5, 0.25, 0.1):
        pred = ratio * eps_star(LAM, Q)
        lo, hi = 0.0, 0.06
        for _ in range(14):
            mid = (lo + hi) / 2
            v, _, _ = global_opt(mid, (ratio, 1.0, 1.0), step=2.0)
            if classify(v) == "faithful": hi = mid
            else: lo = mid
        print(f"  w_j/w_c={ratio}: predicted boundary {pred:.4f}, scanned {hi:.4f}")
