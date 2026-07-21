"""No-go theorem for pairwise-coherence penalties + corrected eps**(beta) curve.

THEOREM (penalty-form independence). Let P(D) = sum_{i<j} g(<d_i, d_j>) be ANY
penalty depending only on pairwise decoder inner products, with g(0) = min g
(orthogonality is optimal -- true for Frobenius Gram, |cos|, max-cos^2/OrtSAE,
and every coherence penalty in the SAE literature). On the manifold of exactly
orthonormal 2-frames, P is CONSTANT (= g(0) per pair). Hence as beta -> inf the
penalized optimum is decided purely by reconstruction+L1 cost among orthonormal
frames -- the penalty's functional form drops out entirely. The faithful frame
{0deg, 90deg} and the anti-rotated absorbed frame {~-45deg, ~+45deg} are BOTH
orthonormal, so no such penalty, at any strength, can distinguish them.

CONSEQUENCES (computed exactly below, lam=q=p0=0.2):
 1. eps**(inf) = 0.0159: the beta->inf global optimum flips from anti-rotated
    to faithful at this child-solo rate -- a penalty-form-INDEPENDENT constant.
    (An earlier report version stated 0.0224; that number was an artifact of a
    coordinate-descent search stuck on the orthogonal valley -- coordinate moves
    break orthogonality and are blocked at large beta, so the search could not
    slide along the valley to the true optimum. Fixed here with joint-rotation
    refinement moves and full-360 valley scans; sanity-checked against known
    configurations.)
 2. The anti-rotated configuration persists as a strict LOCAL optimum of the
    orthogonal-frame family up to eps ~ 0.05-0.06 -- i.e., essentially all the
    way to the ORIGINAL unpenalized boundary eps* = 0.0486. An infinitely
    strong penalty improves the global optimum 3x (0.0486 -> 0.0159) but a
    gradient-trained SAE can remain absorbed nearly as long as with no penalty
    at all, matching the observed GPU multistability.
 3. Corrected finite-beta boundary eps**(beta), via true global 2D scans
    (printed below; the earlier 8-16*beta* tail was inflated by the same
    valley-stuck artifact).

Pure python, no dependencies. Runtime ~2-4 min.
"""
import math, itertools

LAM, Q, P0 = 0.2, 0.2, 0.2
BSTAR = LAM * Q * (8 - 4 * math.sqrt(2) - LAM) / 2
EPS_STAR = LAM * Q * (8 - 4 * math.sqrt(2) - LAM) / (2 * (1 - (2 - math.sqrt(2)) * LAM))

def dirv(deg):
    r = math.radians(deg)
    return (math.cos(r), math.sin(r))

def dot(a, b): return a[0] * b[0] + a[1] * b[1]

def event_loss(x, D, lam):
    """min_{f>=0} ||x - Df||^2 + lam*sum f, exact active-set enumeration."""
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

def loss(angles, eps, beta):
    D = [dirv(a) for a in angles]
    events = [(Q, (1.0, 1.0)), (P0, (1.0, 0.0)), (eps, (0.0, 1.0))]
    L = sum(pr * event_loss(x, D, LAM) for pr, x in events)
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            L += beta * dot(D[i], D[j]) ** 2
    return L

def refine(v, eps, beta, span=2.0):
    """Shrinking-span refinement with single-coordinate AND joint-rotation moves
    (joint moves slide along the orthogonal valley coordinate descent gets stuck on)."""
    L = loss(v, eps, beta)
    s = span
    for _ in range(10):
        moves = ([(s * k / 10, 0) for k in range(-10, 11)]
                 + [(0, s * k / 10) for k in range(-10, 11)]
                 + [(s * k / 10, s * k / 10) for k in range(-10, 11)])
        for da, db in moves:
            cand = [v[0] + da, v[1] + db]
            Lc = loss(cand, eps, beta)
            if Lc < L: L, v = Lc, cand
        s /= 3
    return v, L

def global_opt(eps, beta):
    """True global 2D scan (1.5deg grid, full range) + valley-aware refinement."""
    best, bL = None, None
    a1 = -135.0
    while a1 < 135.0:
        a2 = a1 + 1.5
        while a2 < a1 + 180.0 and a2 < 226.0:
            L = loss([a1, a2], eps, beta)
            if bL is None or L < bL: bL, best = L, [a1, a2]
            a2 += 1.5
        a1 += 1.5
    return refine(best, eps, beta)

def classify(v):
    """faithful if the child-side (larger) angle is >57deg above the parent-side."""
    a1, a2 = sorted(v)
    return "faithful" if a2 > 57 else "anti/absorbed"

def bisect_boundary(beta, lo=0.0, hi=0.06, iters=13):
    for _ in range(iters):
        mid = (lo + hi) / 2
        v, _ = global_opt(mid, beta)
        if classify(v) == "faithful": hi = mid
        else: lo = mid
    v, _ = global_opt(hi, beta)
    return hi, v

if __name__ == "__main__":
    print(f"lam={LAM} q={Q} p0={P0}  vanilla eps*={EPS_STAR:.4f}  beta*={BSTAR:.4f}")
    print()
    print("SANITY CHECKS (known values):")
    print(f"  L([0,90],  eps=.005, b=0) = {loss([0,90],.005,0):.6f}   (expect 0.114950)")
    print(f"  L([-45,45], eps=0,   b=0) = {loss([-45,45],0,0):.6f}   (expect 0.107137)")
    print()
    print("Corrected eps**(beta) -- global-scan boundary:")
    for mult in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 1e4):
        e, v = bisect_boundary(mult * BSTAR)
        print(f"  beta={mult:>7g}*b*: eps** = {e:.4f}   opt at boundary = "
              f"[{v[0]:7.2f},{v[1]:7.2f}]")
    print()
    print("beta->inf limit (exact orthogonal-frame valley scan, penalty-form independent):")
    best_t, bL, results = None, None, {}
    for eps in (0.0150, 0.0155, 0.0159, 0.0165, 0.0175):
        bt, bLL = None, None
        t = -180.0
        while t < 180.0:
            L = loss([t, t + 90], eps, 0.0)   # penalty term = 0 on orthogonal frames
            if bLL is None or L < bLL: bLL, bt = L, t
            t += 0.05
        print(f"  eps={eps:.4f}: global orthogonal-frame opt = [{bt:7.2f},{bt+90:7.2f}]"
              f"  ({classify([bt, bt+90])})")
    print()
    print("Anti-rotated LOCAL-trap persistence ON the orthogonal valley")
    print("(the operative constraint at large beta: motion is along [t, t+90]):")
    for eps in (0.02, 0.03, 0.0486, 0.055, 0.06):
        # 1-D local search in the valley coordinate t, seeded in the anti basin
        bt, bL = None, None
        for k in range(-120, 121):
            t = -44 + 6 * k / 120
            L = loss([t, t + 90], eps, 0.0)
            if bL is None or L < bL: bL, bt = L, t
        s = 0.05
        for _ in range(6):
            s /= 3
            for k in range(-10, 11):
                t = bt + s * k / 10
                L = loss([t, t + 90], eps, 0.0)
                if L < bL: bL, bt = L, t
        Ll = loss([bt - 3, bt - 3 + 90], eps, 0.0)
        Lr = loss([bt + 3, bt + 3 + 90], eps, 0.0)
        trap = bL < Ll and bL < Lr and classify([bt, bt + 90]) != "faithful"
        print(f"  eps={eps:.4f}: valley-local opt [{bt:7.2f},{bt+90:7.2f}]  "
              f"still an absorbed local trap: {trap}")
