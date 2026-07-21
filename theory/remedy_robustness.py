"""Robustness of the weighted remedy to MISESTIMATED weights (direction 1 theory).

Claim: the weighted boundary is eps*_w = (w_j/w_c)*eps*(lam,q), so if the
child-solo weight is off by factor k (w_c_hat = w_c/k, e.g. rarity
underestimated k-fold), the boundary degrades only LINEARLY: eps*_w -> k*eps*_w.
With exact inverse-density weights the boundary condition is C(lam)<1 (eps-free);
with k-misestimated child weight it becomes: absorbed iff eps < k'*(...)*eps
-> transition eliminated iff C(lam) < 1/k_low where k_low = w_c_true/w_c_used.
Graceful degradation: a 2x weight error halves the safety margin, never
catastrophic. Verified numerically below via global scans.
"""
import math, itertools
LAM, Q, P0 = 0.2, 0.2, 0.2
RT2 = math.sqrt(2)
def eps_star(lam, q): return lam*q*(8-4*RT2-lam)/(2*(1-(2-RT2)*lam))
def dirv(deg):
    r = math.radians(deg); return (math.cos(r), math.sin(r))
def dot(a,b): return a[0]*b[0]+a[1]*b[1]
def event_loss(x, D, lam):
    best = dot(x,x); m = len(D)
    for r in range(1, m+1):
        for S in itertools.combinations(range(m), r):
            if r == 1:
                d = D[S[0]]; f0 = dot(d,x)-lam/2
                if f0 > 0:
                    rx = (x[0]-f0*d[0], x[1]-f0*d[1])
                    best = min(best, dot(rx,rx)+lam*f0)
            else:
                d1,d2 = D[S[0]],D[S[1]]; g = dot(d1,d2); det = 1-g*g
                if abs(det) < 1e-12: continue
                b1,b2 = dot(d1,x)-lam/2, dot(d2,x)-lam/2
                f1 = (b1-g*b2)/det; f2 = (b2-g*b1)/det
                if f1 > 0 and f2 > 0:
                    rx = (x[0]-f1*d1[0]-f2*d2[0], x[1]-f1*d1[1]-f2*d2[1])
                    best = min(best, dot(rx,rx)+lam*(f1+f2))
    return best
def wloss(angles, eps, k_err):
    """inverse-density weights but child weight underestimated by factor k_err."""
    D = [dirv(a) for a in angles]
    wj, wp, wc = 1.0/Q, 1.0/P0, (1.0/eps)/k_err if eps > 0 else 0.0
    ev = [(Q*wj,(1.0,1.0)), (P0*wp,(1.0,0.0)), (eps*wc,(0.0,1.0))]
    return sum(p*event_loss(x, D, LAM) for p,x in ev)
def gopt(eps, k_err):
    best, bL = None, None
    for a1 in range(-135, 136, 2):
        for a2 in range(a1+2, 226, 2):
            L = wloss([a1,a2], eps, k_err)
            if bL is None or L < bL: bL, best = L, [a1,a2]
    return best
print(f"vanilla eps* = {eps_star(LAM,Q):.4f}; exact inverse-density: transition eliminated (C={LAM*(8-4*RT2-LAM)/(2*(1-(2-RT2)*LAM)):.3f} < 1)")
print("child-weight underestimated by factor k: predicted boundary = k-dependent")
print("condition: absorbed iff C(lam)*k > 1  ->  critical k* = 1/C(lam) =", f"{2*(1-(2-RT2)*LAM)/(LAM*(8-4*RT2-LAM)):.2f}")
for k in (1, 2, 4, 4.12, 5, 8):
    labels = []
    for eps in (0.001, 0.005, 0.02):
        v = gopt(eps, k)
        a1, a2 = sorted(v)
        labels.append("faithful" if a2 > 57 else "ABSORBED")
    print(f"  k={k:>5}: eps 0.001/0.005/0.02 -> {labels}")
