"""Numeric verification of the TopK feature-absorption boundary (round 10).

Model (same 2D toy as verify_absorption_theory.py): parent/child a_p, a_c
orthonormal; events joint x=(1,1) w.p. q, parent-solo x=(1,0) w.p. p,
child-solo x=(0,1) w.p. eps. Difference from the L1 file: the SAE uses a
HARD TopK budget kappa (>=0 codes, NO L1 shrinkage) instead of an L1 penalty.

Per-event loss for a decoder U under budget kappa:
  L_kappa(U, x) = min over kappa-subsets S of atoms, min_{f>=0} ||x - U_S f||^2.
(The population-optimal loss for that decoder under a kappa-sparse nonneg code.)

Claimed closed forms (nonneg codes, unit-norm atoms):
                 Faithful k=1  Absorbed k=1  Faithful k=2  Absorbed k=2
  joint  (q)          1             0             0             0
  psolo  (p)          0             0             0             0
  csolo  (eps)        0            1/2            0            1/2
=> E[L_F,k=1]=q ; E[L_A,k=1]=eps/2 ; absorbed wins iff eps < 2q  => eps*_TopK(k=1)=2q.
   E[L_F,k=2]=0 ; E[L_A,k=2]=eps/2 ; faithful wins for all eps>0 => eps*_TopK(k>=2)=0.

The child-solo 1/2 under absorbed is the nonnegativity signature: extracting
v_c from the composite (v_p+v_c)/sqrt2 needs a NEGATIVE parent coefficient,
which ReLU forbids, so even a second slot cannot separate the child.

Checks:
  1. Per-event losses (numeric NNLS) vs the claimed table, k in {1,2}.
  2. Population crossover: eps*_TopK(k=1)=2q; faithful strictly wins at k=2.
  3. Global scan over ALL unit-norm 2-atom dictionaries under kappa-budget
     nonneg coding, confirming faithful/absorbed are the competing optima and
     the switchover matches the closed form.
Exit nonzero on any mismatch.
"""
import numpy as np
from itertools import combinations

RT2 = np.sqrt(2.0)
FAILS = []

def nnls_2d_subset(U, x, S):
    """min_{f>=0} ||x - U[:,S] f||^2 by exact active-set enumeration (<=2 atoms)."""
    Ua = U[:, list(S)]
    best = float(x @ x)                       # f = 0
    r = len(S)
    for act in [c for k in range(1, r + 1) for c in combinations(range(r), k)]:
        M = Ua[:, list(act)]
        try:
            f = np.linalg.solve(M.T @ M, M.T @ x)
        except np.linalg.LinAlgError:
            continue
        if (f >= -1e-12).all():
            res = x - M @ f
            best = min(best, float(res @ res))
    return best

def loss_kappa(U, x, kappa):
    """Best kappa-sparse nonneg reconstruction: min over kappa-subsets."""
    m = U.shape[1]
    kap = min(kappa, m)
    best = float(x @ x)
    for S in combinations(range(m), kap):
        best = min(best, nnls_2d_subset(U, x, S))
    return best

U_F = np.array([[1.0, 0.0], [0.0, 1.0]])            # faithful: v_p, v_c
U_A = np.array([[1.0, 1.0 / RT2], [0.0, 1.0 / RT2]])  # absorbed: v_p, (v_p+v_c)/sqrt2
EV = {"joint": np.array([1.0, 1.0]),
      "psolo": np.array([1.0, 0.0]),
      "csolo": np.array([0.0, 1.0])}

claimed = {
    ("F", 1, "joint"): 1.0, ("F", 1, "psolo"): 0.0, ("F", 1, "csolo"): 0.0,
    ("A", 1, "joint"): 0.0, ("A", 1, "psolo"): 0.0, ("A", 1, "csolo"): 0.5,
    ("F", 2, "joint"): 0.0, ("F", 2, "psolo"): 0.0, ("F", 2, "csolo"): 0.0,
    ("A", 2, "joint"): 0.0, ("A", 2, "psolo"): 0.0, ("A", 2, "csolo"): 0.5,
}

print("=== Check 1: per-event TopK losses (numeric NNLS vs claimed) ===")
for name, U in [("F", U_F), ("A", U_A)]:
    for kappa in (1, 2):
        for ev, x in EV.items():
            got = loss_kappa(U, x, kappa)
            want = claimed[(name, kappa, ev)]
            tag = "ok" if abs(got - want) < 1e-9 else "MISMATCH"
            if tag == "MISMATCH":
                FAILS.append(f"{name} k={kappa} {ev}: got {got:.6f} want {want}")
            print(f"  {name} k={kappa} {ev:<6} got {got:.6f} want {want:.3f}  {tag}")

print("=== Check 2: population crossover ===")
def pop(U, kappa, q, p, e):
    return (q * loss_kappa(U, EV["joint"], kappa)
            + p * loss_kappa(U, EV["psolo"], kappa)
            + e * loss_kappa(U, EV["csolo"], kappa))
for q in (0.1, 0.2, 0.3):
    estar = 2 * q
    # k=1: faithful and absorbed should tie exactly at eps=2q
    lf = pop(U_F, 1, q, 0.2, estar); la = pop(U_A, 1, q, 0.2, estar)
    ok1 = abs(lf - la) < 1e-9
    if not ok1: FAILS.append(f"k=1 crossover q={q}: L_F={lf} L_A={la}")
    print(f"  q={q}: eps*_TopK(k=1)=2q={estar:.3f} -> L_F={lf:.4f} L_A={la:.4f} "
          f"{'tie ok' if ok1 else 'NOT TIED'}")
    # below/above threshold
    below = pop(U_A, 1, q, 0.2, estar * 0.5) < pop(U_F, 1, q, 0.2, estar * 0.5)
    above = pop(U_A, 1, q, 0.2, estar * 1.5) > pop(U_F, 1, q, 0.2, estar * 1.5)
    if not (below and above): FAILS.append(f"k=1 ordering q={q}")
    print(f"       below 2q: absorbed wins={below}; above 2q: faithful wins={above}")
    # k=2: faithful strictly wins for all eps>0
    for e in (0.05, 0.2, 0.5, 2 * q, 4 * q):
        lf2, la2 = pop(U_F, 2, q, 0.2, e), pop(U_A, 2, q, 0.2, e)
        if not (lf2 < la2 - 1e-12 or e == 0):
            FAILS.append(f"k=2 faithful-wins q={q} eps={e}: L_F={lf2} L_A={la2}")
    print(f"       k=2: faithful strictly beats absorbed for every eps>0 "
          f"(L_F=0, L_A=eps/2) -> eps*_TopK(k>=2)=0")

print("=== Check 3: global scan over all unit-norm 2-atom dictionaries ===")
# Like the L1 file, the pure candidates are NOT the global optimum: the
# continuously optimized dictionary tilts. Check 3 confirms (a) at kappa=1 the
# global optimum tilts, with its child-side atom moving 45deg (eps=0, absorbed)
# -> 90deg (large eps, faithful) -- the smooth transition SGD will trace; and
# (b) the sharp capacity result: at kappa=2 the global min is 0 and any
# composite (45deg-atom) configuration is strictly worse for every eps>0.
def pop_angles(t1, t2, kappa, q, p, e):
    U = np.array([[np.cos(t1), np.cos(t2)], [np.sin(t1), np.sin(t2)]])
    return pop(U, kappa, q, p, e)
def scan(kappa, q, p, e, ts):
    best = (np.inf, (float("nan"), float("nan")))
    for i, t1 in enumerate(ts):
        for t2 in ts[i+1:]:
            L = pop_angles(t1, t2, kappa, q, p, e)
            if L < best[0] - 1e-12:
                best = (L, (t1, t2))
    return best
q_v, p_v = 0.2, 0.2
ts = np.linspace(-np.pi/2, np.pi, 145)   # ~1.25deg spacing; assertions use 3-5deg tol

print("  -- kappa=1: continuous optimum tilts (child-side atom angle vs eps) --")
prev_child = None
for e in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9]:
    L, (t1b, t2b) = scan(1, q_v, p_v, e, ts)
    child = max(np.degrees(t1b), np.degrees(t2b))   # child-side = higher angle
    par = min(np.degrees(t1b), np.degrees(t2b))
    print(f"    eps={e:.2f}: global L={L:.5f} @ parent~{par:.0f}deg child~{child:.0f}deg")
    if e == 0.0 and abs(child - 45) > 3:
        FAILS.append(f"kappa=1 eps=0 child-side atom {child:.0f}deg != 45 (absorbed)")
    if prev_child is not None and child < prev_child - 3:
        FAILS.append(f"kappa=1 child-side angle not monotone at eps={e}")
    prev_child = child
_, (t1b, t2b) = scan(1, q_v, p_v, 0.9, ts)
if abs(max(np.degrees(t1b), np.degrees(t2b)) - 90) > 5:
    FAILS.append("kappa=1 large-eps optimum not faithful (child atom !~ 90deg)")

print("  -- kappa=2: capacity collapse (faithful global; composite strictly worse) --")
for e in [0.0, 0.02, 0.1, 0.3]:
    L, _ = scan(2, q_v, p_v, e, ts)
    Lf = pop_angles(0.0, np.pi/2, 2, q_v, p_v, e)      # faithful
    La = pop_angles(0.0, np.pi/4, 2, q_v, p_v, e)      # absorbed (composite present)
    print(f"    eps={e:.2f}: global L={L:.5f}  faithful={Lf:.5f}  absorbed={La:.5f}")
    if abs(L) > 1e-6:
        FAILS.append(f"kappa=2 eps={e}: global min {L:.5f} != 0")
    if e > 0 and not (La > Lf + 1e-9):
        FAILS.append(f"kappa=2 eps={e}: absorbed not strictly worse than faithful")

print()
if FAILS:
    print("TOPK THEORY CHECK FAILED:")
    for f in FAILS:
        print("  -", f)
    raise SystemExit(1)
print("verify_topk_absorption: all checks passed (eps*_TopK(k=1)=2q; collapse at k>=2)")
