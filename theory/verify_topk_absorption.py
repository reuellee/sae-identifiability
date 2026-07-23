"""Numerical regression tests + analytic propositions for the TopK absorption
theory (round 10). NOT a proof: the 2-atom inequalities below are elementary
analytic propositions (stated in topk_absorption.md); this script is a
numerical regression harness that checks them and, crucially, the SCOPE
boundaries — most importantly the 3-atom zero-loss counterexample that shows
the 2q crossover is a *capacity-limited (exactly-two-pair-atom)* result and
does NOT survive an overcomplete dictionary.

Model (2D toy): orthonormal v_p, v_c; events joint x=v_p+v_c (q),
parent-solo x=v_p (p), child-solo x=v_c (eps). ORACLE k-sparse nonneg coding:
L_kappa(U,x) = min over kappa-atom subsets S, min_{f>=0} ||x - U_S f||^2.
(An oracle lower bound on any learned encoder; see topk_absorption.md 2b.)

Checks:
  1. Per-event losses (2-atom faithful/absorbed), kappa in {1,2}, vs the table.
  2. Pure-strategy crossover eps*_TopK(k=1)=2q; capacity collapse at k=2 for
     the 2-atom absorbed dictionary.
  3. Continuous 2-atom optimum tilts (child-side atom 45->90 as eps grows) -
     exact assertions on the grid-located angles.
  4. SCOPE: the 3-atom {v_p, v_c, d_comp} dictionary has ZERO loss at k=1 for
     EVERY event and every eps -> no 2q crossover with >=3 atoms (GPT counterex).
  5. SCOPE: faithful is NOT the unique zero-loss k=2 solution (wide-cone
     neighbours), and a redundant {v_p,v_c,d_comp} solution is zero-loss and
     child-recovering while containing the composite.
Exit nonzero on any mismatch.
"""
import numpy as np
from itertools import combinations

RT2 = np.sqrt(2.0)
FAILS = []

def nnls(U, x, S):
    Ua = U[:, list(S)]; best = float(x @ x)
    for act in [c for k in range(1, len(S)+1) for c in combinations(range(len(S)), k)]:
        M = Ua[:, list(act)]
        try: f = np.linalg.solve(M.T @ M, M.T @ x)
        except np.linalg.LinAlgError: continue
        if (f >= -1e-12).all():
            r = x - M @ f; best = min(best, float(r @ r))
    return best

def loss_kappa(U, x, kappa):
    m = U.shape[1]; best = float(x @ x)
    for S in combinations(range(m), min(kappa, m)):
        best = min(best, nnls(U, x, S))
    return best

VP, VC = np.array([1.0, 0.0]), np.array([0.0, 1.0])
DCOMP = (VP + VC) / RT2
U_F = np.array([[1.0, 0.0], [0.0, 1.0]])                 # faithful {v_p, v_c}
U_A = np.array([[1.0, 1.0/RT2], [0.0, 1.0/RT2]])         # absorbed {v_p, d_comp}
U_3 = np.column_stack([VP, VC, DCOMP])                   # redundant triple
EV = {"joint": VP+VC, "psolo": VP, "csolo": VC}

def rec(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok: FAILS.append(name + " " + detail)

print("=== Check 1: 2-atom per-event losses ===")
claim = {("F",1,"joint"):1,("F",1,"psolo"):0,("F",1,"csolo"):0,
         ("A",1,"joint"):0,("A",1,"psolo"):0,("A",1,"csolo"):0.5,
         ("F",2,"joint"):0,("F",2,"psolo"):0,("F",2,"csolo"):0,
         ("A",2,"joint"):0,("A",2,"psolo"):0,("A",2,"csolo"):0.5}
for nm,U in [("F",U_F),("A",U_A)]:
    for kap in (1,2):
        for ev,x in EV.items():
            got=loss_kappa(U,x,kap)
            rec(f"{nm} k={kap} {ev}", abs(got-claim[(nm,kap,ev)])<1e-9, f"got {got:.4f}")

print("=== Check 2: 2-atom crossover eps*=2q and k=2 collapse ===")
def pop(U,kap,q,p,e): return (q*loss_kappa(U,EV["joint"],kap)+p*loss_kappa(U,EV["psolo"],kap)
                              + e*loss_kappa(U,EV["csolo"],kap))
for q in (0.1,0.2,0.3):
    rec(f"k=1 tie at eps=2q (q={q})", abs(pop(U_F,1,q,0.2,2*q)-pop(U_A,1,q,0.2,2*q))<1e-9)
    rec(f"k=1 absorbed below 2q (q={q})", pop(U_A,1,q,0.2,q) < pop(U_F,1,q,0.2,q))
    rec(f"k=1 faithful above 2q (q={q})", pop(U_A,1,q,0.2,3*q) > pop(U_F,1,q,0.2,3*q))
    rec(f"k=2 faithful beats absorbed for eps>0 (q={q})",
        all(pop(U_F,2,q,0.2,e) < pop(U_A,2,q,0.2,e) for e in (0.05,0.2,2*q,4*q)))

print("=== Check 3: 2-atom continuous optimum tilts (exact grid angles) ===")
ts = np.linspace(-np.pi/2, np.pi, 145)
def scan(kap,q,p,e):
    best=(np.inf,(np.nan,np.nan))
    for i,t1 in enumerate(ts):
        for t2 in ts[i+1:]:
            U=np.array([[np.cos(t1),np.cos(t2)],[np.sin(t1),np.sin(t2)]])
            L=pop(U,kap,q,p,e)
            if L<best[0]-1e-12: best=(L,(t1,t2))
    return best
prev=None
for e,exp_child in [(0.0,45),(0.1,52),(0.2,58),(0.3,90),(0.4,90)]:
    _,(t1,t2)=scan(1,0.2,0.2,e); child=max(np.degrees(t1),np.degrees(t2))
    rec(f"k=1 eps={e} child-side atom ~{exp_child}deg", abs(child-exp_child)<=4, f"got {child:.0f}")
    if prev is not None: rec(f"k=1 child angle monotone at eps={e}", child>=prev-3, f"{child:.0f}")
    prev=child

print("=== Check 4 (SCOPE): 3-atom {v_p,v_c,d_comp} is ZERO-loss at k=1 (GPT counterexample) ===")
for ev,x in EV.items():
    L=loss_kappa(U_3,x,1)
    rec(f"3-atom k=1 {ev} loss=0", abs(L)<1e-9, f"got {L:.6f}")
# => with >=3 atoms there is NO 2q crossover: overcomplete escapes the theorem.
for e in (0.0,0.05,0.2,0.5):
    tot=pop(U_3,1,0.2,0.2,e)
    rec(f"3-atom k=1 population loss=0 (eps={e})", abs(tot)<1e-9, f"got {tot:.6f}")

print("=== Check 5 (SCOPE): faithful not unique at k=2; redundant triple child-recovering ===")
# wide-cone neighbours also reach zero loss at k=2
for a,b in [(-30,120),(-10,100),(0,90)]:
    U=np.array([[np.cos(np.radians(a)),np.cos(np.radians(b))],
                [np.sin(np.radians(a)),np.sin(np.radians(b))]])
    L=pop(U,2,0.2,0.2,0.1)
    rec(f"k=2 wide-cone ({a},{b}) zero-loss", abs(L)<1e-9, f"got {L:.6f}")
# the redundant triple contains d_comp yet is zero-loss and has a child atom (v_c)
rec("3-atom k=2 zero loss (contains composite AND child)", abs(pop(U_3,2,0.2,0.2,0.1))<1e-9)
rec("3-atom contains a child atom (v_c column present)",
    max(abs(U_3[:, j] @ VC) for j in range(3)) > 0.999)

print()
if FAILS:
    print("TOPK REGRESSION TESTS FAILED:")
    for f in FAILS: print("  -", f)
    raise SystemExit(1)
print("verify_topk_absorption: all regression tests pass. Scope confirmed: "
      "eps*_TopK=2q holds ONLY for the capacity-limited 2-atom dictionary; "
      "a 3rd atom gives a zero-loss child-recovering solution at k=1 (no crossover).")
