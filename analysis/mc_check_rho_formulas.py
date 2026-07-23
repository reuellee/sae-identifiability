"""M0: analytic Monte Carlo check of the round-9 estimator formulas.

Pre-lock verification (prereg 'Pre-lock development phase'): simulates the
theory note's leak model DIRECTLY (no SAE training) and checks that

  A. background-free: rho_D matches the error decomposition
     rho(1-g1*dJ) + (1-rho)*a0*dS; rho_X matches the odds identity;
     rho_C matches its leak-bias formula; rho_count targets min(rho,1-rho)
     under clean gating,
  B. with background: rho_D matches the h_B mixture formula and rho_X the
     q-cell contamination formula (incl. the lambda-mixture special case),
  C. the M endpoint (J-union-S mask) recovers the background-free formulas,
  D. edge cases: a0=g1=1 -> rho_X nan; empty pair-active -> nan; ties -> S;
     oracle_diags recovers the generative parameters.

Numpy only; runs anywhere. Exit code != 0 on any FAIL.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rho_estimators_lib import rho_estimators, oracle_diags

FAILS = []

def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'} {name}: got {got:.4f} want {want:.4f} (tol {tol})")
    if not ok:
        FAILS.append(name)

def simulate(N, rho, r_parent, a0, a1, g0, g1, dJ, dS,
             q=(1.0, 0.0, 0.0, 0.0), piB=0.5, tie11=0.0, seed=0):
    """q = (q00, q01, q10, q11) background pattern cells (dependence allowed).
    tie11: fraction of non-inverted J-11 tokens set to an exact tie (both 0.8,
    which the frozen rule sends to S -> counts toward dJ's >= inversion)."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    mJ = u < r_parent * rho
    mS = (u >= r_parent * rho) & (u < r_parent)
    mB = ~(mJ | mS)
    ap = np.zeros(N); ac = np.zeros(N)
    # J tokens
    j = np.where(mJ)[0]
    cfire = rng.random(len(j)) < a1
    pfire = rng.random(len(j)) < g1
    both = cfire & pfire
    inv = rng.random(len(j)) < dJ
    tie = (rng.random(len(j)) < tie11) & both & ~inv
    ac[j[cfire]] = 1.0; ap[j[pfire]] = 1.0
    ac[j[both & ~inv]] = 1.0; ap[j[both & ~inv]] = 0.5
    ac[j[both & inv]] = 0.5;  ap[j[both & inv]] = 1.0
    ac[j[tie]] = 0.8; ap[j[tie]] = 0.8
    # S tokens
    s = np.where(mS)[0]
    pfire = rng.random(len(s)) < g0
    cfire = rng.random(len(s)) < a0
    both = pfire & cfire
    inv = rng.random(len(s)) < dS
    ap[s[pfire]] = 1.0; ac[s[cfire]] = 1.0
    ap[s[both & ~inv]] = 1.0; ac[s[both & ~inv]] = 0.5
    ap[s[both & inv]] = 0.5;  ac[s[both & inv]] = 1.0
    # B tokens: draw pattern cells directly
    b = np.where(mB)[0]
    r = rng.random(len(b))
    q00, q01, q10, q11 = q
    pat01 = r < q01
    pat10 = (r >= q01) & (r < q01 + q10)
    pat11 = (r >= q01 + q10) & (r < q01 + q10 + q11)
    ac[b[pat01]] = 1.0
    ap[b[pat10]] = 1.0
    domB = rng.random(len(b)) < piB
    ac[b[pat11 & domB]] = 1.0; ap[b[pat11 & domB]] = 0.5
    ac[b[pat11 & ~domB]] = 0.5; ap[b[pat11 & ~domB]] = 1.0
    return ap, ac, mJ, mS, mB

N = 2_000_000

print("A. background-free formulas (r_parent = 1)")
for rho, a0, g1, dJ, dS in [(0.5, 0.6, 0.55, 0.0, 0.0),
                            (0.3, 0.6, 0.55, 0.04, 0.02),
                            (0.1, 0.16, 0.0, 0.0, 0.0),
                            (0.7, 0.6, 0.55, 0.0, 0.0)]:
    ap, ac, mJ, mS, mB = simulate(N, rho, 1.0, a0, 1.0, 1.0, g1, dJ, dS, seed=1)
    e = rho_estimators(ap, ac)
    want_d = rho * (1 - g1 * dJ) + (1 - rho) * a0 * dS
    check(f"rho_D rho={rho}", e["rho_d"], want_d, 0.003)
    odds = (rho / (1 - rho)) * (1 - g1) / (1 - a0)
    check(f"rho_X rho={rho}", e["rho_x"], odds / (1 + odds), 0.005)
    check(f"rho_C rho={rho}", e["rho_c"], rho + (1 - rho) * a0, 0.003)
ap, ac, *_ = simulate(N, 0.7, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, seed=2)
e = rho_estimators(ap, ac)
check("rho_count clean gating rho=0.7 -> min(rho,1-rho)", e["rho_count"], 0.3, 0.003)

print("B. background contamination")
rho, a0, g1, rp = 0.5, 0.6, 0.55, 0.4
q = (0.90, 0.04, 0.04, 0.02)   # symmetric exclusive cells, dependent q11
piB = 0.5
ap, ac, mJ, mS, mB = simulate(N, rho, rp, a0, 1.0, 1.0, g1, 0.0, 0.0,
                              q=q, piB=piB, seed=3)
e = rho_estimators(ap, ac)
rJ, rS, rB = rp * rho, rp * (1 - rho), 1 - rp
A_ = rJ * (1 - g1); C_ = rS * (1 - a0)
q00, q01, q10, q11 = q
want_x = (A_ + rB * q01) / (A_ + C_ + rB * (q01 + q10))
check("rho_X with background", e["rho_x"], want_x, 0.005)
lam = 2 * rB * q01 / (A_ + C_ + 2 * rB * q01)
check("rho_X lambda-mixture form", e["rho_x"],
      (1 - lam) * A_ / (A_ + C_) + lam / 2, 0.005)
hB = (q01 + q11 * piB) / (q01 + q10 + q11)
want_d = (rJ + rS * a0 * 0.0 + rB * (q01 + q11 * piB)) / (rJ + rS + rB * (1 - q00))
check("rho_D with background (h_B mixture)", e["rho_d"], want_d, 0.003)
dg = oracle_diags(ap, ac, mJ, mS, mB)
check("diag h_B", dg["h_B"], hB, 0.01)
check("diag q11_B", dg["q11_B"], q11, 0.002)

print("C. M endpoint recovers background-free values")
mJS = mJ | mS
em = rho_estimators(ap[mJS], ac[mJS])
check("rho_D-M", em["rho_d"], rho, 0.003)   # dJ=dS=0 -> exact
odds = (rho / (1 - rho)) * (1 - g1) / (1 - a0)
check("rho_X-M", em["rho_x"], odds / (1 + odds), 0.005)

print("D. edge cases + diagnostic recovery")
ap, ac, mJ, mS, mB = simulate(100_000, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0,
                              0.0, 0.0, seed=4)
e = rho_estimators(ap, ac)
ok = np.isnan(e["rho_x"])
print(f"  {'PASS' if ok else 'FAIL'} rho_X undefined at a0=g1=1")
if not ok: FAILS.append("rho_X nan")
e = rho_estimators(np.zeros(1000), np.zeros(1000))
ok = all(np.isnan(e[k]) for k in ("rho_d", "rho_x", "rho_c", "rho_count"))
print(f"  {'PASS' if ok else 'FAIL'} all estimators undefined with no pair-active tokens")
if not ok: FAILS.append("empty nan")
# ties -> S: J-11 exact ties count as inversions (delta_J) and go to S
rho, g1, tf = 0.5, 0.8, 0.5
ap, ac, mJ, mS, mB = simulate(N, rho, 1.0, 0.0, 1.0, 1.0, g1, 0.0, 0.0,
                              tie11=tf, seed=5)
e = rho_estimators(ap, ac)
want_d = rho * (1 - g1 * tf)   # tied J-11 tokens land in S
check("ties -> S shifts rho_D as predicted", e["rho_d"], want_d, 0.003)
dg = oracle_diags(ap, ac, mJ, mS, mB)
check("diag delta_J counts ties as inversions", dg["delta_J"], tf, 0.005)
# parameter recovery
ap, ac, mJ, mS, mB = simulate(N, 0.3, 0.5, 0.6, 1.0, 1.0, 0.55, 0.04, 0.02,
                              q=(0.9, 0.04, 0.04, 0.02), piB=0.3, seed=6)
dg = oracle_diags(ap, ac, mJ, mS, mB)
for name, want in [("a0", 0.6), ("g1", 0.55), ("a1", 1.0), ("g0", 1.0),
                   ("delta_J", 0.04), ("delta_S", 0.02), ("pi_B", 0.3),
                   ("rho_real", 0.3)]:
    check(f"diag {name}", dg[name], want, 0.01)

print()
if FAILS:
    print(f"M0 FAILED: {FAILS}")
    sys.exit(1)
print("M0: all checks passed")
