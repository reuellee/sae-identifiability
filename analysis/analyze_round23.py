"""Round 2+3 analysis: eps_c vs eps* (B-fine), A-oracle verdict, corrected-boundary
validation (C1-boundary vs eps**), C3-rich matryoshka geometry."""
import csv, json, math
from collections import defaultdict
from statistics import mean, median

RT2 = math.sqrt(2)
def eps_star(lam, q):
    return lam * q * (8 - 4 * RT2 - lam) / (2 * (1 - (2 - RT2) * lam))

# ---------------- B-fine: empirical transition location vs theory ----------------
print("=== B-fine: eps_c (phi_func crossing 67.5) vs eps*, 16 seeds/cell ===")
rows = list(csv.DictReader(open("results_b_fine.csv")))
cells = sorted(set((float(r["q"]), float(r["lam"])) for r in rows))
for q, lam in cells:
    es = eps_star(lam, q)
    cross = []
    for s in range(16):
        pts = sorted((float(r["eps"]), float(r["phi_func"])) for r in rows
                     if (float(r["q"]), float(r["lam"])) == (q, lam)
                     and int(r["seed"]) == s)
        for (e0, p0v), (e1, p1v) in zip(pts, pts[1:]):
            if p0v < 67.5 <= p1v:
                cross.append(e0 + (67.5 - p0v) / (p1v - p0v) * (e1 - e0))
                break
    if cross:
        cs = sorted(cross)
        med = median(cs)
        lo, hi = cs[max(0,len(cs)//10)], cs[min(len(cs)-1, 9*len(cs)//10)]
        print(f"q={q} lam={lam:4.2f}  eps*={es:.4f}  eps_c={med:.4f} "
              f"[{lo:.4f},{hi:.4f}]  ratio={med/es:.2f}  (n={len(cross)}/16)")
    else:
        print(f"q={q} lam={lam:4.2f}  eps*={es:.4f}  NO CROSSINGS ({len(cross)}/16)")

# ---------------- A-oracle ----------------
print()
print("=== A-oracle: mmcs / frac>0.9 by condition (identifiability vs trainability) ===")
arows = list(csv.DictReader(open("results_a_oracle.csv")))
for n in (128, 256):
    for k in (2, 4, 8, 16, 24, 32):
        parts = []
        for c in ("random", "reinit", "oracle"):
            sel = [r for r in arows if int(r["n"]) == n and int(r["k"]) == k
                   and r["cond"] == c]
            parts.append(f"{c}: {mean(float(r['mmcs']) for r in sel):.3f}/"
                         f"{mean(float(r['frac_recovered']) for r in sel):.2f}")
        print(f"n={n:3d} k={k:2d}  " + "   ".join(parts))

# ---------------- C1-boundary: basin classification vs eps**(beta) ----------------
print()
print("=== C1-boundary: basin fractions vs eps (predict faithful>50% above eps**) ===")
print("    eps**(1.0b*)=0.0112  eps**(4.0b*)=0.0141")
crows = list(csv.DictReader(open("results_c1_boundary.csv")))
def classify(r):
    inv = json.loads(r["inv"])
    faith = any(rho > 0.8 and 70 < phi < 110 for rho, phi in inv)
    neg = any(rho > 0.7 and -60 < phi < -15 for rho, phi in inv)
    comp = any(rho > 0.7 and 25 < phi < 60 for rho, phi in inv)
    if faith: return "faithful"
    if neg and comp: return "anti"
    inplane = [(rho, phi) for rho, phi in inv if rho > 0.5]
    if len(inplane) == 1 and 15 < inplane[0][1] < 60: return "merged"
    return "other"
for bm in sorted(set(float(r["beta_mult"]) for r in crows)):
    print(f"beta={bm}b*:")
    for e in sorted(set(float(r["eps"]) for r in crows)):
        sel = [r for r in crows if float(r["beta_mult"]) == bm and float(r["eps"]) == e]
        lab = [classify(r) for r in sel]
        n = len(lab)
        fr = {k: lab.count(k) for k in ("faithful", "anti", "merged", "other")}
        phis = [float(r["phi_func"]) for r in sel]
        print(f"  eps={e:6.4f}  faithful={fr['faithful']}/{n} anti={fr['anti']} "
              f"merged={fr['merged']} other={fr['other']}  phi_func_med={median(phis):5.1f}")

# ---------------- C3-rich ----------------
print()
print("=== C3-rich: max |cos| per canonical direction (mean over seeds) ===")
drows = list(csv.DictReader(open("results_c3_rich.csv")))
keys = ["max_parent", "max_child1", "max_child2", "max_comp1", "max_comp2",
        "max_comp3", "max_cc"]
for mode in ("vanilla", "matryoshka"):
    for e in (0.0, 0.01):
        sel = [r for r in drows if r["mode"] == mode and float(r["eps"]) == e]
        vals = "  ".join(f"{k[4:]}={mean(float(r[k]) for r in sel):.2f}" for k in keys)
        print(f"{mode:11s} eps={e}: {vals}")
print()
print("--- C3 matryoshka top in-subspace latents (first 2 runs) ---")
for r in [r for r in drows if r["mode"] == "matryoshka"][:2]:
    print(f"eps={r['eps']} seed={r['seed']}: {r['top']}")
