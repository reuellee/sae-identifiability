"""First-pass analysis of SAE-identifiability Exp A + B against theory."""
import csv, math
from collections import defaultdict
from statistics import median, mean

R2 = math.sqrt(2)

def eps_star(lam, q):
    return lam * q * (8 - 4 * R2 - lam) / (2 * (1 - (2 - R2) * lam))

# ---------------- Experiment B: absorption phase transition ----------------
rows = list(csv.DictReader(open("results_absorption.csv")))
by_cell = defaultdict(list)
for r in rows:
    by_cell[(float(r["q"]), float(r["lam"]), float(r["eps"]))].append(
        float(r["phi_child"]) if r["phi_child"] not in ("", "nan") else float("nan"))

print("=== Exp B: phi_child (median over seeds) vs eps, per (q, lam) ===")
print("    absorbed ~45 deg, faithful ~90 deg; * marks predicted eps*")
groups = sorted(set((q, lam) for q, lam, _ in by_cell))
for q, lam in groups:
    es = eps_star(lam, q)
    eps_list = sorted(e for qq, ll, e in by_cell if (qq, ll) == (q, lam))
    line = []
    trans_emp = None
    prev_e = None
    for e in eps_list:
        vals = [v for v in by_cell[(q, lam, e)] if not math.isnan(v)]
        nan_n = sum(1 for v in by_cell[(q, lam, e)] if math.isnan(v))
        m = median(vals) if vals else float("nan")
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        if trans_emp is None and vals and m > 67.5:
            trans_emp = (prev_e, e)  # bracketing interval
        line.append(f"{e:.3f}:{m:5.1f}" + (f"±{spread:4.1f}" if spread > 5 else "      ")
                    + ("!" * nan_n))
        prev_e = e
    print(f"q={q} lam={lam:4.2f}  eps*={es:.4f}  emp. transition in {trans_emp}")
    print("   " + "  ".join(line))

print()
print("=== Exp B: parent latent present? (should stay present when faithful) ===")
for q, lam in groups:
    eps_list = sorted(e for qq, ll, e in by_cell if (qq, ll) == (q, lam))
    par = []
    for e in eps_list:
        ps = [int(r["has_parent"]) for r in rows
              if (float(r["q"]), float(r["lam"]), float(r["eps"])) == (q, lam, e)]
        par.append(f"{e:.3f}:{sum(ps)}/{len(ps)}")
    print(f"q={q} lam={lam:4.2f}  " + "  ".join(par))

# ---------------- Experiment A: recovery boundary ----------------
print()
print("=== Exp A: recovery vs worst-case Donoho-Elad k* ===")
arows = list(csv.DictReader(open("results_recovery.csv")))
by_nk = defaultdict(list)
for r in arows:
    by_nk[(int(r["n"]), int(r["k"]))].append(
        (float(r["mmcs"]), float(r["frac_recovered"]), float(r["mu"]), float(r["kstar_worstcase"])))
for (n, k), vals in sorted(by_nk.items()):
    mm = mean(v[0] for v in vals); fr = mean(v[1] for v in vals)
    mu = mean(v[2] for v in vals); ks = mean(v[3] for v in vals)
    print(f"n={n:3d} k={k:2d}  mmcs={mm:.3f}  frac>0.9={fr:.3f}  mu={mu:.3f}  worst-case k*={ks:.2f}")
