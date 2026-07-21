"""Post-run analysis: summary stats + triple-solution detection.

A 'triple' run has BOTH a faithful child latent (angle >= 78 deg, rho > 0.8)
AND a composite latent (40-55 deg, rho > 0.8) — the unconstrained optimum for
eps > 0 per the redundant-triple remark. Absorbed = composite only. Faithful =
child only. Hedged = single in-between latent.
"""
import json, math
import pandas as pd

def eps_star(lam, q):
    s2 = math.sqrt(2)
    return lam * q * (8 - 4 * s2 - lam) / (2 * (1 - (2 - s2) * lam))

ab = pd.read_csv("results_absorption.csv")

def classify(row):
    lat = [(r, a) for r, a in json.loads(row.top_latents) if r > 0.8]
    child = any(a >= 78 for r, a in lat)
    comp = any(40 <= a <= 55 for r, a in lat)
    if child and comp: return "triple"
    if child: return "faithful"
    if comp: return "absorbed"
    return "hedged"

ab["cls"] = ab.apply(classify, axis=1)
ab["eps_ratio"] = ab.apply(lambda r: r.eps / eps_star(r.lam, r.q), axis=1)

print("=== classification by eps/eps* bin ===")
bins = [0, 0.5, 1.0, 2.0, 5.0, 100]
labels = ["<0.5", "0.5-1", "1-2", "2-5", ">5"]
ab["bin"] = pd.cut(ab.eps_ratio, bins=bins, labels=labels, include_lowest=True)
print(ab.groupby("bin", observed=True).cls.value_counts().unstack(fill_value=0))

print("\n=== eps=0 runs (must be absorbed, phi ~ 45) ===")
z = ab[ab.eps == 0]
print(z[["lam", "q", "seed", "phi_child", "cls"]].to_string(index=False))
print("mean phi at eps=0:", round(z.phi_child.mean(), 2))

print("\n=== largest-eps runs (must be faithful-ish, phi ~ 90) ===")
m = ab[ab.eps == ab.eps.max()]
print("eps =", ab.eps.max(), "mean phi:", round(m.phi_child.mean(), 2),
      "| classes:", dict(m.cls.value_counts()))

print("\n=== transition sharpness: mean phi by eps_ratio ===")
ab["rbin"] = pd.cut(ab.eps_ratio, bins=[0, .25, .5, .75, 1, 1.5, 2, 3, 5, 100],
                    include_lowest=True)
print(ab.groupby("rbin", observed=True).agg(
    phi=("phi_child", "mean"), n=("phi_child", "size")).round(1).to_string())

rec = pd.read_csv("results_recovery.csv")
print("\n=== recovery summary ===")
print(rec.groupby(["n", "k"]).agg(mmcs=("mmcs", "mean"),
      frac=("frac_recovered", "mean"), mu=("mu", "mean"),
      kstar=("kstar_worstcase", "mean")).round(3).to_string())
