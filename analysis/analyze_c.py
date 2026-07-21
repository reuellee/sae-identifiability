"""Score Experiment C pre-registered predictions + Gemini's discriminating test:
escape hypothesis predicts rho (in-plane fraction) decreases with beta while phi stays ~45."""
import csv, json, math
from collections import defaultdict
from statistics import mean, median

rows = list(csv.DictReader(open("results_remedies.csv")))

print("=== C1: coherence penalty (prediction: phi 45->~90 as beta crosses beta*) ===")
print("    detail = top in-plane latents (rho, phi); escape test: does rho drop with beta?")
c1 = [r for r in rows if r["exp"] == "C1"]
by = defaultdict(list)
for r in c1:
    by[(float(r["eps"]), float(r["beta_mult"]))].append(r)
for (eps, bm), rs in sorted(by.items()):
    phis = [float(r["phi_child"]) for r in rs]
    # per-run top-latent list [(rho, phi), ...]; take the child-side entry (20<phi<120) max rho
    rho_childs, n_inplane = [], []
    for r in rs:
        top = json.loads(r["detail"])
        n_inplane.append(len(top))
        cand = [t for t in top if 20 < t[1] < 120]
        rho_childs.append(max((t[0] for t in cand), default=float("nan")))
    print(f"eps={eps:.2f} beta={bm}b*  phi={mean(phis):5.1f}  "
          f"rho_child={mean(rho_childs):.3f}  n_latents_inplane(rho>.5)={mean(n_inplane):.1f}")

print()
print("=== C2: matryoshka single-child (prediction: absorption persists ~45) ===")
for r in [r for r in rows if r["exp"] == "C2"]:
    top = json.loads(r["detail"])
    print(f"eps={r['eps']} seed={r['seed']} phi={float(r['phi_child']):5.1f}  top={top[:3]}")

print()
print("=== C3: two-child (prediction: vanilla absorbed / matryoshka faithful) ===")
for r in [r for r in rows if r["exp"].startswith("C3")]:
    print(f"{r['exp']:14s} eps={r['eps']} seed={r['seed']} -> {r['label']:9s} {r['detail']}")
