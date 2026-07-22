"""Score round 8 against notes/prereg-round8-scaling-robustness.md.
E1 from results/round8/r8e1_runs.csv, E2/E3 from r8syn_runs.csv."""
import csv, math, os, sys
import numpy as np

base = os.path.join(os.path.dirname(__file__) or ".", "..", "results", "round8")

def load(name):
    p = os.path.join(base, name)
    if not os.path.exists(p): return []
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in r.items():
            if k != "exp":
                try: r[k] = float(v)
                except (ValueError, TypeError): r[k] = float("nan")
    return rows

rng = np.random.default_rng(0)
def seed_boot(d, stat, n=10000):
    seeds = sorted(d)
    out = []
    for _ in range(n):
        pool = [v for s in rng.choice(seeds, len(seeds), replace=True) for v in d[s]]
        if pool: out.append(stat(pool))
    return np.percentile(out, [2.5, 97.5])

# ------------------------------------------------------------------- E1
e1 = load("r8e1_runs.csv")
if e1:
    print("=== E1: v1.2 held-out cutoff transfer (L_HI = 1.9) ===")
    ab = [r for r in e1 if r["eps"] == 0.002 and r["absorbed"] == 1 and r["distinct"] == 1]
    ex = [r for r in e1 if r["eps"] == 0.002 and (r["absorbed"] == 0 or r["distinct"] == 0)]
    print(f"absorbed-formed {len(ab)}/{len(ab)+len(ex)} (exclusions disclosed: {len(ex)})")
    for m in (128, 256):
        c = [r for r in ab if r["m"] == m]
        fl = sum(int(r["tp_flagged"]) for r in c)
        print(f"  m={m}: flagged {fl}/{len(c)} "
              f"(tp_lift {np.mean([r['tp_lift'] for r in c]):.3f} "
              f"+- {np.std([r['tp_lift'] for r in c]):.3f})")
    # WIDTH-SPECIFIC confirmatory endpoints (amendment, pre-collection):
    # T1a/T1b per width; pooled recall secondary only.
    for m, tag in ((128, "T1a"), (256, "T1b")):
        c = [r for r in ab if r["m"] == m]
        rec = np.mean([r["tp_flagged"] for r in c]) if c else float("nan")
        d = {}
        for r in c: d.setdefault(int(r["seed"]), []).append(r["tp_flagged"])
        lo, hi = seed_boot(d, np.mean) if d else (float("nan"),) * 2
        print(f"{tag} recall(m={m}): point {rec:.4f}  CI [{lo:.3f}, {hi:.3f}] -> "
              f"{'PASS' if rec >= 0.90 else ('FALSIFIED' if rec < 0.70 else 'INCONCLUSIVE')}")
    pooled = np.mean([r["tp_flagged"] for r in ab]) if ab else float("nan")
    print(f"pooled recall (secondary, cannot override widths): {pooled:.4f}")
    # T2 per width + all-pairs specificity readouts (amendment §4)
    for m in (128, 256):
        fa = [r for r in e1 if r["eps"] == 0.05 and r["m"] == m]
        t2 = np.mean([r["tp_flagged"] for r in fa]) if fa else float("nan")
        any_flag = np.mean([r["n_flagged"] > 0 for r in fa]) if fa else float("nan")
        print(f"T2 m={m}: oracle-pair flag rate {t2:.4f} (pass <= 0.10) -> "
              f"{'PASS' if t2 <= 0.10 else 'FAIL'}; "
              f"faithful SAEs with >=1 full-scan flag: {any_flag:.2f}")
    # Orientation + downstream endpoints, conditional on detection (amendment §3)
    for m in (128, 256):
        fl = [r for r in ab if r["m"] == m and r["tp_flagged"] == 1]
        if not fl: continue
        oo = [int(r["rate_comp"] < r["rate_par"]) if not math.isnan(r["rate_comp"])
              else 0 for r in fl]   # detected comp = rarer; correct iff oracle comp rarer
        crc = [r.get("child_res_cos", float("nan")) for r in fl]
        print(f"orientation m={m} (conditional on detection): accuracy {np.mean(oo):.2f} "
              f"(n={len(fl)}); child_res_cos auto-orient {np.nanmean(crc):.3f} "
              f"(oracle-orient recompute from saved weights: separate pass)")
    for m in (128, 256):
        c = [r for r in e1 if r["m"] == m]
        npairs = m * (m - 1) / 2
        print(f"T3 m={m}: flags/SAE {np.mean([r['n_flagged'] for r in c]):.1f} "
              f"({np.mean([r['n_flagged'] for r in c])/npairs*1e6:.0f}/M pairs; "
              f"real-background candidates, natural status unknown); "
              f"rho_hat {np.nanmean([r.get('rho_hat', float('nan')) for r in c]):.3f} (true 0.5, descriptive)")

# ------------------------------------------------------------------- E2
syn = load("r8syn_runs.csv")
e2n = [r for r in syn if r["exp"] == "E2null"]
e2a = [r for r in syn if r["exp"] == "E2abs"]
if e2n:
    print("\n=== E2: width-scaling null calibration (v1.1) ===")
    print(f"{'d':>5} {'m':>5} {'pairs':>7} {'FP/SAE':>7} {'FP/M':>7} "
          f"{'recall(abs-formed)':>19} {'formed':>7} {'wall_s':>7} {'mem_MB':>7}")
    for m in sorted(set(int(r["m"]) for r in e2n)):
        cn = [r for r in e2n if r["m"] == m]
        ca = [r for r in e2a if r["m"] == m]
        ab = [r for r in ca if r["absorbed"] == 1]
        npairs = m * (m - 1) / 2
        fp = np.mean([r["fp_count"] for r in cn])
        rec = np.mean([r["tp_flagged"] for r in ab]) if ab else float("nan")
        print(f"{int(cn[0]['d']):>5} {m:>5} {int(npairs):>7} {fp:>7.2f} "
              f"{fp/npairs*1e6:>7.0f} {rec:>19.3f} {len(ab):>4}/{len(ca):<2} "
              f"{cn[0]['wall_s']:>7.0f} {cn[0]['mem_mb']:>7.0f}")
    print("registered soft expectation: FP/M falls with width")

# ------------------------------------------------------------------- E3
if syn:
    print("\n=== E3: robustness cells (descriptive) ===")
    for exp, key in (("E3angle", "ccos"), ("E3prev", "rho"), ("E3topk", "topk")):
        for val in sorted(set(r[key] for r in syn if r["exp"] == exp)):
            c = [r for r in syn if r["exp"] == exp and r[key] == val]
            ab = [r for r in c if r["absorbed"] == 1]
            fl = [r for r in ab if r["tp_flagged"] == 1]
            crc = [r.get("child_res_cos", float("nan")) for r in fl]
            oo = [r.get("orient_ok", float("nan")) for r in fl]
            print(f"  {exp} {key}={val}: formed {len(ab)}/{len(c)}, flagged "
                  f"{len(fl)}/{len(ab) if ab else 0}, orient_ok "
                  f"{np.nanmean(oo) if fl else float('nan'):.2f}, res_cos(u*) "
                  f"{np.nanmean(crc) if fl else float('nan'):.3f}, "
                  f"tp_cos {np.mean([r['tp_cos'] for r in ab]) if ab else float('nan'):.3f}, "
                  f"tp_lift {np.mean([r['tp_lift'] for r in ab]) if ab else float('nan'):.2f}")
