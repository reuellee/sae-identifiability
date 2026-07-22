"""Score Arm A (prereg_bimodality_armA.py) against the registered M1-M5
pass/fail thresholds in notes/prereg-bimodality-estimator.md.

Confirmatory metrics use ABSORBED runs only (cos_comp > 0.98); exclusions
disclosed with counts. Bootstrap CIs resample seeds (10k draws, np seed 0).
"""
import csv, math, os, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__) or ".", "..", "results", "prereg_armA",
    "armA_runs.csv")
rows = list(csv.DictReader(open(path)))
for r in rows:
    for k, v in r.items():
        if k not in ("block", "h4_mus", "h4_ws"):
            try: r[k] = float(v)
            except (ValueError, TypeError): r[k] = float("nan")

rng = np.random.default_rng(0)

def boot_ci(vals, n=10000):
    vals = np.asarray(vals, float)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])

def cell(rs, key):
    return [r[key] for r in rs if not math.isnan(r[key])]

# ---------------------------------------------------------------- disclosure
print("=" * 72)
for blk in ("R", "S", "M", "X"):
    b = [r for r in rows if r["block"] == blk]
    if not b: continue
    na = [r for r in b if r["absorbed"] < 1]
    print(f"block {blk}: {len(b)} runs, non-absorbed EXCLUDED: {len(na)}"
          + (f"  (seeds: {sorted(set(int(r['seed']) for r in na))})" if na else ""))
print("=" * 72)

R = [r for r in rows if r["block"] == "R" and r["absorbed"] == 1]

# ---------------------------------------------------------------- M1 (H1)
print("\n=== M1 (H1 no-go): TV(rho=0.02 vs 0.20), frozen absorbed SAEs ===")
ex = [r["tv_full"] - r["tv_null"] for r in R]
inp = [r["tv_inplane"] - r["tv_inplane_null"] for r in R]
if ex:
    lo, hi = boot_ci(ex)
    print(f"raw TV mean={np.mean(cell(R,'tv_full')):.4f}  "
          f"null={np.mean(cell(R,'tv_null')):.4f}")
    print(f"EXCESS TV (null-calibrated): mean={np.mean(ex):.4f}  "
          f"boot95=[{lo:.4f},{hi:.4f}]")
    print(f"in-plane excess TV:          mean={np.mean(inp):.4f}")
    print(f"diagnostic TV(host-only vs host+child sigs): "
          f"mean={np.mean(cell(R,'tv_cond')):.4f}")
    m1_pass = np.mean(ex) < 0.02 and lo <= 0
    m1_fals = np.mean(ex) > 0.05
    print(f"M1 verdict: {'PASS (H1 confirmed)' if m1_pass else ('FALSIFIED (TV>0.05)' if m1_fals else 'INCONCLUSIVE')}")

# ---------------------------------------------------------------- M2 (H2)
print("\n=== M2 (H2 estimator): |rho_hat - rho| from 2-comp GMM ===")
errs, pairs = [], []
for r in R:
    if not math.isnan(r["rho_hat_gmm"]):
        errs.append(abs(r["rho_hat_gmm"] - r["rho1"]))
        pairs.append((r["rho1"], r["rho_hat_gmm"]))
for rho in sorted(set(r["rho1"] for r in R)):
    c = [r for r in R if r["rho1"] == rho]
    e = [abs(r["rho_hat_gmm"] - rho) for r in c if not math.isnan(r["rho_hat_gmm"])]
    h = [r["rho_hat_gmm"] for r in c if not math.isnan(r["rho_hat_gmm"])]
    print(f"  rho={rho:.2f}: rho_hat mean={np.mean(h):.4f}+-{np.std(h):.4f} "
          f"|err|={np.mean(e):.4f}  (n={len(e)})")
if pairs:
    x, y = np.array(pairs).T
    pear = np.corrcoef(x, y)[0, 1] if len(set(x)) > 1 else float("nan")
    print(f"mean |err| = {np.mean(errs):.4f}   Pearson r = {pear:.4f}")
    m2_pass = np.mean(errs) < 0.02 and pear > 0.9
    m2_fals = np.mean(errs) >= 0.05 or pear <= 0.9
    print(f"M2 verdict: {'PASS (H2 confirmed)' if m2_pass else ('FALSIFIED' if m2_fals else 'INCONCLUSIVE')}")

# ---------------------------------------------------------------- M3
print("\n=== M3: estimator route vs binarized route, per rho ===")
for rho in sorted(set(r["rho1"] for r in R)):
    c = [r for r in R if r["rho1"] == rho]
    eg = [abs(r["rho_hat_gmm"] - rho) for r in c if not math.isnan(r["rho_hat_gmm"])]
    eb = [abs(r["rho_hat_bin"] - rho) for r in c if not math.isnan(r["rho_hat_bin"])]
    bh = [r["rho_hat_bin"] for r in c if not math.isnan(r["rho_hat_bin"])]
    win = "GMM" if np.mean(eg) < np.mean(eb) else "BIN"
    print(f"  rho={rho:.2f}: |err| gmm={np.mean(eg):.4f}  bin={np.mean(eb):.4f} "
          f"(bin rho_hat={np.mean(bh):.4f})  -> {win}")

# ---------------------------------------------------------------- M4 (H3)
S = [r for r in rows if r["block"] == "S" and r["absorbed"] == 1]
print("\n=== M4 (H3 SNR boundary): |err| vs sigma at rho=0.10 ===")
sig_star = None
for sig in sorted(set(r["sigma"] for r in S)):
    c = [r for r in S if r["sigma"] == sig]
    e = [abs(r["rho_hat_gmm"] - 0.10) for r in c if not math.isnan(r["rho_hat_gmm"])]
    print(f"  sigma={sig:.2f}: |err|={np.mean(e):.4f}+-{np.std(e):.4f} (n={len(e)})")
    if sig_star is None and e and np.mean(e) > 0.05:
        sig_star = sig
print(f"sigma* (first |err|>0.05) = {sig_star}")

# ---------------------------------------------------------------- M5 (H4)
M = [r for r in rows if r["block"] == "M"]
print("\n=== M5 (H4 multi-child, exploratory) ===")
for m in sorted(set(int(r["m_children"]) for r in M)):
    c = [r for r in M if r["m_children"] == m]
    ab = [r for r in c if r["absorbed"] == 1]
    agg_true = sum([r["rho1"] + r["rho2"] + r["rho3"] for r in c[:1]])
    e = [abs(r["rho_hat_gmm"] - agg_true) for r in ab
         if not math.isnan(r["rho_hat_gmm"])]
    print(f"  m={m}: absorbed(mono-composite) {len(ab)}/{len(c)}; "
          f"agg rho={agg_true:.2f}, 2-comp rho_hat err={np.mean(e) if e else float('nan'):.4f}")
    for r in ab[:3]:
        if r.get("h4_mus") and isinstance(r["h4_mus"], str):
            print(f"    seed {int(r['seed'])}: mus={r['h4_mus']} ws={r['h4_ws']}")

# ---------------------------------------------------------------- X block
X = [r for r in rows if r["block"] == "X" and r["absorbed"] == 1]
if X:
    print("\n=== X (exploratory, m_lat=31 capacity-forced single-latent) ===")
    ex = [r["tv_full"] - r["tv_null"] for r in X]
    print(f"excess TV mean={np.mean(ex):.4f}  "
          f"cond TV={np.mean(cell(X,'tv_cond')):.4f}")
    for rho in sorted(set(r["rho1"] for r in X)):
        c = [r for r in X if r["rho1"] == rho]
        h = [r["rho_hat_gmm"] for r in c if not math.isnan(r["rho_hat_gmm"])]
        fh = [r["fire|host_only"] for r in c if not math.isnan(r["fire|host_only"])]
        print(f"  rho={rho:.2f}: rho_hat={np.mean(h):.4f}+-{np.std(h):.4f} "
              f"fire|host_only={np.mean(fh):.3f} (n={len(h)})")
