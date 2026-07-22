"""Score pair-identification Arm 1 against the registered D1-D4 thresholds
(notes/prereg-pair-identification.md). Exclusions disclosed as in Arm A."""
import csv, math, os, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__) or ".", "..", "results", "prereg_pairid",
    "arm1_runs.csv")
rows = list(csv.DictReader(open(path)))
for r in rows:
    for k, v in r.items():
        if k != "cond":
            try: r[k] = float(v)
            except (ValueError, TypeError): r[k] = float("nan")

def g(rs, k): return [r[k] for r in rs if not math.isnan(r.get(k, float("nan")))]

Aab = [r for r in rows if r["cond"] == "A" and r["absorbed"] == 1]
Ana = [r for r in rows if r["cond"] == "A" and r["absorbed"] == 0]
print(f"A: {len(Aab)} absorbed / {len(Aab)+len(Ana)} (non-absorbed excluded, "
      f"disclosed per cell below)")
for rho in sorted(set(r["rho"] for r in rows if r["cond"] == "A")):
    for sig in sorted(set(r["sigma"] for r in rows if r["cond"] == "A")):
        c = [r for r in rows if r["cond"] == "A" and r["rho"] == rho
             and r["sigma"] == sig]
        ab = [r for r in c if r["absorbed"] == 1]
        fl = [r for r in ab if r["tp_flagged"] == 1]
        print(f"  rho={rho:.2f} sig={sig:.1f}: absorbed {len(ab)}/{len(c)}, "
              f"true pair flagged {len(fl)}/{len(ab)}"
              + (f"  (tp_cos={np.mean(g(ab,'tp_cos')):.3f}, "
                 f"tp_lift={np.mean(g(ab,'tp_lift')):.3f})" if ab else ""))

print("\n=== D1 (G1 recall over absorbed A-runs) ===")
rec = np.mean([r["tp_flagged"] for r in Aab]) if Aab else float("nan")
print(f"recall = {rec:.4f}  (pass >= 0.90; falsified < 0.70)")
print(f"D1 verdict: {'PASS' if rec >= 0.90 else ('FALSIFIED' if rec < 0.70 else 'INCONCLUSIVE')}")

print("\n=== D2 (G2 discrimination) ===")
CD = [r for r in rows if r["cond"] == "CD"]
cd_rate = np.mean([r["tp_flagged"] for r in CD]) if CD else float("nan")
print(f"(a) planted CD pair flag rate = {cd_rate:.4f}  (pass <= 0.10; falsified > 0.30)")
for br in sorted(set(r["b_rate"] for r in CD)):
    c = [r for r in CD if r["b_rate"] == br]
    print(f"    b_rate={br:.2f}: flag rate {np.mean([r['tp_flagged'] for r in c]):.3f} "
          f"(tp_cos={np.mean(g(c,'tp_cos')):.3f}, tp_lift={np.mean(g(c,'tp_lift')):.3f})")
fp_pool = [r for r in rows if r["cond"] in ("A", "CD", "F", "N")]
fp = np.mean([r["fp_count"] for r in fp_pool])
print(f"(b) mean false-positive pairs/SAE (A,CD,F,N) = {fp:.4f}  (pass <= 0.10)")
for cond in ("A", "CD", "F", "N"):
    c = [r for r in rows if r["cond"] == cond]
    print(f"    {cond}: fp/SAE = {np.mean([r['fp_count'] for r in c]):.3f}, "
          f"n_flagged/SAE = {np.mean([r['n_flagged'] for r in c]):.3f}")
d2 = cd_rate <= 0.10 and fp <= 0.10
print(f"D2 verdict: {'PASS' if d2 else ('FALSIFIED' if cd_rate > 0.30 else 'INCONCLUSIVE')}")

print("\n=== D3 (G3 child recovery, correctly flagged A pairs) ===")
flA = [r for r in Aab if r["tp_flagged"] == 1]
cr = g(flA, "child_res_cos")
med = float(np.median(cr)) if cr else float("nan")
print(f"median cos(residual, v_c) = {med:.4f} (n={len(cr)}; "
      f"orientation correct {np.mean(g(flA,'orient_ok')):.3f})")
print(f"D3 verdict: {'PASS' if med > 0.9 else ('FALSIFIED' if med <= 0.75 else 'INCONCLUSIVE')}")

print("\n=== D4 (G4 end-to-end rho; confirmatory at sigma=0 per locked note) ===")
flA0 = [r for r in flA if r["sigma"] == 0.0]
errs = [abs(r["rho_hat"] - r["rho"]) for r in flA0
        if not math.isnan(r.get("rho_hat", float("nan")))]
for sig in sorted(set(r["sigma"] for r in flA)):
    for rho in sorted(set(r["rho"] for r in flA if r["sigma"] == sig)):
        c = [r for r in flA if r["rho"] == rho and r["sigma"] == sig]
        e = [abs(x["rho_hat"] - rho) for x in c if not math.isnan(x["rho_hat"])]
        tag = "CONF" if sig == 0.0 else "expl"
        print(f"  [{tag}] sig={sig:.1f} rho={rho:.2f}: rho_hat={np.mean(g(c,'rho_hat')):.4f} "
              f"|err|={np.mean(e):.4f} (n={len(e)})")
m4 = np.mean(errs) if errs else float("nan")
print(f"mean |rho_hat - rho| (sigma=0) = {m4:.4f}  (pass <= 0.03)")
print(f"D4 verdict: {'PASS' if m4 <= 0.03 else 'FAIL'}")

print("\n=== G5 (CDX exclusive-correlated, exploratory) ===")
CDX = [r for r in rows if r["cond"] == "CDX"]
if CDX:
    print(f"planted pair flagged {np.mean([r['tp_flagged'] for r in CDX]):.3f} "
          f"(predicted ~1: known equivalence class)  "
          f"tp_lift={np.mean(g(CDX,'tp_lift')):.3f}")

print("\n=== F / N controls ===")
for cond in ("F", "N"):
    c = [r for r in rows if r["cond"] == cond]
    print(f"{cond}: n_flagged/SAE = {np.mean([r['n_flagged'] for r in c]):.3f}"
          + (f", cos_child={np.mean(g(c,'cos_child')):.3f} (faithful check)"
             if cond == "F" else ""))

# ---------------------------------------------------------------- bootstrap CIs
# Pre-registered: seed bootstrap, 10k draws, np seed 0 (implemented post-Arm-1
# after external research review flagged the omission; deterministic given data).
print("\n=== Seed-bootstrap 95% CIs (10,000 draws, np.random.default_rng(0)) ===")
rng = np.random.default_rng(0)
def seed_boot(pairs_by_seed, stat, n=10000):
    """pairs_by_seed: dict seed -> list of values; resample seeds, pool, stat."""
    seeds = sorted(pairs_by_seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(seeds, len(seeds), replace=True)
        pool = [v for s in pick for v in pairs_by_seed[s]]
        if pool: vals.append(stat(pool))
    return np.percentile(vals, [2.5, 97.5])

def by_seed(rs, key):
    d = {}
    for r in rs:
        if not math.isnan(r.get(key, float("nan"))):
            d.setdefault(int(r["seed"]), []).append(r[key])
    return d

if Aab:
    lo, hi = seed_boot(by_seed(Aab, "tp_flagged"), np.mean)
    print(f"D1 recall:        point {np.mean([r['tp_flagged'] for r in Aab]):.4f}  CI [{lo:.3f}, {hi:.3f}]"
          f"  -> CI {'establishes' if lo >= 0.90 else 'does NOT establish'} population >= 0.90")
if CD:
    lo, hi = seed_boot(by_seed(CD, "tp_flagged"), np.mean)
    print(f"D2a CD flag rate: point {cd_rate:.4f}  CI [{lo:.3f}, {hi:.3f}]"
          f"  -> CI {'establishes' if hi <= 0.10 else 'does NOT establish'} population <= 0.10")
lo, hi = seed_boot(by_seed(fp_pool, "fp_count"), np.mean)
print(f"D2b fp/SAE:       point {fp:.4f}  CI [{lo:.3f}, {hi:.3f}]"
      f"  -> CI {'establishes' if hi <= 0.10 else 'does NOT establish'} population <= 0.10")
if cr:
    lo, hi = seed_boot(by_seed(flA, "child_res_cos"), np.median)
    print(f"D3 median cos:    point {med:.4f}  CI [{lo:.3f}, {hi:.3f}]"
          f"  -> CI {'establishes' if lo > 0.9 else 'does NOT establish'} population > 0.9")
if errs:
    err_by_seed = {}
    for r in flA0:
        if not math.isnan(r.get("rho_hat", float("nan"))):
            err_by_seed.setdefault(int(r["seed"]), []).append(abs(r["rho_hat"] - r["rho"]))
    lo, hi = seed_boot(err_by_seed, np.mean)
    print(f"D4 |rho err|:     point {m4:.4f}  CI [{lo:.4f}, {hi:.4f}]"
          f"  -> CI {'establishes' if hi <= 0.03 else 'does NOT establish'} population <= 0.03")

# ---------------------------------------------------------------- scaling metrics
print("\n=== Scaling metrics (post-review addition) ===")
n_pairs = 32 * 31 // 2
fp_per_M = fp / n_pairs * 1e6
rec_pt = np.mean([r["tp_flagged"] for r in Aab]) if Aab else float("nan")
print(f"false positives per million candidate pairs (m=32): {fp_per_M:.0f}")
print("precision at assumed absorbed-pair prevalence (per candidate pair):")
for prev in (1e-3, 1e-4, 1e-5):
    fpr_pair = fp / n_pairs
    prec = prev * rec_pt / (prev * rec_pt + (1 - prev) * fpr_pair)
    print(f"  prevalence {prev:.0e}: precision {prec:.3f}")
print("NOTE: 32-latent scale only (496 pairs/SAE); production-width null")
print("calibration is an open item (see review response).")
