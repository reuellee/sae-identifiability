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

print("\n=== D4 (G4 end-to-end rho from detected pair) ===")
errs = [abs(r["rho_hat"] - r["rho"]) for r in flA
        if not math.isnan(r.get("rho_hat", float("nan")))]
for rho in sorted(set(r["rho"] for r in flA)):
    c = [r for r in flA if r["rho"] == rho]
    e = [abs(x["rho_hat"] - rho) for x in c if not math.isnan(x["rho_hat"])]
    print(f"  rho={rho:.2f}: rho_hat={np.mean(g(c,'rho_hat')):.4f} |err|={np.mean(e):.4f} (n={len(e)})")
m4 = np.mean(errs) if errs else float("nan")
print(f"mean |rho_hat - rho| = {m4:.4f}  (pass <= 0.03)")
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
