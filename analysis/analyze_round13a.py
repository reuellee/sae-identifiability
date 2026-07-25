"""Round 13a evaluator — applies the registered criteria in
notes/prereg-round13a-family-endpoint.md to round13a_family.json.

Written and committed BEFORE the 13a results were read. Criteria are transcribed
from the prereg, not chosen to fit an outcome:
  gate 1 conformance (model/layer/m/theta/tau; L1 lam=4.5 per Amendment 1; TopK k=32)
  gate 2 seeds exactly {0..7} per arm, 16 SAEs, no duplicates
  gate 3 SHA256 provenance recorded for every scored file
  gate 4 recomputed SINGLE-latent rate reproduces frozen round-12 fl.json
         within 0.002 per SAE -- if this fails, NOTHING else is reported
  P1 pooled rate_family: SURVIVES if boot CI lower > 0.01;
     DISSOLVES if CI upper < 0.01; else INDETERMINATE
  P2 R^2(rate_family ~ family max sel) < 0.40 (comparator: single-latent 0.673)
  P3 top-3 letter share of absorbed instances (comparator 53% L1 / 73% TopK)
  P4 seed-paired L1-TopK diff on rate_family (SECONDARY, not confirmatory)
  P5 |F_L| distribution + paired arch diff
"""
import json, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "results", "real")
BOOT = 10_000
REG = dict(model="EleutherAI/pythia-1.4b", layer=12, m=16384, theta=0.0, tau=0.30,
           lam_l1=4.5, k_topk=32)
GATE4_TOL = 0.002
P1_BAR = 0.01
P2_BAR = 0.40
P2_COMPARATOR = 0.673


def boot_ci(x, reps=BOOT, seed=1):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    ms = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def main():
    rows = json.load(open(os.environ.get("R13A", os.path.join(RD, "round13a_family.json"))))
    print(f"scored SAEs: {len(rows)}")

    # ---------------- gate 1: conformance ----------------
    viol = []
    for r in rows:
        for f, want in (("model", REG["model"]), ("layer", REG["layer"]), ("m", REG["m"]),
                        ("theta", REG["theta"]), ("tau", REG["tau"])):
            if r.get(f) != want:
                viol.append(f"{r['sae']}: {f}={r.get(f)}!={want}")
        if r["arch"] == "l1" and r.get("lam") != REG["lam_l1"]:
            viol.append(f"{r['sae']}: lam={r.get('lam')}!={REG['lam_l1']}")
        if r["arch"] == "topk" and r.get("k") != REG["k_topk"]:
            viol.append(f"{r['sae']}: k={r.get('k')}!={REG['k_topk']}")
    g1 = not viol
    print(f"\n=== gate 1 conformance === {'OK' if g1 else 'VIOLATIONS'}")
    for v in viol:
        print("   ", v)

    # ---------------- gate 2: seeds ----------------
    g2 = True
    for arch in ("l1", "topk"):
        seeds = sorted(r["seed"] for r in rows if r["arch"] == arch)
        ok = seeds == list(range(8))
        g2 &= ok
        print(f"=== gate 2 seeds [{arch}] === {seeds} -> {'OK' if ok else 'MISMATCH'}")

    # ---------------- gate 3: provenance ----------------
    shas = {r["sae"]: r.get("sha256") for r in rows}
    g3 = all(bool(v) for v in shas.values()) and len(set(shas.values())) == len(rows)
    print(f"=== gate 3 provenance === all sha256 present & distinct: {'OK' if g3 else 'FAIL'}")

    # ---------------- gate 4: baseline reproduction ----------------
    print(f"\n=== gate 4 baseline reproduction (|d| <= {GATE4_TOL}) ===")
    g4 = True
    for r in rows:
        frozen_p = os.path.join(RD, r["sae"].replace(".pt", "_fl.json"))
        if not os.path.exists(frozen_p):
            print(f"  {r['sae']}: NO frozen fl.json -> cannot verify"); g4 = False; continue
        ref = json.load(open(frozen_p))["absorption_rate"]
        d = abs(r["rate_single"] - ref)
        ok = d <= GATE4_TOL
        g4 &= ok
        print(f"  {r['sae']}: recomputed={r['rate_single']:.4f} frozen={ref:.4f} "
              f"|d|={d:.4f} {'ok' if ok else 'MISMATCH'}")
    print(f"  gate 4 -> {'OK' if g4 else 'FAILED'}")
    if not g4:
        print("\nGATE 4 FAILED: the re-score harness does not reproduce the frozen "
              "round-12 result. Per the prereg, NOTHING else is reported.")
        return

    # ---------------- P1 ----------------
    print("\n=== P1 (PRIMARY): does absorption survive the family correction? ===")
    fam = np.array([r["rate_family"] for r in rows])
    single = np.array([r["rate_single"] for r in rows])
    lo, hi = boot_ci(fam)
    print(f"  pooled rate_single = {single.mean():.4f}")
    print(f"  pooled rate_family = {fam.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]")
    print(f"  reduction: {(1 - fam.mean()/max(single.mean(),1e-9))*100:.1f}% of the "
          f"single-latent endpoint is removed by scoring against the family")
    if lo > P1_BAR:
        p1 = f"SURVIVES (CI lower {lo:.4f} > {P1_BAR})"
    elif hi < P1_BAR:
        p1 = f"DISSOLVES (CI upper {hi:.4f} < {P1_BAR})"
    else:
        p1 = f"INDETERMINATE (CI [{lo:.4f},{hi:.4f}] spans the {P1_BAR} bar)"
    print(f"  P1 VERDICT: {p1}")
    for arch in ("l1", "topk"):
        v = np.array([r["rate_family"] for r in rows if r["arch"] == arch])
        s = np.array([r["rate_single"] for r in rows if r["arch"] == arch])
        print(f"    {arch:5s}: single={s.mean():.4f} family={v.mean():.4f}")

    # ---------------- P2 ----------------
    print("\n=== P2 (PRIMARY): is the family endpoint still a selectivity re-expression? ===")
    xs, ys = [], []
    for r in rows:
        for L, v in r["per_letter"].items():
            if v.get("clean_latent"):
                xs.append(v["fam_max_sel"]); ys.append(v["rate_family"])
    xs, ys = np.array(xs), np.array(ys)
    slope, inter = np.polyfit(xs, ys, 1)
    pred = slope * xs + inter
    r2 = 1 - ((ys - pred) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-12)
    print(f"  n cells={len(xs)}  rate_family = {slope:+.3f}*max_sel {inter:+.3f}  R^2={r2:.3f}")
    print(f"  comparator (single-latent endpoint) R^2 = {P2_COMPARATOR}")
    print(f"  P2 VERDICT: {'PASS' if r2 < P2_BAR else 'FAIL'} (bar R^2 < {P2_BAR})")

    # ---------------- P3 ----------------
    print("\n=== P3 (secondary): heterogeneity under the family endpoint ===")
    for arch in ("l1", "topk"):
        tot = {}
        for r in rows:
            if r["arch"] != arch:
                continue
            for L, v in r["per_letter"].items():
                if v.get("clean_latent"):
                    tot[L] = tot.get(L, 0) + v["absorbed_family"]
        if sum(tot.values()) == 0:
            print(f"  {arch:5s}: zero absorbed instances under the family endpoint")
            continue
        srt = sorted(tot.items(), key=lambda kv: -kv[1])
        share = sum(v for _, v in srt[:3]) / sum(tot.values())
        print(f"  {arch:5s}: top-3 {[k for k,_ in srt[:3]]} carry {share*100:.0f}% "
              f"(single-latent comparator: {'53' if arch=='l1' else '73'}%)")

    # ---------------- P4 ----------------
    print("\n=== P4 (secondary, NOT confirmatory): arch diff on the family endpoint ===")
    a = {r["seed"]: r["rate_family"] for r in rows if r["arch"] == "l1"}
    b = {r["seed"]: r["rate_family"] for r in rows if r["arch"] == "topk"}
    common = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in common])
    lo4, hi4 = boot_ci(d, seed=4)
    print(f"  paired L1-TopK diff = {d.mean():+.4f} 95% CI [{lo4:+.4f},{hi4:+.4f}] (n={len(d)})")
    print(f"  per-seed: {[round(x,4) for x in d]}")
    print("  CAVEAT: reuses round-12 weights+data. Cannot confirm an arch claim; sizes 13b.")

    # ---------------- P5 ----------------
    print("\n=== P5 (descriptive): split-family size |F_L| ===")
    fs = {}
    for arch in ("l1", "topk"):
        v = [x["fam_size"] for r in rows if r["arch"] == arch
             for x in r["per_letter"].values() if x.get("clean_latent")]
        fs[arch] = np.array(v)
        print(f"  {arch:5s}: mean |F_L|={fs[arch].mean():.2f} median={np.median(fs[arch]):.0f} "
              f"max={fs[arch].max()} (cap {rows[0]['fam_cap']})")
    pa = {}
    for arch in ("l1", "topk"):
        pa[arch] = {}
        for r in rows:
            if r["arch"] != arch:
                continue
            for L, v in r["per_letter"].items():
                if v.get("clean_latent"):
                    pa[arch].setdefault(L, []).append(v["fam_size"])
    common_L = sorted(set(pa["l1"]) & set(pa["topk"]))
    dl = np.array([np.mean(pa["l1"][L]) - np.mean(pa["topk"][L]) for L in common_L])
    lo5, hi5 = boot_ci(dl, seed=5)
    print(f"  paired-by-letter |F_L| diff (L1-TopK) = {dl.mean():+.2f} CI [{lo5:+.2f},{hi5:+.2f}] "
          f"({int((dl>0).sum())}/{len(dl)} letters larger for L1)")

    print("\n=== REGISTERED VERDICTS ===")
    print(f"  gates: conformance={g1} seeds={g2} provenance={g3} baseline_repro={g4}")
    print(f"  P1 (primary): {p1}")
    print(f"  P2 (primary): R^2={r2:.3f} -> {'PASS' if r2 < P2_BAR else 'FAIL'}")


if __name__ == "__main__":
    main()
