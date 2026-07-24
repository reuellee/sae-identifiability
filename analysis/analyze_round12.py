"""Round-12 registered scoring: real-model first-letter absorption, L1 vs TopK.

Reads results/real/sae_*_fl.json (from real_firstletter.py MODE=score) for both
architectures and >=5 seeds each. Scores:
  P1 (PRIMARY): mean absorption_rate(L1) - mean(TopK); sign + seed-bootstrap
                95% CI on the difference. Registered: CI excludes 0 with L1>TopK
                => confirm; CI excludes 0 with TopK>L1 => FALSIFIED; else inconclusive.
                GATED on (a) theta-sign stability across the grid
                (absorption_by_theta) and (b) matched L0: both arches' mean
                held-out L0 in [28,36] (widen [24,40], disclosed). A sign flip or
                unmatched L0 => P1 not confirmed (reported, not hidden).
  P2 (causal, NON-CIRCULAR): per-SAE causal_diff_mean = mean over absorbed
                instances of (drop in the TRUE letter's probe logit - drop in
                OTHER letters' logits) when the top-magnitude carrier latents are
                ablated. Carriers are chosen by activation magnitude, NOT by probe
                alignment, so this can fail. Pool the per-SAE means; bootstrap 95%
                CI > 0 => the carriers letter-specifically carry the first letter.
  P3 (secondary): RECALL of ground-truth main-L-latents in the detector's
                flagged set (involved_latents), by arch. Precision reported
                descriptively (the detector also flags splitting, so low
                precision is not a failure).
Pure stdlib. Frozen at lock.
"""
import json, glob, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "results", "real")
BOOT = 10_000

def load(arch):
    rows = []
    for p in sorted(glob.glob(os.path.join(RD, f"sae_*_{arch}_*_fl.json"))):
        d = json.load(open(p)); d["_path"] = os.path.basename(p); rows.append(d)
    return rows

def boot_diff(a, b, reps=BOOT, seed=0):
    rng = random.Random(seed)
    ds = sorted(st.mean(rng.choice(a) for _ in a) - st.mean(rng.choice(b) for _ in b)
                for _ in range(reps))
    return ds[int(0.025*reps)], ds[int(0.975*reps)]

def boot_mean(x, reps=BOOT, seed=1):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choice(x) for _ in x) for _ in range(reps))
    return ms[int(0.025*reps)], ms[int(0.975*reps)]

def main():
    l1 = load("l1"); tk = load("topk")
    print(f"L1 SAEs={len(l1)}  TopK SAEs={len(tk)}")
    if not l1 or not tk:
        print("MISSING SAE fl.json files"); return
    for tag, rows in (("TopK", tk), ("L1", l1)):
        for r in rows:
            print(f"  {tag} {r['_path']}: absorption={r['absorption_rate']} L0={r.get('l0')} "
                  f"grid={r.get('absorption_by_theta')} fvu={r.get('fvu')} "
                  f"n_letters={r.get('n_letters_scored')} causal_diff={r.get('causal_diff_mean')} "
                  f"(true={r.get('causal_true_mean')} other={r.get('causal_other_mean')} n={r.get('n_causal')})")
    a_l1 = [r["absorption_rate"] for r in l1]
    a_tk = [r["absorption_rate"] for r in tk]

    # ---- matched-L0 gate (P1 is only valid at matched sparsity) ----
    BAND, WIDE = (28.0, 36.0), (24.0, 40.0)
    l0_l1 = [r["l0"] for r in l1 if r.get("l0") is not None]
    l0_tk = [r["l0"] for r in tk if r.get("l0") is not None]
    print(f"\n=== matched-L0 gate (registered band {BAND}, widen {WIDE}) ===")
    if l0_l1 and l0_tk:
        ml1, mtk = st.mean(l0_l1), st.mean(l0_tk)
        rng_l1 = (min(l0_l1), max(l0_l1)); rng_tk = (min(l0_tk), max(l0_tk))
        print(f"  L1  mean L0={ml1:.1f} range={rng_l1}   TopK mean L0={mtk:.1f} range={rng_tk}")
        inband = lambda v, b: b[0] <= v <= b[1]
        if inband(ml1, BAND) and inband(mtk, BAND):
            l0_ok, l0_note = True, "MATCHED (both means in registered band)"
        elif inband(ml1, WIDE) and inband(mtk, WIDE):
            l0_ok, l0_note = True, "MATCHED-WIDENED (both means in widened band; disclosed)"
        else:
            l0_ok, l0_note = False, "UNMATCHED (a mean L0 is outside the band -> P1 confounded)"
    else:
        l0_ok, l0_note = False, "NO L0 DATA (cannot confirm matched sparsity)"
    print(f"  L0 gate: {l0_note}")

    print("\n=== P1 (PRIMARY: absorption L1 vs TopK, at primary theta) ===")
    m1, mt = st.mean(a_l1), st.mean(a_tk)
    lo, hi = boot_diff(a_l1, a_tk)
    print(f"  mean absorption  L1={m1:.4f} (n={len(a_l1)})  TopK={mt:.4f} (n={len(a_tk)})")
    print(f"  diff (L1-TopK) = {m1-mt:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        p1_ci = "L1>TopK (CI excludes 0)"
    elif hi < 0:
        p1_ci = "TopK>L1 (CI excludes 0, FALSIFIED direction)"
    else:
        p1_ci = "CI straddles 0 (inconclusive)"
    print(f"  P1 CI result: {p1_ci}")
    # theta-robustness: sign of (L1-TopK) at each grid theta
    thetas = sorted({t for r in (l1 + tk) for t in (r.get("absorption_by_theta") or {})},
                    key=float)
    signs = []
    for t in thetas:
        gl = [r["absorption_by_theta"][t] for r in l1 if t in (r.get("absorption_by_theta") or {})]
        gt = [r["absorption_by_theta"][t] for r in tk if t in (r.get("absorption_by_theta") or {})]
        if gl and gt:
            dsign = st.mean(gl) - st.mean(gt)
            signs.append(dsign)
            print(f"    theta={t}: L1={st.mean(gl):.4f} TopK={st.mean(gt):.4f} diff={dsign:+.4f}")
    theta_stable = all(x > 0 for x in signs) or all(x < 0 for x in signs)
    print(f"  theta-robustness: sign {'STABLE' if theta_stable else 'FLIPS across grid -> threshold artifact'}")
    # combined P1 verdict: needs CI(L1>TopK) AND theta-stable AND matched L0
    if lo > 0 and theta_stable and l0_ok:
        p1 = "CONFIRM (L1 absorbs more than TopK: CI excludes 0, theta-stable, L0 matched)"
    elif hi < 0 and theta_stable and l0_ok:
        p1 = "FALSIFIED (TopK absorbs more than L1: CI excludes 0 in the other direction, L0 matched)"
    elif not l0_ok:
        p1 = f"NOT CONFIRMED - L0 gate: {l0_note}; CI={p1_ci}"
    elif not theta_stable:
        p1 = f"NOT CONFIRMED - theta sign flips across grid (artifact); CI={p1_ci}"
    else:
        p1 = f"INCONCLUSIVE ({p1_ci})"
    print(f"  P1 VERDICT: {p1}")

    print("\n=== P2 (causal, NON-CIRCULAR: true-letter vs other-letter drop) ===")
    diffs = [r["causal_diff_mean"] for r in (l1 + tk) if r.get("causal_diff_mean") is not None]
    if len(diffs) >= 2:
        glo, ghi = boot_mean(diffs)
        p2 = ("CONFIRM (carriers letter-specifically carry the first letter)" if glo > 0
              else "NOT CONFIRMED (CI includes 0 -> carriers are letter-agnostic)")
        print(f"  per-SAE causal_diff mean={st.mean(diffs):+.4f} 95% CI [{glo:+.4f},{ghi:+.4f}] (n={len(diffs)} SAEs)")
    else:
        p2 = "NO CAUSAL DATA"
    print(f"  P2 VERDICT: {p2}")

    print("\n=== P3 (recall of main-L-latents in detector flagged set) ===")
    fl_by_tag = {r["_path"].replace("_fl.json", ""): r for r in (l1 + tk)}
    p3_lines = []
    for arch in ("topk", "l1"):
        recs = []
        for p in sorted(glob.glob(os.path.join(RD, f"sae_*_{arch}_*_pairs.json"))):
            base = os.path.basename(p).replace("_pairs.json", "")
            fl = fl_by_tag.get(base)
            d = json.load(open(p))
            involved = set(d.get("involved_latents") or [])
            if fl and fl.get("main_latents") and involved is not None:
                gt = set(fl["main_latents"].values())
                rec = len(gt & involved) / max(len(gt), 1)
                recs.append(rec)
                print(f"  {base} [{arch}]: main-L-latents={len(gt)} recalled={len(gt & involved)} "
                      f"recall={rec:.3f} (flagged={d.get('n_flagged')} per_Mpairs={d.get('flagged_per_million_pairs')})")
        if recs:
            p3_lines.append(f"{arch} mean main-L-latent recall={st.mean(recs):.3f} (n={len(recs)})")
    for ln in p3_lines: print(f"  {ln}")

    print("\n=== REGISTERED VERDICTS ===")
    print(f"  L0 gate: {l0_note}")
    print(f"  P1: {p1}")
    print(f"  P2: {p2}")
    print(f"  P3 (secondary): " + ("; ".join(p3_lines) if p3_lines else "no detector/GT overlap data"))

if __name__ == "__main__":
    main()
