"""Round-12 registered scoring: real-model first-letter absorption, L1 vs TopK.

Reads results/real/sae_*_fl.json (from real_firstletter.py MODE=score) for both
architectures and >=5 seeds each. Scores:
  P1 (PRIMARY): mean absorption_rate(L1) - mean(TopK) at the primary theta=0;
                sign + seed-bootstrap 95% CI. GATED on matched L0 (both arches'
                mean held-out L0 in [28,36], widen [24,40], disclosed). theta=0
                is the ONLY matched point and is CONSERVATIVE for L1, so L1>TopK
                here is the strong result. Robustness: absorption on the
                INTERSECTION of letters both arms score cleanly (removes the
                different-letter-subset confound, review Finding 3). The theta>0
                grid is DESCRIPTIVE only -- unmatched and biased toward L1 (an
                upper bracket), never a confirmation gate (review Finding 2).
  P2 (DESCRIPTIVE attribution, NOT a causal-validity bar): concentration =
                cos(top-magnitude-carrier recon, wL) - cos(random-firing recon,
                wL), pooled; bootstrap 95% CI > 0 => the absorbed letter is
                CONCENTRATED in the word's dominant latents. Magnitude-normalized
                (cosine), so it is not the "recon contains the present letter"
                near-tautology (review Finding 1). The real causal test is the
                deferred Chanin forward-pass. specificity (d_true - d_other) is
                reported but is near-guaranteed positive -> descriptive only.
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
                  f"n_letters={r.get('n_letters_scored')} conc={r.get('causal_conc_mean')} "
                  f"spec={r.get('causal_spec_mean')} n_causal={r.get('n_causal')}")
    a_l1 = [r["absorption_rate"] for r in l1]
    a_tk = [r["absorption_rate"] for r in tk]

    # ---- Finding-3 check: are the two arms scored on the same letters? ----
    nl_l1 = [r.get("n_letters_scored", 0) for r in l1]
    nl_tk = [r.get("n_letters_scored", 0) for r in tk]
    print(f"\n=== letters-scored (selection-bias check) ===")
    print(f"  n_letters_scored  L1 mean={st.mean(nl_l1):.1f} {nl_l1}   TopK mean={st.mean(nl_tk):.1f} {nl_tk}")
    if len(nl_l1) >= 2 and len(nl_tk) >= 2:
        nlo, nhi = boot_diff(nl_l1, nl_tk)
        print(f"  diff (L1-TopK) letters = {st.mean(nl_l1)-st.mean(nl_tk):+.1f}, 95% CI [{nlo:+.1f},{nhi:+.1f}]"
              + ("  <- arms score DIFFERENT letter sets; P1 confounded unless intersection agrees" if (nlo>0 or nhi<0) else ""))

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
    # combined P1 verdict: theta=0 CI direction AND matched L0. (theta=0 is the ONLY
    # matched point and is CONSERVATIVE for L1 -- see the grid below -- so L1>TopK
    # here is the strong result.)
    if lo > 0 and l0_ok:
        p1 = "CONFIRM (L1 absorbs more than TopK at matched theta=0 sparsity: CI excludes 0, L0 matched)"
    elif hi < 0 and l0_ok:
        p1 = "FALSIFIED (TopK absorbs more than L1 at matched theta=0: CI excludes 0 in the other direction, L0 matched)"
    elif not l0_ok:
        p1 = f"NOT CONFIRMED - L0 gate: {l0_note}; CI={p1_ci}"
    else:
        p1 = f"INCONCLUSIVE ({p1_ci})"
    print(f"  P1 VERDICT: {p1}")
    # theta grid: DESCRIPTIVE ONLY. theta>0 is UNMATCHED (L1's soft acts get zeroed
    # more than TopK's -> L1 effective-L0 drops -> absorption inflated toward L1), so
    # it is an UPPER bracket, not a robustness check. It can only DISCONFIRM: if the
    # gap shrinks/reverses as theta rises despite the pro-L1 bias, that argues against L1.
    print("  theta grid (DESCRIPTIVE; theta>0 unmatched, biased toward L1 -> upper bracket):")
    thetas = sorted({t for r in (l1 + tk) for t in (r.get("absorption_by_theta") or {})}, key=float)
    for t in thetas:
        gl = [r["absorption_by_theta"][t] for r in l1 if t in (r.get("absorption_by_theta") or {})]
        gt = [r["absorption_by_theta"][t] for r in tk if t in (r.get("absorption_by_theta") or {})]
        if gl and gt:
            print(f"    theta={t}: L1={st.mean(gl):.4f} TopK={st.mean(gt):.4f} diff={st.mean(gl)-st.mean(gt):+.4f}"
                  + ("  (matched)" if float(t) == 0 else "  (unmatched/biased+L1)"))

    # ---- P1 robustness: absorption on the INTERSECTION of letters both arms score
    # cleanly (Finding 3 -- removes the different-letter-subset confound) ----
    def majority_clean(rows):
        cnt = {}
        for r in rows:
            for L, v in (r.get("per_letter") or {}).items():
                if v.get("clean_latent"): cnt[L] = cnt.get(L, 0) + 1
        return {L for L, c in cnt.items() if c >= (len(rows) + 1) // 2}
    common = majority_clean(l1) & majority_clean(tk)
    def matched_abs(r, S):
        pres = sum((r["per_letter"][L].get("letter_present", 0)) for L in S
                   if L in r.get("per_letter", {}) and r["per_letter"][L].get("clean_latent"))
        absd = sum((r["per_letter"][L].get("absorbed", 0)) for L in S
                   if L in r.get("per_letter", {}) and r["per_letter"][L].get("clean_latent"))
        return absd / pres if pres else None
    mi_l1 = [x for x in (matched_abs(r, common) for r in l1) if x is not None]
    mi_tk = [x for x in (matched_abs(r, common) for r in tk) if x is not None]
    print(f"  intersection-matched letters ({len(common)}): {sorted(common)}")
    if len(mi_l1) >= 2 and len(mi_tk) >= 2:
        milo, mihi = boot_diff(mi_l1, mi_tk)
        p1_matched = ("L1>TopK (holds on common letters)" if milo > 0 else
                      "TopK>L1 (common letters)" if mihi < 0 else "straddles 0 on common letters")
        print(f"  matched-letter absorption  L1={st.mean(mi_l1):.4f} TopK={st.mean(mi_tk):.4f} "
              f"diff={st.mean(mi_l1)-st.mean(mi_tk):+.4f} 95% CI [{milo:+.4f},{mihi:+.4f}] -> {p1_matched}")
    else:
        p1_matched = "insufficient common-letter data"
        print(f"  matched-letter absorption: {p1_matched}")

    print("\n=== P2 (attribution, DESCRIPTIVE; real causal test = deferred Chanin forward-pass) ===")
    # PRIMARY P2 contrast: CONCENTRATION = cos(top-carrier recon, wL) - cos(random
    # firing recon, wL). Magnitude-normalized (cosine) -> NOT the "does recon contain
    # the present letter" near-tautology; asks if the letter is concentrated in the
    # dominant latents. Falsifiable (diffuse/tail letter -> <=0).
    conc = [r["causal_conc_mean"] for r in (l1 + tk) if r.get("causal_conc_mean") is not None]
    spec = [r["causal_spec_mean"] for r in (l1 + tk) if r.get("causal_spec_mean") is not None]
    if len(conc) >= 2:
        clo, chi = boot_mean(conc)
        p2 = ("CONCENTRATED (letter rides on the dominant latents; CI>0)" if clo > 0 else
              "DIFFUSE/TAIL (CI includes 0 or <0 -> letter not concentrated in dominant latents)")
        print(f"  concentration (falsifiable) mean={st.mean(conc):+.4f} 95% CI [{clo:+.4f},{chi:+.4f}] (n={len(conc)} SAEs)")
        if spec:
            print(f"  specificity (descriptive, near-guaranteed +): mean={st.mean(spec):+.4f}")
    else:
        p2 = "NO ATTRIBUTION DATA"
    print(f"  P2 (descriptive): {p2}")

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
    print(f"  P1 (PRIMARY): {p1}")
    print(f"     robustness: matched-letter -> {p1_matched}")
    print(f"  P2 (descriptive attribution): {p2}")
    print(f"  P3 (secondary): " + ("; ".join(p3_lines) if p3_lines else "no detector/GT overlap data"))

if __name__ == "__main__":
    main()
