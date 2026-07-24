"""Round-12 registered scoring: real-model first-letter absorption, L1 vs TopK.

Reads results/real/sae_*_fl.json (from real_firstletter.py MODE=score) for both
architectures and the registered seeds each. Scores:
  P1 (PRIMARY): mean absorption_rate(L1) - mean(TopK) at the primary theta=0,
                as a PAIRED per-seed bootstrap (init is shared per seed -> valid
                pairs). GATED on: (a) config conformance (each fl.json produced
                with the registered theta/sel_min/n_carriers/model/layer/k/lam),
                (b) the registered seed set present in both arms, (c) matched L0
                (|mean L0_L1 - mean L0_TopK| small AND both near target), and
                (d) the L1>TopK sign surviving on the INTERSECTION of letters both
                arms score cleanly. Any gate failing => P1 not confirmed (reported).
                absorption now REQUIRES the letter to survive in the reconstruction
                (retention check in the metric), so it is not feature loss;
                loss_rate is reported alongside. With few seeds P1 is SUGGESTIVE.
  P2 (DESCRIPTIVE attribution, by ARCH): concentration = cos(top-carrier recon,
                wL) - cos(random-firing recon, wL), magnitude-normalized. > 0 =>
                letter concentrated in the dominant latents. Not a causal bar (real
                test = deferred Chanin forward-pass); may be inflated by in-fold
                probe leakage / tail-control asymmetry -- descriptive only.
  P3 (secondary): detector recall of ground-truth main-L-latents vs the
                opportunity baseline (involved fraction), enrichment, + precision.
Pure stdlib. Frozen at lock. Registered config below is the lock.
"""
import json, glob, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "results", "real")
BOOT = 10_000
# ---- REGISTERED CONFIG (the lock) ----
REG = dict(theta=0.0, sel_min=0.30, n_carriers=3, model="EleutherAI/pythia-1.4b",
           layer=12, k=32)
N_SEEDS = int(os.environ.get("N_SEEDS", "8"))
SEEDS = set(range(N_SEEDS))
L0_TARGET, L0_TOL, L0_BAND = 32.0, 3.0, (24.0, 40.0)   # matched: |dL0|<=TOL and both in band
LOCK_LAM = os.environ.get("LOCK_LAM")                   # frozen L1 lambda (from calibration); checked if set

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

def paired(l1, tk, key):
    """Per-seed paired values (L1[s]-TopK[s]) for seeds present in both arms."""
    a = {r["seed"]: r for r in l1}; b = {r["seed"]: r for r in tk}
    return [(s, a[s][key], b[s][key]) for s in sorted(set(a) & set(b))
            if a[s].get(key) is not None and b[s].get(key) is not None]

def conformance(rows, arch):
    bad = []
    for r in rows:
        for f in ("theta", "sel_min", "n_carriers", "layer"):
            if r.get(f) != REG[f]: bad.append(f"{r['_path']}: {f}={r.get(f)}!={REG[f]}")
        if r.get("model") != REG["model"]: bad.append(f"{r['_path']}: model={r.get('model')}")
        if arch == "topk" and r.get("k") != REG["k"]: bad.append(f"{r['_path']}: k={r.get('k')}")
        if arch == "l1" and LOCK_LAM is not None and str(r.get("lam")) != str(LOCK_LAM):
            bad.append(f"{r['_path']}: lam={r.get('lam')}!=LOCK_LAM {LOCK_LAM}")
    if arch == "l1" and len({r.get("lam") for r in rows}) > 1:
        bad.append(f"L1 lam not constant across seeds: {sorted({r.get('lam') for r in rows})}")
    return bad

def main():
    l1 = load("l1"); tk = load("topk")
    print(f"L1 SAEs={len(l1)}  TopK SAEs={len(tk)}")
    if not l1 or not tk:
        print("MISSING SAE fl.json files"); return

    # ---- gate: config conformance (frozen scorer must enforce the frozen design) ----
    viol = conformance(l1, "l1") + conformance(tk, "topk")
    conf_ok = not viol
    print(f"\n=== config-conformance gate ===")
    print("  OK" if conf_ok else "  VIOLATIONS:\n    " + "\n    ".join(viol))

    # ---- gate: registered seed set present in both arms ----
    s_l1 = sorted(r["seed"] for r in l1); s_tk = sorted(r["seed"] for r in tk)
    seeds_ok = (set(s_l1) == SEEDS and set(s_tk) == SEEDS
                and len(s_l1) == len(SEEDS) and len(s_tk) == len(SEEDS))
    print(f"=== seed gate (registered {sorted(SEEDS)}) ===")
    print(f"  L1 seeds={s_l1}  TopK seeds={s_tk}  -> {'OK' if seeds_ok else 'MISMATCH (P1 not confirmable)'}")

    for tag, rows in (("TopK", tk), ("L1", l1)):
        for r in rows:
            print(f"  {tag} {r['_path']}: absorption={r['absorption_rate']} loss={r.get('loss_rate')} "
                  f"L0={r.get('l0')} fvu={r.get('fvu')} n_letters={r.get('n_letters_scored')} "
                  f"conc={r.get('causal_conc_mean')} grid={r.get('absorption_by_theta')}")

    # ---- matched-L0 gate: arms must MATCH, not merely each sit in a band ----
    l0_l1 = [r["l0"] for r in l1 if r.get("l0") is not None]
    l0_tk = [r["l0"] for r in tk if r.get("l0") is not None]
    print(f"\n=== matched-L0 gate (|dL0|<={L0_TOL}, both in {L0_BAND}, target {L0_TARGET}) ===")
    if l0_l1 and l0_tk:
        ml1, mtk = st.mean(l0_l1), st.mean(l0_tk)
        dl0 = abs(ml1 - mtk); inband = all(L0_BAND[0] <= v <= L0_BAND[1] for v in (ml1, mtk))
        l0_ok = dl0 <= L0_TOL and inband
        print(f"  L1 mean L0={ml1:.1f} {(min(l0_l1),max(l0_l1))}   TopK mean L0={mtk:.1f} {(min(l0_tk),max(l0_tk))}")
        print(f"  |dL0|={dl0:.1f}  both-in-band={inband}  -> {'MATCHED' if l0_ok else 'UNMATCHED (P1 confounded)'}")
    else:
        l0_ok = False; print("  NO L0 DATA")

    # ---- loss vs absorption (sanity: absorption must not be feature loss) ----
    lr_l1 = [r.get("loss_rate") for r in l1 if r.get("loss_rate") is not None]
    lr_tk = [r.get("loss_rate") for r in tk if r.get("loss_rate") is not None]
    if lr_l1 and lr_tk:
        print(f"  loss_rate  L1 mean={st.mean(lr_l1):.4f}  TopK mean={st.mean(lr_tk):.4f} "
              f"(reported so absorption is not confounded by feature loss)")

    # ---- P1: PAIRED per-seed bootstrap ----
    print("\n=== P1 (PRIMARY: absorption L1 vs TopK, paired per-seed, theta=0 matched) ===")
    pr = paired(l1, tk, "absorption_rate")
    if len(pr) >= 2:
        diffs = [d - t for _, d, t in pr]
        m1 = st.mean(v for _, v, _ in pr); mt = st.mean(v for _, _, v in pr)
        lo, hi = boot_mean(diffs)
        allpos = all(x > 0 for x in diffs); allneg = all(x < 0 for x in diffs)
        print(f"  L1={m1:.4f} TopK={mt:.4f}  paired diff mean={st.mean(diffs):+.4f} 95% CI [{lo:+.4f},{hi:+.4f}] "
              f"(n_pairs={len(pr)}; per-seed diffs {[round(x,3) for x in diffs]})")
        print(f"  sign consistent across all seed-pairs: {'yes' if (allpos or allneg) else 'NO'} "
              f"(exact paired sign-test floor 2/2^{len(pr)}={2/2**len(pr):.4f})")
        p1_dir = "L1>TopK" if lo > 0 else ("TopK>L1" if hi < 0 else "straddles-0")
    else:
        lo = hi = 0; p1_dir = "insufficient pairs"; diffs = []
        print("  insufficient paired seeds")

    # ---- P1 matched-letter robustness: absorption on letters clean in EVERY SAE ----
    def clean_letters(rows):
        sets = [{L for L, v in (r.get("per_letter") or {}).items() if v.get("clean_latent")} for r in rows]
        return set.intersection(*sets) if sets else set()
    common = clean_letters(l1) & clean_letters(tk)
    def matched_abs(r, S):
        pres = sum(r["per_letter"][L]["letter_present"] for L in S)
        absd = sum(r["per_letter"][L]["absorbed"] for L in S)
        return absd / pres if pres else None
    prc = [(s, matched_abs(a, common), matched_abs(b, common))
           for s, a, b in [(s, {r["seed"]: r for r in l1}[s], {r["seed"]: r for r in tk}[s])
                           for s in sorted(set(r["seed"] for r in l1) & set(r["seed"] for r in tk))]]
    prc = [(s, d, t) for s, d, t in prc if d is not None and t is not None]
    print(f"  intersection letters (clean in EVERY SAE): {len(common)} {sorted(common)}")
    matched_sign_ok = False
    if len(prc) >= 2:
        cd = [d - t for _, d, t in prc]
        clo, chi = boot_mean(cd)
        same = (lo > 0 and clo > 0) or (hi < 0 and chi < 0)
        matched_sign_ok = same
        print(f"  matched-letter L1={st.mean(v for _,v,_ in prc):.4f} TopK={st.mean(v for _,_,v in prc):.4f} "
              f"diff={st.mean(cd):+.4f} CI [{clo:+.4f},{chi:+.4f}] -> sign {'HOLDS' if same else 'DOES NOT HOLD'} vs full")
    else:
        print("  insufficient common-letter data")

    # ---- combined P1 verdict ----
    gates = dict(conformance=conf_ok, seeds=seeds_ok, matched_L0=l0_ok, matched_letters=matched_sign_ok)
    if lo > 0 and all(gates.values()):
        p1 = "CONFIRM-SUGGESTIVE (L1 absorbs more than TopK at matched theta=0; all gates pass; few-seed => suggestive)"
    elif hi < 0 and gates["conformance"] and gates["seeds"] and gates["matched_L0"]:
        p1 = "FALSIFIED-DIRECTION (TopK absorbs more than L1; L0 matched)"
    else:
        failed = [k for k, v in gates.items() if not v]
        p1 = f"NOT CONFIRMED (dir={p1_dir}; failing gates: {failed or 'none, but CI straddles 0'})"
    print(f"  P1 gates: {gates}")
    print(f"  P1 VERDICT: {p1}")

    # ---- theta grid: DESCRIPTIVE (theta>0 unmatched, biased toward L1) ----
    print("  theta grid (DESCRIPTIVE; theta>0 unmatched/biased+L1 -> upper bracket):")
    thetas = sorted({t for r in (l1 + tk) for t in (r.get("absorption_by_theta") or {})}, key=float)
    for t in thetas:
        gl = [r["absorption_by_theta"][t] for r in l1 if t in (r.get("absorption_by_theta") or {})]
        gt = [r["absorption_by_theta"][t] for r in tk if t in (r.get("absorption_by_theta") or {})]
        if gl and gt:
            print(f"    theta={t}: L1={st.mean(gl):.4f} TopK={st.mean(gt):.4f} diff={st.mean(gl)-st.mean(gt):+.4f}"
                  + ("  (matched)" if float(t) == 0 else "  (unmatched/+L1)"))

    # ---- letters-scored (selection-bias descriptive) ----
    nl_l1 = [r.get("n_letters_scored", 0) for r in l1]; nl_tk = [r.get("n_letters_scored", 0) for r in tk]
    print(f"  n_letters_scored  L1 mean={st.mean(nl_l1):.1f} {nl_l1}  TopK mean={st.mean(nl_tk):.1f} {nl_tk}")

    # ---- P2: DESCRIPTIVE concentration, BY ARCH ----
    print("\n=== P2 (attribution, DESCRIPTIVE by arch; real causal test = deferred Chanin) ===")
    p2_lines = []
    for name, rows in (("L1", l1), ("TopK", tk)):
        c = [r["causal_conc_mean"] for r in rows if r.get("causal_conc_mean") is not None]
        if len(c) >= 2:
            clo, chi = boot_mean(c)
            verd = "CONCENTRATED(+)" if clo > 0 else "diffuse/tail"
            p2_lines.append(f"{name}: conc mean={st.mean(c):+.4f} CI [{clo:+.4f},{chi:+.4f}] -> {verd}")
        elif c:
            p2_lines.append(f"{name}: conc={c[0]:+.4f} (n=1)")
    for ln in p2_lines: print("  " + ln)
    print("  (caveat: probe direction is in-fold and the tail control is magnitude-asymmetric; descriptive only)")

    # ---- P3: recall vs opportunity baseline + precision ----
    print("\n=== P3 (detector recall of main-L-latents vs opportunity baseline) ===")
    fl_by_tag = {r["_path"].replace("_fl.json", ""): r for r in (l1 + tk)}
    p3_lines = []
    for arch in ("topk", "l1"):
        recs, bases, precs = [], [], []
        for p in sorted(glob.glob(os.path.join(RD, f"sae_*_{arch}_*_pairs.json"))):
            base = os.path.basename(p).replace("_pairs.json", "")
            fl = fl_by_tag.get(base); d = json.load(open(p))
            involved = set(d.get("involved_latents") or [])
            m = d.get("m") or (fl.get("m") if fl else None)
            if fl and fl.get("main_latents") and m:
                gt = set(fl["main_latents"].values())
                rec = len(gt & involved) / max(len(gt), 1)
                baseline = len(involved) / m                # chance recall of any latent set
                prec = len(gt & involved) / max(len(involved), 1)
                recs.append(rec); bases.append(baseline); precs.append(prec)
                enr = f"{rec/baseline:.2f}x" if baseline else "n/a"
                print(f"  {base} [{arch}]: recall={rec:.3f} baseline={baseline:.3f} "
                      f"enrichment={enr} precision={prec:.4f}")
        if recs:
            mb = st.mean(bases)
            enr = f"{st.mean(recs)/mb:.2f}x" if mb else "n/a"
            p3_lines.append(f"{arch}: recall={st.mean(recs):.3f} vs baseline={mb:.3f} (enrichment {enr})")
    for ln in p3_lines: print("  " + ln)

    print("\n=== REGISTERED VERDICTS ===")
    print(f"  gates: {gates}")
    print(f"  P1 (PRIMARY): {p1}")
    print(f"  P2 (descriptive): " + " | ".join(p2_lines))
    print(f"  P3 (secondary): " + ("; ".join(p3_lines) if p3_lines else "no detector/GT overlap"))

if __name__ == "__main__":
    main()
