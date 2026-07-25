"""Round 13b evaluator (FROZEN AT LOCK) — capacity sweep.

Transcribes the registered criteria in notes/prereg-round13b-capacity.md.
Committed before any 13b SAE was trained, so the criteria cannot be fitted.

Gates 1-4 + the manipulation check (MC). If MC fails, P1/P2 are reported
UNINTERPRETABLE regardless of what the endpoint does.

P1 rate_family(m=2048) - rate_family(m=16384), paired by seed. CONFIRMED if CI>0.
P2 interaction [L1-TopK]@2048 - [L1-TopK]@16384, paired by seed. CONFIRMED if CI>0.
   Registered as the least-powered test in the round; a null is weak evidence.
P3 mean |F_L| by width/arch.  P4 single-vs-family inflation by width.
P5 CONFOUND CONTROL: loss_rate by width. If absorption falls while loss rises,
   P1 must NOT be read as "capacity does not drive absorption".
"""
import json, os, re
import numpy as np

BOOT = 10_000
WIDTHS = [2048, 4096, 16384]
SMALL, LARGE = 2048, 16384
REG = dict(model="EleutherAI/pythia-1.4b", layer=12, theta=0.0, tau=0.30, k_topk=32)
L0_TOL, L0_BAND = 3.0, (24.0, 40.0)
MC_DEAD_DROP = 0.15
NAME_RE = re.compile(r"^sae_pythia-1\.4b_L12_(l1|topk)_x(1|2|8)_s([0-7])\.pt$")


def boot_ci(x, reps=BOOT, seed=1):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ms = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    def rank(a):
        o = np.argsort(a); r = np.empty(len(a), float); r[o] = np.arange(len(a), dtype=float)
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main():
    rows = json.load(open(os.environ.get("R13B", "round13b_results.json")))
    print(f"scored SAEs: {len(rows)}")
    get = lambda a, m, s: next((r for r in rows if r["arch"] == a and r["m"] == m
                                and r["seed"] == s), None)

    # ---------------- gate 1 conformance ----------------
    viol = []
    for r in rows:
        for f, want in (("model", REG["model"]), ("layer", REG["layer"]),
                        ("theta", REG["theta"]), ("tau", REG["tau"])):
            if r.get(f) != want:
                viol.append(f"{r['sae']}: {f}={r.get(f)}")
        if r["m"] != r["expansion"] * 2048:
            viol.append(f"{r['sae']}: m={r['m']} != expansion*2048")
        if r["arch"] == "topk" and r.get("k") != REG["k_topk"]:
            viol.append(f"{r['sae']}: k={r.get('k')}")
    for m in WIDTHS:                      # one constant lambda per L1 cell
        lams = {r.get("lam") for r in rows if r["arch"] == "l1" and r["m"] == m}
        if len(lams) > 1:
            viol.append(f"l1 m={m}: lam not constant {sorted(lams)}")
    g1 = not viol
    print(f"\n=== gate 1 conformance === {'OK' if g1 else 'VIOLATIONS'}")
    for v in viol[:20]:
        print("   ", v)

    # ---------------- gate 2 seeds / anti-contamination ----------------
    g2 = len(rows) == 48
    bad_names = [r["sae"] for r in rows if not NAME_RE.match(r["sae"])]
    g2 &= not bad_names
    for a in ("l1", "topk"):
        for m in WIDTHS:
            seeds = sorted(r["seed"] for r in rows if r["arch"] == a and r["m"] == m)
            ok = seeds == list(range(8))
            g2 &= ok
            if not ok:
                print(f"  cell {a} m={m}: seeds {seeds} MISMATCH")
    print(f"=== gate 2 seeds/naming === 48 SAEs, 6 cells x seeds 0-7, "
          f"names pinned -> {'OK' if g2 else 'FAIL'}")
    if bad_names:
        print(f"   off-pattern files (contamination risk): {bad_names[:5]}")

    # ---------------- gate 4 provenance ----------------
    shas = [r.get("sha256") for r in rows]
    g4 = all(shas) and len(set(shas)) == len(rows)
    print(f"=== gate 4 provenance === {'OK' if g4 else 'FAIL'}")

    # ---------------- per-cell table ----------------
    print("\n=== per-cell summary ===")
    print(f"  {'arch':5s} {'m':>6s} {'lam':>5s} {'L0':>6s} {'FVU':>6s} {'dead%':>7s} "
          f"{'live':>7s} {'single':>7s} {'family':>7s} {'lost':>6s} {'|F_L|':>6s}")
    cell = {}
    for a in ("l1", "topk"):
        for m in WIDTHS:
            rs = [r for r in rows if r["arch"] == a and r["m"] == m]
            if not rs:
                continue
            dead = np.mean([r["dead_pct"] for r in rs])
            fam = np.mean([np.mean([v["fam_size"] for v in r["per_letter"].values()
                                    if v.get("clean_latent")]) for r in rs])
            cell[(a, m)] = dict(
                dead=dead, live=m * (1 - dead), fvu=np.mean([r["fvu"] for r in rs]),
                l0=np.mean([r["l0"] for r in rs]),
                single=np.mean([r["rate_single"] for r in rs]),
                family=np.mean([r["rate_family"] for r in rs]),
                lost=np.mean([r["rate_lost"] for r in rs]), famsize=fam,
                lam=rs[0].get("lam"))
            c = cell[(a, m)]
            print(f"  {a:5s} {m:6d} {str(c['lam']):>5s} {c['l0']:6.1f} {c['fvu']:6.3f} "
                  f"{c['dead']*100:6.1f}% {c['live']:7.0f} {c['single']:7.4f} "
                  f"{c['family']:7.4f} {c['lost']:6.4f} {c['famsize']:6.2f}")

    # ---------------- gate 3 matched L0 within width ----------------
    print(f"\n=== gate 3 matched-L0 within width (|dL0|<={L0_TOL}, both in {L0_BAND}) ===")
    ok_width = {}
    for m in WIDTHS:
        if ("l1", m) not in cell or ("topk", m) not in cell:
            ok_width[m] = False; continue
        a, b = cell[("l1", m)]["l0"], cell[("topk", m)]["l0"]
        ok = abs(a - b) <= L0_TOL and all(L0_BAND[0] <= v <= L0_BAND[1] for v in (a, b))
        ok_width[m] = ok
        print(f"  m={m:5d}: L1 L0={a:.1f} TopK L0={b:.1f} |d|={abs(a-b):.1f} "
              f"-> {'MATCHED' if ok else 'UNMATCHED (excluded from P2)'}")

    # ---------------- manipulation check ----------------
    print(f"\n=== MANIPULATION CHECK (dead% must drop >={MC_DEAD_DROP*100:.0f}pp small vs large) ===")
    d_small = np.mean([cell[(a, SMALL)]["dead"] for a in ("l1", "topk") if (a, SMALL) in cell])
    d_large = np.mean([cell[(a, LARGE)]["dead"] for a in ("l1", "topk") if (a, LARGE) in cell])
    live_prof = [np.mean([cell[(a, m)]["live"] for a in ("l1", "topk") if (a, m) in cell])
                 for m in WIDTHS]
    mono = all(live_prof[i] < live_prof[i + 1] for i in range(len(live_prof) - 1))
    mc = (d_large - d_small) >= MC_DEAD_DROP and mono
    print(f"  dead% m={SMALL}: {d_small*100:.1f}%   m={LARGE}: {d_large*100:.1f}%   "
          f"drop={{{(d_large-d_small)*100:.1f}}}pp")
    print(f"  live latents by width {WIDTHS}: {[round(v) for v in live_prof]} "
          f"monotone={mono}")
    print(f"  MC -> {'PASS' if mc else 'FAIL (P1/P2 UNINTERPRETABLE)'}")

    # ---------------- P1 ----------------
    print(f"\n=== P1 (PRIMARY): absorption rises as capacity falls "
          f"(m={SMALL} vs m={LARGE}) ===")
    pooled = []
    for a in ("l1", "topk"):
        d = [get(a, SMALL, s)["rate_family"] - get(a, LARGE, s)["rate_family"]
             for s in range(8) if get(a, SMALL, s) and get(a, LARGE, s)]
        pooled += d
        lo, hi = boot_ci(np.array(d), seed=11)
        print(f"  {a:5s}: diff={np.mean(d):+.4f} CI [{lo:+.4f},{hi:+.4f}] (n={len(d)})")
        prof = [cell[(a, m)]["family"] for m in WIDTHS if (a, m) in cell]
        liv = [cell[(a, m)]["live"] for m in WIDTHS if (a, m) in cell]
        print(f"         width profile {[round(v,4) for v in prof]}  "
              f"spearman(live, family)={spearman(liv, prof):+.2f}")
    lo1, hi1 = boot_ci(np.array(pooled), seed=12)
    print(f"  POOLED: diff={np.mean(pooled):+.4f} CI [{lo1:+.4f},{hi1:+.4f}] (n={len(pooled)})")
    if not mc:
        p1 = "UNINTERPRETABLE (manipulation check failed)"
    elif lo1 > 0:
        p1 = "CONFIRMED (absorption higher under scarcity)"
    elif hi1 < 0:
        p1 = "FALSIFIED-DIRECTION (absorption LOWER under scarcity)"
    else:
        p1 = "NOT CONFIRMED (CI straddles 0)"
    print(f"  P1 VERDICT: {p1}")

    # ---------------- P2 ----------------
    print(f"\n=== P2 (PRIMARY): does the L1-TopK gap open under scarcity? ===")
    usable = ok_width.get(SMALL, False) and ok_width.get(LARGE, False)
    inter = []
    for s in range(8):
        cs = [get(a, m, s) for a in ("l1", "topk") for m in (SMALL, LARGE)]
        if any(c is None for c in cs):
            continue
        gap_s = get("l1", SMALL, s)["rate_family"] - get("topk", SMALL, s)["rate_family"]
        gap_l = get("l1", LARGE, s)["rate_family"] - get("topk", LARGE, s)["rate_family"]
        inter.append(gap_s - gap_l)
    lo2, hi2 = boot_ci(np.array(inter), seed=13)
    gs = cell[("l1", SMALL)]["family"] - cell[("topk", SMALL)]["family"] if ("l1", SMALL) in cell else float("nan")
    gl = cell[("l1", LARGE)]["family"] - cell[("topk", LARGE)]["family"] if ("l1", LARGE) in cell else float("nan")
    print(f"  gap@m={SMALL}: {gs:+.4f}   gap@m={LARGE}: {gl:+.4f}")
    print(f"  interaction={np.mean(inter):+.4f} CI [{lo2:+.4f},{hi2:+.4f}] (n={len(inter)})")
    print(f"  per-seed: {[round(x,4) for x in inter]}")
    if not mc:
        p2 = "UNINTERPRETABLE (manipulation check failed)"
    elif not usable:
        p2 = "NOT CONFIRMABLE (a required width failed the matched-L0 gate)"
    elif lo2 > 0:
        p2 = "CONFIRMED (gap opens under scarcity -> round-12 null was regime-bound)"
    elif hi2 < 0:
        p2 = "FALSIFIED-DIRECTION (gap closes/reverses under scarcity)"
    else:
        p2 = "NOT CONFIRMED (CI straddles 0)"
    print(f"  P2 VERDICT: {p2}")
    print(f"  registered power caveat: difference-of-differences on n={len(inter)} seeds, "
          f"CI half-width {(hi2-lo2)/2:.4f}. A null here is WEAK evidence against H1.")

    # ---------------- P3 / P4 / P5 ----------------
    print("\n=== P3 (secondary): split-family size vs capacity ===")
    for a in ("l1", "topk"):
        print(f"  {a:5s}: |F_L| by width {[round(cell[(a,m)]['famsize'],2) for m in WIDTHS if (a,m) in cell]}")

    print("\n=== P4 (secondary): single-latent inflation vs capacity ===")
    for a in ("l1", "topk"):
        inf = [(cell[(a, m)]["single"] - cell[(a, m)]["family"]) / max(cell[(a, m)]["single"], 1e-9)
               for m in WIDTHS if (a, m) in cell]
        print(f"  {a:5s}: inflation by width {[f'{v*100:.0f}%' for v in inf]} "
              f"(13a measured 25% at m=16384)")

    print("\n=== P5 (CONFOUND CONTROL): absorption vs outright loss ===")
    for a in ("l1", "topk"):
        fam = [cell[(a, m)]["family"] for m in WIDTHS if (a, m) in cell]
        lost = [cell[(a, m)]["lost"] for m in WIDTHS if (a, m) in cell]
        tot = [f + l for f, l in zip(fam, lost)]
        print(f"  {a:5s}: family {[round(v,4) for v in fam]}")
        print(f"         lost   {[round(v,4) for v in lost]}")
        print(f"         family+lost {[round(v,4) for v in tot]}")
    if np.mean(pooled) < 0:
        ls = np.mean([cell[(a, SMALL)]["lost"] for a in ("l1", "topk") if (a, SMALL) in cell])
        ll = np.mean([cell[(a, LARGE)]["lost"] for a in ("l1", "topk") if (a, LARGE) in cell])
        if ls > ll:
            print("  *** RETENTION CONFOUND FIRED: absorption falls under scarcity while loss")
            print("      RISES. Per the prereg, P1 must NOT be read as 'capacity does not")
            print("      drive absorption' -- scarcity converted absorption into loss. ***")

    print("\n=== REGISTERED VERDICTS ===")
    print(f"  gates: conformance={g1} seeds={g2} provenance={g4} "
          f"matched_L0={ {m: ok_width.get(m) for m in WIDTHS} } MC={mc}")
    print(f"  P1 (PRIMARY): {p1}")
    print(f"  P2 (PRIMARY): {p2}")


if __name__ == "__main__":
    main()
