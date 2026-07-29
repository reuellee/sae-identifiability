"""Round 16 evaluator (FROZEN AT LOCK) — L0 axis at fixed width.

Transcribes the registered criteria in notes/prereg-round16-l0axis.md.
Committed before any round-16 SAE was trained, so the criteria cannot be
fitted.

Gates 1-4 + the per-arch manipulation check (MC). An arch failing MC
contributes no diffs anywhere; both failing -> round UNINTERPRETABLE.
Gate 3 (arch-matched L0 per target) gates P3 ONLY.

P1 (PRIMARY)  d_s = rate_family(k16,s) - rate_family(k64,s) per arch/seed;
              u_s = mean over MC-passing arches; 10k bootstrap over the 8 u_s.
              CONFIRMED if CI lower > 0 (D1's direction).
P2 (secondary, L1 only) mean |F_L| at k64 - k16 per seed; CI > 0 = CONFIRMED.
P3 (secondary) [L1-TopK]@k16 - [L1-TopK]@k64 per seed; registered power caveat.
P4 (CONFOUND CONTROL) rate_lost by cell; conservative-direction reading per
   the prereg.
D-control fam_fire_present / fam_fire_absent by cell (descriptive only).

Env: R16=fresh rows json  R16_INT=interior rows json (13b x8 cell, re-scored)
Self-test: python analyze_round16.py --selftest   (exits nonzero on mismatch)
"""
import json, os, re, sys
import numpy as np

BOOT = 10_000
TARGETS = (16, 64)
SMALLK, LARGEK = 16, 64
REG = dict(model="EleutherAI/pythia-1.4b", layer=12, theta=0.0, tau=0.30,
           m=16384, expansion=8)
MC_RATIO = 2.5
G3_TOL_FRAC, G3_BAND = 0.15, (0.75, 1.25)
NAME_RE = re.compile(r"^sae_pythia-1\.4b_L12_(l1|topk)_x8_k(16|64)_s([0-7])\.pt$")

# 13b x8 cell, pinned at lock from results/real/round13b_results.json
INTERIOR_PINS = {
    ("sae_pythia-1.4b_L12_l1_x8_s0.pt", "161d7c50cbb9d3a4"),
    ("sae_pythia-1.4b_L12_l1_x8_s1.pt", "7276d60379a9299a"),
    ("sae_pythia-1.4b_L12_l1_x8_s2.pt", "71e73659738ba4d4"),
    ("sae_pythia-1.4b_L12_l1_x8_s3.pt", "8f13269c7cc80ccd"),
    ("sae_pythia-1.4b_L12_l1_x8_s4.pt", "fa80a8319dec0515"),
    ("sae_pythia-1.4b_L12_l1_x8_s5.pt", "4b868915499515e9"),
    ("sae_pythia-1.4b_L12_l1_x8_s6.pt", "3915894f80b00ffe"),
    ("sae_pythia-1.4b_L12_l1_x8_s7.pt", "8bbf4f067ba5e108"),
    ("sae_pythia-1.4b_L12_topk_x8_s0.pt", "98fcc8c7a70c9216"),
    ("sae_pythia-1.4b_L12_topk_x8_s1.pt", "a96942e197486a29"),
    ("sae_pythia-1.4b_L12_topk_x8_s2.pt", "7451d68f9ac652b0"),
    ("sae_pythia-1.4b_L12_topk_x8_s3.pt", "c2fd51829627b0fa"),
    ("sae_pythia-1.4b_L12_topk_x8_s4.pt", "b8830afd5ff3ede5"),
    ("sae_pythia-1.4b_L12_topk_x8_s5.pt", "9220a77bc12772c8"),
    ("sae_pythia-1.4b_L12_topk_x8_s6.pt", "be53d73e6f56b3e3"),
    ("sae_pythia-1.4b_L12_topk_x8_s7.pt", "e8299e5d34ba269f"),
}


def boot_ci(x, reps=BOOT, seed=1):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ms = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def cell_of(row):
    m = NAME_RE.match(row.get("sae", ""))
    return (m.group(1), int(m.group(2))) if m else None


def mean_famsize(row):
    fs = [v["fam_size"] for v in row["per_letter"].values() if v.get("clean_latent")]
    return float(np.mean(fs)) if fs else float("nan")


def mean_ff(row, key):
    fs = [v[key] for v in row["per_letter"].values()
          if v.get("clean_latent") and key in v]
    return float(np.mean(fs)) if fs else float("nan")


def analyze(rows, interior):
    out = {}
    print(f"fresh rows: {len(rows)}   interior rows: {len(interior)}")
    get = lambda a, kt, s: next((r for r in rows if cell_of(r) == (a, kt)
                                 and r["seed"] == s), None)

    # ---------------- gate 1 conformance (fail-closed) ----------------
    viol = []
    for r in rows:
        for f, want in (("model", REG["model"]), ("layer", REG["layer"]),
                        ("theta", REG["theta"]), ("tau", REG["tau"]),
                        ("m", REG["m"]), ("expansion", REG["expansion"]),
                        ("words_model", REG["model"]), ("words_layer", REG["layer"]),
                        ("mb_k", 16)):
            if r.get(f) != want:
                viol.append(f"{r['sae']}: {f}={r.get(f)}")
        if not str(r.get("eval_src", "")).startswith("held-out"):
            viol.append(f"{r['sae']}: eval_src={r.get('eval_src')} (not held-out)")
        c = cell_of(r)
        if c and c[0] == "topk" and r.get("k") != c[1]:
            viol.append(f"{r['sae']}: blob k={r.get('k')} != filename k{c[1]}")
        if c and r.get("arch") != c[0]:
            viol.append(f"{r['sae']}: blob arch={r.get('arch')} != filename {c[0]}")
        m2 = NAME_RE.match(r.get("sae", ""))
        if m2 and r.get("seed") != int(m2.group(3)):
            viol.append(f"{r['sae']}: blob seed={r.get('seed')} != filename s{m2.group(3)}")
    lam_by_cell = {}
    for kt in TARGETS:
        lams = {r.get("lam") for r in rows if cell_of(r) == ("l1", kt)}
        if len(lams) > 1 or None in lams:
            viol.append(f"l1 k{kt}: lam not a single constant {sorted(map(str, lams))}")
        elif lams:
            lam_by_cell[kt] = next(iter(lams))
    if len(lam_by_cell) == 2 and not lam_by_cell[SMALLK] > lam_by_cell[LARGEK]:
        viol.append(f"lam(k16)={lam_by_cell.get(SMALLK)} !> lam(k64)="
                    f"{lam_by_cell.get(LARGEK)} (shrinkage monotonicity)")
    lam_file = os.environ.get("R16_LAMBDAS", "")
    lam_file_note = ""
    if lam_file and os.path.exists(lam_file):
        chosen = {}
        for line in open(lam_file):
            parts = line.split()
            if len(parts) == 2:
                chosen[int(parts[0])] = float(parts[1])
        for kt in TARGETS:
            if kt in chosen and kt in lam_by_cell and \
                    abs(lam_by_cell[kt] - chosen[kt]) > 1e-9:
                viol.append(f"l1 k{kt}: lam={lam_by_cell[kt]} != calibration "
                            f"output {chosen[kt]}")
    else:
        lam_file_note = ("    (R16_LAMBDAS calibration file ABSENT -- lambda-vs-"
                         "calibration equality unverifiable; named deficiency)")
    int_ids = {(r.get("sae"), r.get("sha256")) for r in interior}
    if interior and int_ids != INTERIOR_PINS:
        viol.append(f"interior cell: {len(int_ids - INTERIOR_PINS)} unpinned / "
                    f"{len(INTERIOR_PINS - int_ids)} missing vs the 16 locked hashes")
    g1 = not viol
    print(f"\n=== gate 1 conformance === {'OK' if g1 else 'VIOLATIONS'}")
    for v in viol[:20]:
        print("   ", v)
    if lam_file_note:
        print(lam_file_note)
    if not interior:
        print("    (interior cell ABSENT -- monotonicity/D-control profiles "
              "will lack the midpoint; named deficiency per the prereg, "
              "not gating)")

    # ---------------- gate 2 seeds / anti-contamination ----------------
    g2 = len(rows) == 32
    bad_names = [r["sae"] for r in rows if not NAME_RE.match(r["sae"])]
    g2 &= not bad_names
    for a in ("l1", "topk"):
        for kt in TARGETS:
            seeds = sorted(r["seed"] for r in rows if cell_of(r) == (a, kt))
            if seeds != list(range(8)):
                g2 = False
                print(f"  cell {a} k{kt}: seeds {seeds} MISMATCH")
    print(f"=== gate 2 seeds/naming === 32 SAEs, 4 cells x seeds 0-7, "
          f"names pinned -> {'OK' if g2 else 'FAIL'}")
    if bad_names:
        print(f"   off-pattern files (contamination risk): {bad_names[:5]}")

    # ---------------- gate 4 provenance ----------------
    shas = [r.get("sha256") for r in rows + interior]
    g4 = all(shas) and len(set(shas)) == len(shas)
    print(f"=== gate 4 provenance (16-hex sha256 prefixes, as recorded since "
          f"13a) === {'OK' if g4 else 'FAIL'}")
    global_ok = g1 and g2 and g4
    if not global_ok:
        print("*** GLOBAL GATE FAILURE: all registered verdicts will be "
              "reported UNINTERPRETABLE ***")

    # ---------------- per-cell table ----------------
    print("\n=== per-cell summary (fresh cells + interior reference) ===")
    print(f"  {'arch':5s} {'cell':>5s} {'lam':>7s} {'L0':>6s} {'FVU':>6s} "
          f"{'dead%':>7s} {'single':>7s} {'family':>7s} {'lost':>6s} "
          f"{'|F_L|':>6s} {'ff_pres':>8s} {'ff_abs':>7s}")
    cell = {}
    def add_cell(key, rs, label):
        if not rs:
            return
        cell[key] = dict(
            l0=float(np.mean([r["l0"] for r in rs])),
            fvu=float(np.mean([r["fvu"] for r in rs])),
            dead=float(np.mean([r["dead_pct"] for r in rs])),
            single=float(np.mean([r["rate_single"] for r in rs])),
            family=float(np.mean([r["rate_family"] for r in rs])),
            lost=float(np.mean([r["rate_lost"] for r in rs])),
            famsize=float(np.nanmean([mean_famsize(r) for r in rs])),
            ffp=float(np.nanmean([mean_ff(r, "fam_fire_present") for r in rs])),
            ffa=float(np.nanmean([mean_ff(r, "fam_fire_absent") for r in rs])),
            lam=rs[0].get("lam"))
        c = cell[key]
        print(f"  {key[0]:5s} {label:>5s} {str(c['lam']):>7s} {c['l0']:6.1f} "
              f"{c['fvu']:6.3f} {c['dead']*100:6.1f}% {c['single']:7.4f} "
              f"{c['family']:7.4f} {c['lost']:6.4f} {c['famsize']:6.2f} "
              f"{c['ffp']:8.3f} {c['ffa']:7.3f}")
    for a in ("l1", "topk"):
        for kt in TARGETS:
            add_cell((a, kt), [r for r in rows if cell_of(r) == (a, kt)], f"k{kt}")
        add_cell((a, "int"), [r for r in interior if r["arch"] == a], "int32")

    # ---------------- manipulation check (per arch) ----------------
    print(f"\n=== MANIPULATION CHECK (per arch: ratio >= {MC_RATIO}x AND both "
          f"cells in [0.75,1.25]*target) ===")
    mc = {}
    for a in ("l1", "topk"):
        if (a, SMALLK) in cell and (a, LARGEK) in cell:
            lo, hi = cell[(a, SMALLK)]["l0"], cell[(a, LARGEK)]["l0"]
            in_band = (0.75 * SMALLK <= lo <= 1.25 * SMALLK
                       and 0.75 * LARGEK <= hi <= 1.25 * LARGEK)
            mc[a] = hi >= MC_RATIO * lo and in_band
            print(f"  {a:5s}: L0 {lo:.1f} -> {hi:.1f} (x{hi/max(lo,1e-9):.2f}) "
                  f"band={'ok' if in_band else 'VIOLATED'} "
                  f"-> {'PASS' if mc[a] else 'FAIL (arch excluded everywhere)'}")
        else:
            mc[a] = False
            print(f"  {a:5s}: cell missing -> FAIL")
    any_mc = any(mc.values())
    if not any_mc:
        print("  MC FAILED IN BOTH ARCHES -> round UNINTERPRETABLE")

    # ---------------- gate 3 arch-matched L0 (gates P3 only) ----------------
    print(f"\n=== gate 3 arch-matched L0 per target (P3 only; "
          f"|d|<={G3_TOL_FRAC}*target, both in {G3_BAND}*target) ===")
    ok_target = {}
    for kt in TARGETS:
        if ("l1", kt) not in cell or ("topk", kt) not in cell:
            ok_target[kt] = False
            continue
        a, b = cell[("l1", kt)]["l0"], cell[("topk", kt)]["l0"]
        ok = (abs(a - b) <= G3_TOL_FRAC * kt
              and all(G3_BAND[0] * kt <= v <= G3_BAND[1] * kt for v in (a, b)))
        ok_target[kt] = ok
        print(f"  k{kt}: L1 L0={a:.1f} TopK L0={b:.1f} |d|={abs(a-b):.1f} "
              f"-> {'MATCHED' if ok else 'UNMATCHED (P3 excluded)'}")

    # ---------------- P1 ----------------
    print(f"\n=== P1 (PRIMARY): absorption falls as L0 rises (k16 vs k64, "
          f"seed-clustered) ===")
    diffs = {}
    for a in ("l1", "topk"):
        d = {s: get(a, SMALLK, s)["rate_family"] - get(a, LARGEK, s)["rate_family"]
             for s in range(8) if get(a, SMALLK, s) and get(a, LARGEK, s)}
        diffs[a] = d
        if d:
            lo, hi = boot_ci(np.array(list(d.values())), seed=11)
            note = "" if mc.get(a) else "   [MC FAILED -- uninterpretable, excluded]"
            print(f"  {a:5s}: diff={np.mean(list(d.values())):+.4f} "
                  f"CI [{lo:+.4f},{hi:+.4f}] (n={len(d)}){note}")
        prof = [cell[key]["family"] for key in ((a, SMALLK), (a, "int"), (a, LARGEK))
                if key in cell]
        if len(prof) == 3:
            mono = prof[0] > prof[1] > prof[2]
            print(f"         3-point profile k16->int32->k64: "
                  f"{[round(v, 4) for v in prof]}  monotone-falling={mono} "
                  f"(descriptive)")
    u = [float(np.mean([diffs[a][s] for a in ("l1", "topk")
                        if mc.get(a) and s in diffs[a]]))
         for s in range(8)
         if any(mc.get(a) and s in diffs[a] for a in ("l1", "topk"))]
    lo1, hi1 = boot_ci(np.array(u), seed=12)
    print(f"  POOLED (u_s over MC-passing arches): diff={np.mean(u):+.4f} "
          f"CI [{lo1:+.4f},{hi1:+.4f}] (n={len(u)} seeds)" if u else
          "  POOLED: no usable seeds")
    # D-gate (registered): the fall must not be dominated by indiscriminate
    # family firing. Pooled over MC-passing arches, cell means.
    passing = [a for a in ("l1", "topk") if mc.get(a)
               and (a, SMALLK) in cell and (a, LARGEK) in cell]
    if passing:
        rise_ffa = float(np.mean([cell[(a, LARGEK)]["ffa"] - cell[(a, SMALLK)]["ffa"]
                                  for a in passing]))
        d_abs = float(np.mean([cell[(a, SMALLK)]["family"] - cell[(a, LARGEK)]["family"]
                               for a in passing]))
        d_gate = rise_ffa <= 0.5 * d_abs
        print(f"  D-gate: rise(fam_fire_absent)={rise_ffa:+.4f} vs "
              f"0.5*fall(absorption)={0.5*d_abs:+.4f} -> "
              f"{'HOLDS' if d_gate else 'FAILS (mechanical account too large)'}")
    else:
        d_gate = False
    if not global_ok:
        p1 = "UNINTERPRETABLE (global gate failure: conformance/seeds/provenance)"
    elif not any_mc:
        p1 = "UNINTERPRETABLE (manipulation check failed in both arches)"
    elif not u:
        p1 = "UNINTERPRETABLE (no usable seed-level diffs)"
    elif lo1 > 0 and d_gate:
        p1 = "CONFIRMED (absorption falls as L0 rises -- D1's direction on Pythia)"
    elif lo1 > 0:
        p1 = ("CONFIRMED-BUT-MECHANICAL (CI excludes 0 but the D-gate fails: "
              "must NOT be read as confirming capacity dynamics)")
    elif hi1 < 0:
        p1 = "FALSIFIED-DIRECTION (absorption RISES with L0 on Pythia)"
    else:
        p1 = "NOT CONFIRMED (CI straddles 0)"
    print(f"  P1 VERDICT: {p1}")

    # ---- D-sensitivity (registered): matched-budget contrast ----
    print(f"\n=== D-SENSITIVITY (matched-budget top-16 mask, GPT pre-lock "
          f"P1.3): does the contrast survive a common firing budget? ===")
    dmb = {}
    for a in ("l1", "topk"):
        dmb[a] = {s: get(a, SMALLK, s)["rate_family_mb"] - get(a, LARGEK, s)["rate_family_mb"]
                  for s in range(8)
                  if get(a, SMALLK, s) and get(a, LARGEK, s)
                  and get(a, SMALLK, s).get("rate_family_mb") is not None
                  and get(a, LARGEK, s).get("rate_family_mb") is not None}
    umb = [float(np.mean([dmb[a][s] for a in ("l1", "topk")
                          if mc.get(a) and s in dmb[a]]))
           for s in range(8)
           if any(mc.get(a) and s in dmb[a] for a in ("l1", "topk"))]
    lomb = himb = float("nan")
    if umb:
        lomb, himb = boot_ci(np.array(umb), seed=15)
        print(f"  pooled matched-budget diff={np.mean(umb):+.4f} "
              f"CI [{lomb:+.4f},{himb:+.4f}] (n={len(umb)} seeds)")
    if not p1.startswith("CONFIRMED"):
        dsens = "N/A (P1 not confirmed natively)"
    elif not umb:
        dsens = "NOT LICENSED (no matched-budget rows)"
    elif lomb > 0:
        dsens = ("SURVIVES -- representational-change language is licensed "
                 "(the contrast holds under a common firing budget)")
    else:
        dsens = ("DOES NOT SURVIVE -- P1 establishes an L0 ASSOCIATION only; "
                 "representational-change language is NOT licensed")
    print(f"  D-SENSITIVITY: {dsens}")

    # ---------------- P2 ----------------
    print("\n=== P2 (secondary): split families grow with L0 "
          "(L1, letter-paired) ===")

    def clean_fams(r):
        return {L: v["fam_size"] for L, v in r["per_letter"].items()
                if v.get("clean_latent")}

    d2, npairs = [], []
    for s in range(8):
        r16s, r64s = get("l1", SMALLK, s), get("l1", LARGEK, s)
        if not (r16s and r64s):
            continue
        f16, f64 = clean_fams(r16s), clean_fams(r64s)
        both = sorted(set(f16) & set(f64))
        if both:
            d2.append(float(np.mean([f64[L] - f16[L] for L in both])))
            npairs.append(len(both))
    lo2, hi2 = boot_ci(np.array(d2), seed=13)
    print(f"  L1 paired |F_L| k64-k16: diff={np.mean(d2):+.3f} "
          f"CI [{lo2:+.3f},{hi2:+.3f}] (n={len(d2)} seeds, "
          f"paired letters/seed {npairs})" if d2 else "  L1: no usable seeds")
    tkp = [float(np.nanmean([mean_famsize(get("topk", kt, s)) for s in range(8)
                             if get("topk", kt, s)])) for kt in TARGETS]
    print(f"  TopK |F_L| by cell (descriptive, unpaired): "
          f"{[round(v, 2) for v in tkp]}")
    if not global_ok:
        p2 = "UNINTERPRETABLE (global gate failure)"
    elif not mc.get("l1"):
        p2 = "UNINTERPRETABLE (L1 failed MC)"
    elif not d2:
        p2 = "UNINTERPRETABLE (no usable seeds)"
    elif lo2 > 0:
        p2 = "CONFIRMED (families grow with L0, D1's co-movement)"
    elif hi2 < 0:
        p2 = "FALSIFIED-DIRECTION (families shrink as L0 rises)"
    else:
        p2 = "NOT CONFIRMED (CI straddles 0)"
    print(f"  P2 VERDICT: {p2}")

    # ---------------- P3 ----------------
    print("\n=== P3 (secondary): arch x L0 interaction ===")
    usable3 = ok_target.get(SMALLK, False) and ok_target.get(LARGEK, False) \
        and mc.get("l1") and mc.get("topk")
    lo3 = hi3 = float("nan")
    inter = []
    for s in range(8):
        cs = [get(a, kt, s) for a in ("l1", "topk") for kt in TARGETS]
        if any(c is None for c in cs):
            continue
        gap_s = get("l1", SMALLK, s)["rate_family"] - get("topk", SMALLK, s)["rate_family"]
        gap_l = get("l1", LARGEK, s)["rate_family"] - get("topk", LARGEK, s)["rate_family"]
        inter.append(gap_s - gap_l)
    if inter:
        lo3, hi3 = boot_ci(np.array(inter), seed=14)
        print(f"  interaction={np.mean(inter):+.4f} CI [{lo3:+.4f},{hi3:+.4f}] "
              f"(n={len(inter)})")
    if not global_ok:
        p3 = "UNINTERPRETABLE (global gate failure)"
    elif not usable3 or not inter:
        p3 = "NOT CONFIRMABLE (gate 3 or MC excluded a required cell)"
    elif lo3 > 0:
        p3 = "CONFIRMED (L1's absorption is more L0-sensitive)"
    elif hi3 < 0:
        p3 = "FALSIFIED-DIRECTION"
    else:
        p3 = "NOT CONFIRMED (CI straddles 0)"
    print(f"  P3 VERDICT: {p3}")
    if inter:
        print(f"  registered power caveat: n={len(inter)} seeds, CI half-width "
              f"{(hi3-lo3)/2:.4f}; a null here is weak evidence.")

    # ---------------- P4 confound control ----------------
    print("\n=== P4 (CONFOUND CONTROL): absorption vs outright loss ===")
    for a in ("l1", "topk"):
        keys = [(a, SMALLK), (a, "int"), (a, LARGEK)]
        fam = [cell[k]["family"] for k in keys if k in cell]
        lost = [cell[k]["lost"] for k in keys if k in cell]
        print(f"  {a:5s}: family {[round(v, 4) for v in fam]}   "
              f"lost {[round(v, 4) for v in lost]}   "
              f"family+lost {[round(f + l, 4) for f, l in zip(fam, lost)]}")
    ls = np.mean([cell[(a, SMALLK)]["lost"] for a in passing
                  if (a, SMALLK) in cell] or [np.nan])
    ll = np.mean([cell[(a, LARGEK)]["lost"] for a in passing
                  if (a, LARGEK) in cell] or [np.nan])
    if len(passing) < 2:
        print(f"  (P4 interpretive comparison restricted to MC-passing "
              f"arches: {passing or 'none'})")
    if p1.startswith("CONFIRMED") and ls > ll:
        print("  loss is higher at k16, i.e. the retention confound DEPRESSED the")
        print("  k16 endpoint: P1's confirmation survived a registered headwind.")
    if p1.startswith("FALSIFIED") and ls > ll:
        print("  *** RETENTION CONFOUND: loss rises at k16 while absorption fell.")
        print("      Per the prereg, P1 must NOT be read as 'the L0 direction")
        print("      reverses on Pythia' without the combined endpoint. ***")

    # ---------------- D-control ----------------
    print("\n=== D-control (descriptive): family fire rates by cell ===")
    for a in ("l1", "topk"):
        keys = [(a, SMALLK), (a, "int"), (a, LARGEK)]
        ffp = [round(cell[k]["ffp"], 3) for k in keys if k in cell]
        ffa = [round(cell[k]["ffa"], 3) for k in keys if k in cell]
        print(f"  {a:5s}: fam_fire_present {ffp}   fam_fire_absent {ffa}")
    print("  (rise in ff_absent comparable to the absorption drop -> mechanical")
    print("   fire-rate account; flat ff_absent with rising ff_present -> the")
    print("   budget buys letter-specific firing. Interpretation only.)")

    # ---------------- verdicts ----------------
    print("\n=== REGISTERED VERDICTS ===")
    print(f"  gates: conformance={g1} seeds={g2} provenance={g4} "
          f"interior={'present' if interior else 'ABSENT (named deficiency)'} "
          f"arch_matched={ {f'k{kt}': ok_target.get(kt) for kt in TARGETS} } "
          f"MC={ {a: mc.get(a) for a in ('l1', 'topk')} }")
    print(f"  P1 (PRIMARY): {p1}")
    print(f"  D-SENSITIVITY: {dsens}")
    print(f"  P2: {p2}")
    print(f"  P3: {p3}")
    out.update(g1=g1, g2=g2, g4=g4, global_ok=global_ok, mc=mc,
               ok_target=ok_target, p1=p1, p2=p2, p3=p3, dsens=dsens)
    return out


# ---------------------------------------------------------------- self-test
def _mkrow(arch, kt, seed, fam, l0, sha, lam=None, famsize=2.0,
           lost=0.03, name=None, blob_k=None, model=REG["model"], ffa=0.05,
           fam_mb=None, eval_src="held-out:acts_eval.pt"):
    fam_mb = fam if fam_mb is None else fam_mb
    per = {"a": dict(clean_latent=True, latent=1, sel=0.5, fam_size=famsize,
                     fam_max_sel=0.5, letter_present=100, absorbed_single=5,
                     absorbed_family=4, rate_single=fam + 0.02,
                     lost_family=3, rate_lost=lost, rate_family=fam,
                     fam_fire_present=0.8, fam_fire_absent=ffa,
                     mb_clean_latent=True, mb_fam_size=famsize,
                     mb_rate_family=fam_mb, mb_rate_lost=lost)}
    return dict(sae=name or f"sae_pythia-1.4b_L12_{arch}_x8_k{kt}_s{seed}.pt",
                sha256=sha, arch=arch,
                k=blob_k if blob_k is not None else (kt if arch == "topk" else 32),
                lam=lam,
                m=16384, seed=seed, theta=0.0, tau=0.30, fam_cap=32,
                model=model, layer=12, rate_single=fam + 0.02,
                rate_family=fam, rate_lost=lost, dead_pct=0.5, fvu=0.07,
                rate_family_mb=fam_mb, rate_lost_mb=lost, mb_k=16,
                eval_src=eval_src, words_model=model, words_layer=12,
                l0=l0, expansion=8, per_letter=per)


def _mkset(delta, l1_l0=(16.0, 64.0), topk_l0=(16.0, 64.0),
           fam64_minus_fam16=0.8, ffa64=0.05, mb_delta=None,
           l1_k16_extra=0.0):
    mb_delta = delta if mb_delta is None else mb_delta
    rows, i = [], 0
    for a in ("l1", "topk"):
        l0s = l1_l0 if a == "l1" else topk_l0
        for kt, l0 in zip(TARGETS, l0s):
            for s in range(8):
                i += 1
                base = 0.055 + 0.001 * s
                fam = base + (delta if kt == SMALLK else 0.0)
                if a == "l1" and kt == SMALLK:
                    fam += l1_k16_extra
                fam_mb = base + (mb_delta if kt == SMALLK else 0.0)
                fsz = 2.0 + (fam64_minus_fam16 if kt == LARGEK else 0.0)
                rows.append(_mkrow(a, kt, s, round(fam, 4), l0, f"sha{i:04d}",
                                   lam=(9.0 if kt == SMALLK else 2.0) if a == "l1" else None,
                                   famsize=fsz,
                                   ffa=(ffa64 if kt == LARGEK else 0.05),
                                   fam_mb=round(fam_mb, 4)))
    return rows


def selftest():
    import io, contextlib
    fails = []

    def run(rows, interior=()):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            v = analyze(rows, list(interior))
        return v

    v = run(_mkset(+0.03))
    if not v["p1"].startswith("CONFIRMED"):
        fails.append(f"planted +0.03 -> {v['p1']}")
    if not v["p2"].startswith("CONFIRMED"):
        fails.append(f"planted famsize growth -> {v['p2']}")
    if not (v["g1"] and v["g2"] and v["g4"]):
        fails.append("clean set failed a gate")

    v = run(_mkset(-0.03))
    if not v["p1"].startswith("FALSIFIED-DIRECTION"):
        fails.append(f"planted -0.03 -> {v['p1']}")

    # D-gate: effect present but fam_fire_absent rises by more than half of it
    v = run(_mkset(+0.03, ffa64=0.09))
    if not v["p1"].startswith("CONFIRMED-BUT-MECHANICAL"):
        fails.append(f"planted +0.03 with ffa rise 0.04 -> {v['p1']}")

    # MC band: ratio passes (12->30 is 2.5x) but k64 cell is out of band
    v = run(_mkset(+0.03, l1_l0=(12.0, 30.0)))
    if v["mc"]["l1"]:
        fails.append("MC band check missed an out-of-band L1 cell")

    # D-sensitivity: native effect survives / vanishes under the matched budget
    v = run(_mkset(+0.03, mb_delta=+0.03))
    if not v["dsens"].startswith("SURVIVES"):
        fails.append(f"mb contrast planted -> {v['dsens']}")
    v = run(_mkset(+0.03, mb_delta=0.0))
    if not v["dsens"].startswith("DOES NOT SURVIVE"):
        fails.append(f"mb contrast flat -> {v['dsens']}")

    # global-gate suppression: an off-pattern row must suppress every verdict
    rows = _mkset(+0.03)
    rows[0]["sae"] = "sae_pythia-1.4b_L12_l1_x8_s0.pt"
    v = run(rows)
    if v["g2"] or not v["p1"].startswith("UNINTERPRETABLE (global gate"):
        fails.append(f"global gate failure did not suppress P1 -> {v['p1']}")
    if not v["p2"].startswith("UNINTERPRETABLE (global gate"):
        fails.append(f"global gate failure did not suppress P2 -> {v['p2']}")

    # both-arch MC failure -> round uninterpretable
    v = run(_mkset(+0.03, l1_l0=(30.0, 34.0), topk_l0=(30.0, 34.0)))
    if not v["p1"].startswith("UNINTERPRETABLE (manipulation"):
        fails.append(f"both-arch MC failure -> {v['p1']}")

    # P3 sign branches (interaction planted via an L1-k16-only bump)
    v = run(_mkset(+0.03, l1_k16_extra=+0.02))
    if not v["p3"].startswith("CONFIRMED"):
        fails.append(f"planted positive interaction -> {v['p3']}")
    v = run(_mkset(+0.03, l1_k16_extra=-0.02))
    if not v["p3"].startswith("FALSIFIED-DIRECTION"):
        fails.append(f"planted negative interaction -> {v['p3']}")

    # eval_src gate: an in-cache row must fail conformance
    rows = _mkset(+0.03)
    rows[0]["eval_src"] = "in-cache (NOT held-out)"
    if run(rows)["g1"]:
        fails.append("gate 1 missed a non-held-out eval_src")

    # interior ABSENT is a named deficiency, not a gate failure
    v = run(_mkset(+0.03), interior=())
    if not v["g1"]:
        fails.append("interior absence wrongly failed gate 1")

    v = run(_mkset(0.0))
    if not v["p1"].startswith("NOT CONFIRMED"):
        fails.append(f"planted 0 -> {v['p1']}")

    # L1 calibration failure: L0 barely separated -> L1 MC fails, pooled= topk
    v = run(_mkset(+0.03, l1_l0=(28.0, 40.0)))
    if v["mc"]["l1"] or not v["mc"]["topk"]:
        fails.append("MC per-arch logic wrong")
    if not v["p1"].startswith("CONFIRMED"):
        fails.append(f"topk-only pooled -> {v['p1']}")
    if not v["p2"].startswith("UNINTERPRETABLE"):
        fails.append(f"P2 with failed L1 MC -> {v['p2']}")
    if not v["p3"].startswith("NOT CONFIRMABLE"):
        fails.append(f"P3 with failed L1 MC -> {v['p3']}")

    # missing seed -> gate 2
    rows = _mkset(+0.03)[:-1]
    if run(rows)["g2"]:
        fails.append("gate 2 missed a missing seed")

    # off-pattern name -> gate 2
    rows = _mkset(+0.03)
    rows[0]["sae"] = "sae_pythia-1.4b_L12_l1_x8_s0.pt"
    if run(rows)["g2"]:
        fails.append("gate 2 missed an off-pattern name")

    # topk blob-k mismatch -> gate 1
    rows = _mkset(+0.03)
    bad = next(r for r in rows if cell_of(r) == ("topk", 16))
    bad["k"] = 32
    if run(rows)["g1"]:
        fails.append("gate 1 missed a blob-k mismatch")

    # duplicate sha -> gate 4
    rows = _mkset(+0.03)
    rows[1]["sha256"] = rows[0]["sha256"]
    if run(rows)["g4"]:
        fails.append("gate 4 missed a duplicate sha")

    # interior pin mismatch -> gate 1
    rows = _mkset(+0.03)
    interior = [dict(_mkrow("l1", 16, s, 0.054, 31.8, f"badsha{s}",
                            name=f"sae_pythia-1.4b_L12_l1_x8_s{s}.pt", lam=4.5),
                     ) for s in range(8)]
    if run(rows, interior)["g1"]:
        fails.append("gate 1 missed an interior pin mismatch")

    # lam monotonicity violation -> gate 1
    rows = _mkset(+0.03)
    for r in rows:
        if cell_of(r) == ("l1", 16):
            r["lam"] = 1.0
    if run(rows)["g1"]:
        fails.append("gate 1 missed lam(k16) <= lam(k64)")

    if fails:
        print("SELFTEST FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("SELFTEST OK (22 branch checks)")


def main():
    rows = json.load(open(os.environ.get("R16", "round16_results.json")))
    ip = os.environ.get("R16_INT", "round16_interior.json")
    interior = json.load(open(ip)) if os.path.exists(ip) else []
    analyze(rows, interior)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
