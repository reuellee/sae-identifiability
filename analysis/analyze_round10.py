"""Round-10 registered scoring (prereg notes/prereg-topk-absorption.md).

Activation-aware: cell statistic = recovery_rate = mean(child_recovered) over
the 24 seeds (binary, always defined -> no missingness handling needed).
Scores P1 (two-atom crossover + q-scaling), P2 (two-atom capacity collapse),
P3 (overcomplete escape), P4 (TopK-vs-L1, descriptive). Bars FROZEN at lock;
never edit after results exist. Pure stdlib.
"""
import csv, math, os, random, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "round10", "r10_runs.csv")

# ---- bars (FROZEN at lock)
CROSS = 0.5                      # recovery_rate crossover reference
ABSORB_HI = 0.25                # recovery_rate <= 0.25 => absorbing cell
RECOVER_LO = 0.75               # recovery_rate >= 0.75 => recovered
P2_GAP = 0.5                    # k2-k1 gap pass
P2_FAIL_K2 = 0.5; P2_FAIL_GAP = 0.25
P3_ABSORB_FAIL = 0.25
BOOT = 10_000

def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def rate(rows):
    v = [f(r["child_recovered"]) for r in rows if f(r["child_recovered"]) is not None]
    return (st.mean(v) if v else float("nan")), len(v)

def boot_ci(rows, reps=BOOT, seed=7):
    v = [f(r["child_recovered"]) for r in rows if f(r["child_recovered"]) is not None]
    if len(v) < 2: return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choice(v) for _ in v) for _ in range(reps))
    return round(ms[int(0.025*reps)], 3), round(ms[int(0.975*reps)], 3)

def eps_mid(series):
    series = sorted((e, r) for e, r in series if r == r)
    for i in range(1, len(series)):
        e0, r0 = series[i-1]; e1, r1 = series[i]
        if r0 < CROSS <= r1:
            return e0 + (e1 - e0) * (CROSS - r0) / (r1 - r0)
    if series and all(r >= CROSS for _, r in series):
        return series[0][0]
    return float("inf")

def main():
    rows = list(csv.DictReader(open(CSV)))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], f(r["k"]), f(r["q"]), f(r["eps"]))].append(r)
    R = {}
    for key, rs in cells.items():
        rr, n = rate(rs)
        R[key] = (rr, n, boot_ci(rs))

    print("=== recovery_rate by cell ===")
    for key in sorted(R, key=lambda z: (z[0], z[1], z[2], z[3])):
        rr, n, ci = R[key]
        print(f"  {key[0]} k={key[1]:.0f} q={key[2]} eps={key[3]:<5}: "
              f"recov={rr:.3f} CI{ci} n={n}")

    verdicts = {}

    # ---------- P1 (two-atom crossover + q-scaling, arm A k=1)
    print("\n=== P1 (two-atom crossover + q-scaling, arm A k=1) — PRIMARY ===")
    emid = {}
    for q in (0.1, 0.2):
        series = [(e, R[("A", 1.0, q, e)][0]) for e in
                  (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60) if ("A", 1.0, q, e) in R]
        em = eps_mid(series); emid[q] = em
        top = max((r for _, r in series), default=float("nan"))
        print(f"  q={q}: eps_mid={'>0.60' if em==float('inf') else round(em,3)} "
              f"(2q={2*q}, band [{0.3*2*q:.2f},{2*q:.2f}]); max recov={top:.2f}")
    e1, e2 = emid[0.1], emid[0.2]
    censored = (e1 == float("inf")) or (e2 == float("inf"))
    if censored:
        p1 = "INCONCLUSIVE (eps_mid right-censored)"
    else:
        qscale = e2 > e1
        inband = (0.15*0.2 <= e1 <= 1.3*0.2) and (0.15*0.4 <= e2 <= 1.3*0.4)
        strict = (0.3*0.2 <= e1 <= 0.2) and (0.3*0.4 <= e2 <= 0.4)
        if (not qscale) or (not inband):
            p1 = "FALSIFIED"
        else:
            p1 = "PASS" if strict else "PASS(loose-band)"
    verdicts["P1"] = p1; print(f"  P1 VERDICT: {p1}")

    # ---------- P2 (two-atom capacity collapse, arm A)
    print("\n=== P2 (two-atom capacity collapse, arm A) — PRIMARY ===")
    predesig = [(0.2, 0.05), (0.2, 0.10), (0.1, 0.05)]
    results = []
    for q, eps in predesig:
        k1 = R.get(("A", 1.0, q, eps)); k2 = R.get(("A", 2.0, q, eps))
        if not (k1 and k2):
            results.append((q, eps, "MISSING")); continue
        r1, r2 = k1[0], k2[0]; gap = r2 - r1
        if r1 > ABSORB_HI:
            v = "vacuous(k1-not-absorbing)"
        elif r2 >= RECOVER_LO and gap >= P2_GAP:
            v = "pass"
        elif r2 <= P2_FAIL_K2 or gap <= P2_FAIL_GAP:
            v = "FAIL"
        else:
            v = "inconclusive"
        results.append((q, eps, v))
        print(f"  q={q} eps={eps}: k1_recov={r1:.2f} k2_recov={r2:.2f} gap={gap:.2f} -> {v}")
    fails = [(q, e) for q, e, v in results if v == "FAIL"]
    absorbing = [v for q, e, v in results if v not in ("vacuous(k1-not-absorbing)", "MISSING")]
    if fails:
        p2 = "FALSIFIED" + (" (localized eps=0.05; SGD-reachability diagnosis, verdict unchanged)"
                            if set(e for _, e in fails) == {0.05} else "")
    elif not absorbing:
        p2 = "INCONCLUSIVE (tight-budget absorption not instantiated at k=1)"
    elif all(v == "pass" for v in absorbing):
        p2 = "PASS"
    else:
        p2 = "INCONCLUSIVE/PARTIAL"
    verdicts["P2"] = p2; print(f"  P2 VERDICT: {p2}")

    # ---------- P3 (overcomplete escape, arm B TopK k=1)
    print("\n=== P3 (overcomplete escape, arm B m=16 TopK k=1) — SGD behaviour ===")
    oks = []
    for eps in (0.05, 0.10, 0.20, 0.40):
        c = R.get(("B", 1.0, 0.2, eps))
        if c: print(f"  eps={eps}: recov={c[0]:.2f} CI{c[2]}"); oks.append((eps, c[0]))
    e20 = dict(oks).get(0.20)
    ge = [r for e, r in oks if e >= 0.10]
    if e20 is not None and e20 <= P3_ABSORB_FAIL:
        p3 = "FALSIFIED (SGD did not escape at eps=0.20)"
    elif ge and all(r >= RECOVER_LO for r in ge):
        p3 = "PASS (SGD finds the child-recovering solution for eps>=0.10)"
    else:
        p3 = "PARTIAL/INCONCLUSIVE"
    verdicts["P3"] = p3; print(f"  P3 VERDICT: {p3}")

    # ---------- P4 (TopK vs L1, descriptive)
    print("\n=== P4 (TopK resists L1 absorption, descriptive) ===")
    for eps in (0.05, 0.10, 0.20, 0.40):
        b = R.get(("B", 1.0, 0.2, eps)); c = R.get(("C", 0.0, 0.2, eps))
        bt = b[0] if b else float("nan"); l1 = c[0] if c else float("nan")
        print(f"  eps={eps}: TopK recov={bt:.2f}  L1 recov={l1:.2f}  "
              f"delta(TopK-L1)={bt-l1:+.2f}")

    print("\n=== REGISTERED VERDICTS ===")
    for k in ("P1", "P2", "P3"):
        print(f"  {k}: {verdicts[k]}")
    print("  P4: descriptive (see table above)")
    return verdicts

if __name__ == "__main__":
    main()
