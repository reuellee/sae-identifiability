"""Round-9 registered scoring (prereg notes/prereg-gating-corrected-rho.md).

Reads results/round9/r9_runs.csv; scores P1M/P2M, P1O/P2O, P3, P4, P5 as
registered. BARS below are the [AT-LOCK] values - frozen at the lock commit,
never edited after results exist. Pure stdlib.
"""
import csv, math, os, random, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "round9", "r9_runs.csv")

# ---- [AT-LOCK] bars (finalized from D1's error budget at the lock commit)
BARS = dict(
    mech_pass=None, mech_fail=None,        # P1M/P2M: rho_d-M MAE per cell
    op_pass=0.05, op_fail=0.15,            # P1O/P2O: rho_d-O MAE per cell
    p3_margin=0.05,                        # P3: MAE(rho_d-O) <= MAE(rho_c-O) - margin
    p3_cells=None,                         # P3 eligible cells, predefined at lock
    p4_pass=0.05, p4_fail=0.10,            # P4: cell-median delta_J / delta_S
    min_include=16, delta_min_n=100,
)
BOOT = 10_000

def f(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None

def mae(vals, target):
    return st.mean(abs(v - target) for v in vals)

def boot_ci(vals, target, reps=BOOT, seed=99):
    rng = random.Random(seed)
    ms = sorted(mae([rng.choice(vals) for _ in vals], target) for _ in range(reps))
    return ms[int(0.025 * reps)], ms[int(0.975 * reps)]

def main():
    rows = list(csv.DictReader(open(CSV)))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["exp"], f(r["rho"]), f(r["sigma"]))].append(r)
    verdicts = []
    print(f"{'cell':<22}{'n_inc':>6}{'excl':>6}  {'rho_d-M MAE':>12}{'rho_d-O MAE':>12}"
          f"{'rho_c-O MAE':>12}{'med dJ':>8}{'med dS':>8}")
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
        exp, rho, sigma = key
        rs = cells[key]
        inc = [r for r in rs if r["absorbed"] == "1"]
        n_inc, n_all = len(inc), len(rs)
        scoreable = n_inc >= BARS["min_include"]
        def col(c):
            return [v for v in (f(r.get(c)) for r in inc) if v is not None]
        stats = {}
        for c in ("m_rho_d", "rho_d", "rho_c", "rho_x", "m_rho_x", "rho_count"):
            v = col(c)
            stats[c] = dict(mae=mae(v, rho) if v else None,
                            bias=st.mean(x - rho for x in v) if v else None,
                            ci=boot_ci(v, rho) if len(v) > 1 else None, n=len(v))
        dJ = [f(r["delta_J"]) for r in inc
              if f(r.get("n_j11")) and f(r["n_j11"]) >= BARS["delta_min_n"]
              and f(r.get("delta_J")) is not None]
        dS = [f(r["delta_S"]) for r in inc
              if f(r.get("n_s11")) and f(r["n_s11"]) >= BARS["delta_min_n"]
              and f(r.get("delta_S")) is not None]
        med_dJ = st.median(dJ) if dJ else None
        med_dS = st.median(dS) if dS else None
        wmass = [ (1 - rho) * f(r["a0"]) * f(r["delta_S"]) + rho * f(r["g1"]) * f(r["delta_J"])
                  for r in inc
                  if None not in (f(r.get("a0")), f(r.get("delta_S")),
                                  f(r.get("g1")), f(r.get("delta_J"))) ]
        verdicts.append(dict(key=key, scoreable=scoreable, n_inc=n_inc,
                             excl=n_all - n_inc, stats=stats,
                             med_dJ=med_dJ, med_dS=med_dS,
                             wmass=st.mean(wmass) if wmass else None))
        def s(c):
            m = stats[c]["mae"]
            return f"{m:.4f}" if m is not None else "  -   "
        def d4(x):
            return f"{x:.4f}" if x is not None else "  -   "
        print(f"{exp} rho={rho} sig={sigma:<6}{n_inc:>4}/{n_all:<3}{n_all-n_inc:>4}  "
              f"{s('m_rho_d'):>12}{s('rho_d'):>12}{s('rho_c'):>12}"
              f"{d4(med_dJ):>8}{d4(med_dS):>8}"
              f"{'' if scoreable else '  UNSCOREABLE'}")

    print("\n== registered verdicts ==")
    def per_cell(pred, harness, col, pass_bar, fail_bar):
        outs = []
        for v in verdicts:
            if v["key"][0] != harness or not v["scoreable"]: continue
            m = v["stats"][col]["mae"]
            if m is None: outs.append((v["key"], "NO-DATA")); continue
            outs.append((v["key"], "pass" if m <= pass_bar else
                         ("FALSIFIED" if m > fail_bar else "inconclusive")))
        overall = ("FALSIFIED" if any(o == "FALSIFIED" for _, o in outs) else
                   "PASS" if outs and all(o == "pass" for _, o in outs) else
                   "INCONCLUSIVE/PARTIAL")
        print(f"{pred}: {overall}  " + "; ".join(f"{k[1]}/{k[2]}:{o}" for k, o in outs))
    if BARS["mech_pass"] is not None:
        per_cell("P1M", "RC", "m_rho_d", BARS["mech_pass"], BARS["mech_fail"])
        per_cell("P2M", "SC", "m_rho_d", BARS["mech_pass"], BARS["mech_fail"])
    per_cell("P1O", "RC", "rho_d", BARS["op_pass"], BARS["op_fail"])
    per_cell("P2O", "SC", "rho_d", BARS["op_pass"], BARS["op_fail"])
    # P3
    if BARS["p3_cells"]:
        outs = []
        for v in verdicts:
            if list(v["key"]) not in [list(c) for c in BARS["p3_cells"]] or not v["scoreable"]:
                continue
            d, c = v["stats"]["rho_d"]["mae"], v["stats"]["rho_c"]["mae"]
            outs.append((v["key"], "pass" if d is not None and c is not None
                         and d <= c - BARS["p3_margin"] else "FALSIFIED"))
        overall = ("PASS" if outs and all(o == "pass" for _, o in outs)
                   else "FALSIFIED" if outs else "NO-CELLS")
        print(f"P3: {overall}  " + "; ".join(f"{k}:{o}" for k, o in outs))
    # P4
    outs = []
    for v in verdicts:
        if not v["scoreable"]: continue
        worst = max(x for x in (v["med_dJ"], v["med_dS"]) if x is not None) \
            if (v["med_dJ"] is not None or v["med_dS"] is not None) else None
        if worst is None: outs.append((v["key"], "NO-DATA")); continue
        outs.append((v["key"], "pass" if worst <= BARS["p4_pass"] else
                     ("FALSIFIED" if worst > BARS["p4_fail"] else "inconclusive")))
    overall = ("FALSIFIED" if any(o == "FALSIFIED" for _, o in outs) else
               "PASS" if outs and all(o == "pass" for _, o in outs) else
               "INCONCLUSIVE/PARTIAL")
    print(f"P4: {overall}  " + "; ".join(f"{k[1]}/{k[2]}:{o}" for k, o in outs))
    # P5 descriptive
    print("\nP5 (descriptive) - rho_x-O bias in leak-symmetric, bg-symmetric cells:")
    for v in verdicts:
        inc_bias = v["stats"]["rho_x"]["bias"]
        print(f"  {v['key']}: rho_x bias={inc_bias if inc_bias is None else round(inc_bias,4)}"
              f" wmass={v['wmass'] if v['wmass'] is None else round(v['wmass'],4)}")
    return verdicts

if __name__ == "__main__":
    main()
