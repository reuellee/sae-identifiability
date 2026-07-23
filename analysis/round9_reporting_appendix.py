"""Round-9 REPORTING-ONLY appendix (dated 2026-07-23, added on results-stage
external review — GPT-5.6 required change #9).

Prints the registered descriptive readouts that the frozen scorer computed
into the CSV but did not surface: bootstrap CIs, signed bias, RMSE,
theta-sensitivity, automatic orientation (directed + unordered), tie rates,
P4 contributor counts, P5 eligibility + odds-formula prediction, MAE vs
REALIZED rho, h_B with an NA rule + background weight w_B, formation
arithmetic, and the operational-bias decomposition table (post-run
DIAGNOSTIC, not a prediction). It reads the frozen r9_runs.csv and modifies
NO endpoint, eligibility rule, bar, or verdict.
"""
import csv, math, os, statistics as st
from collections import defaultdict

import analyze_round9 as AZ

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "round9", "r9_runs.csv")
f = AZ.f

def cellsort(k):
    return (k[0], k[1], k[2])

def main():
    rows = list(csv.DictReader(open(CSV)))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["exp"], f(r["rho"]), f(r["sigma"]))].append(r)

    # ---------------- formation arithmetic
    sc_all = sum(len(v) for k, v in cells.items() if k[0] == "SC")
    sc_inc = sum(1 for k, v in cells.items() if k[0] == "SC"
                 for r in v if r["absorbed"] == "1")
    rc_all = sum(len(v) for k, v in cells.items() if k[0] == "RC")
    rc_inc = sum(1 for k, v in cells.items() if k[0] == "RC"
                 for r in v if r["absorbed"] == "1")
    print(f"Formation: SC {sc_inc}/{sc_all} included; RC {rc_inc}/{rc_all}; "
          f"total {sc_inc+rc_inc}/{sc_all+rc_all}")

    hdr = (f"\n{'cell':<20}{'bias_O':>8}{'RMSE_O':>8}{'CI_O_lo':>8}{'CI_O_hi':>8}"
           f"{'MAEvsREAL_M':>12}{'t002':>7}{'t010':>7}{'autoMAE':>8}{'unordMAE':>9}"
           f"{'ties':>6}{'nJ11c':>6}{'nS11c':>6}{'w_B':>7}{'h_B':>7}{'pred_O':>8}"
           f"{'obs_O':>8}{'resid':>7}")
    print(hdr)
    p5rows = []
    for k in sorted(cells, key=cellsort):
        exp, rho, sigma = k
        inc = [r for r in cells[k] if r["absorbed"] == "1"]
        if not inc:
            continue
        def col(c):
            return [v for v in (f(r.get(c)) for r in inc) if v is not None]
        d_o = col("rho_d")
        bias = st.mean(x - rho for x in d_o)
        rmse = math.sqrt(st.mean((x - rho) ** 2 for x in d_o))
        lo, hi = AZ.boot_ci(d_o, rho)
        # mechanism MAE vs realized rho (per-run pairing)
        mv = [abs(f(r["m_rho_d"]) - f(r["rho_real"])) for r in inc
              if f(r.get("m_rho_d")) is not None and f(r.get("rho_real")) is not None]
        mvr = st.mean(mv) if mv else float("nan")
        t002 = st.mean(abs(x - rho) for x in col("rho_d_t002")) if col("rho_d_t002") else float("nan")
        t010 = st.mean(abs(x - rho) for x in col("rho_d_t010")) if col("rho_d_t010") else float("nan")
        auto = col("rho_d_auto")
        amae = st.mean(abs(x - rho) for x in auto) if auto else float("nan")
        umae = st.mean(min(abs(x - rho), abs(1 - x - rho)) for x in auto) if auto else float("nan")
        ties = st.mean(col("ties")) if col("ties") else float("nan")
        nj = sum(1 for r in inc if (f(r.get("n_j11")) or 0) >= 100)
        ns = sum(1 for r in inc if (f(r.get("n_s11")) or 0) >= 100)
        # background weight from committed counts: B-active = all-active - JS-active
        wbs, hbs, preds, obss = [], [], [], []
        for r in inc:
            allact = sum(f(r[c]) or 0 for c in ("n11", "n10", "n01"))
            jsact = sum(f(r[c]) or 0 for c in ("m_n11", "m_n10", "m_n01"))
            hb = f(r.get("h_B"))
            if allact and allact > jsact and hb is not None:
                w = (allact - jsact) / allact
                wbs.append(w); hbs.append(hb)
                m = f(r.get("m_rho_d"))
                if m is not None:
                    preds.append((1 - w) * m + w * hb)
                    obss.append(f(r["rho_d"]))
        w_B = st.mean(wbs) if wbs else float("nan")
        h_B = st.mean(hbs) if len(hbs) >= 5 else float("nan")   # NA when B-active mass negligible
        pred = st.mean(preds) if preds else float("nan")
        obs = st.mean(obss) if obss else float("nan")
        resid = obs - pred if (preds and obss) else float("nan")
        def n(x, w=7, p=3):
            return f"{x:>{w}.{p}f}" if x == x else f"{'NA':>{w}}"
        print(f"{exp} r={rho} s={sigma:<6}{bias:>8.4f}{rmse:>8.4f}{lo:>8.4f}{hi:>8.4f}"
              f"{n(mvr,12,4)}{n(t002)}{n(t010)}{n(amae,8)}{n(umae,9)}"
              f"{ties:>6.1f}{nj:>6}{ns:>6}{n(w_B)}{n(h_B)}{n(pred,8,4)}"
              f"{n(obs,8,4)}{n(resid,7,4)}")
        # ---------------- P5 (registered): eligibility + odds-formula prediction
        a0 = st.mean(col("a0")) if col("a0") else None
        g1 = st.mean(col("g1")) if col("g1") else None
        q01 = st.mean(col("q01_B")) if col("q01_B") else 0.0
        q10 = st.mean(col("q10_B")) if col("q10_B") else 0.0
        if a0 is not None and g1 is not None:
            sym_leak = abs(a0 - g1) <= 0.10
            qsum = q01 + q10
            sym_bg = (qsum == 0) or (abs(q01 - q10) <= 0.2 * qsum)
            r_par = 0.25 if exp == "SC" else 0.4
            rJ, rS, rB = r_par * rho, r_par * (1 - rho), 1 - r_par
            den = rJ * (1 - g1) + rS * (1 - a0) + rB * (q01 + q10)
            xpred = (rJ * (1 - g1) + rB * q01) / den if den > 0 else float("nan")
            xobs = st.mean(col("rho_x")) if col("rho_x") else float("nan")
            p5rows.append((k, sym_leak, sym_bg, xpred, xobs, xobs - rho))
    print("\nP5 (registered 0.10 descriptive bound; eligible = leak-symmetric AND bg-exclusive-symmetric):")
    for k, sl, sb, xp, xo, xb in p5rows:
        elig = "ELIGIBLE" if (sl and sb) else f"not-eligible(leak={sl},bg={sb})"
        print(f"  {k}: {elig}  pred_rho_x={xp:.4f} obs={xo:.4f} bias={xb:+.4f}"
              f"{'  <=0.10 OK' if (sl and sb and abs(xb) <= 0.10) else ''}")

if __name__ == "__main__":
    main()
