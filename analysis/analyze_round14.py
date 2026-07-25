"""Round 14 evaluator (FROZEN AT LOCK) — does the absorbed state have a carrier?

Transcribes notes/prereg-round14-carrier.md. Committed before any round-14 number
was computed, so the criteria cannot be fitted.

Gate 1 short-circuits everything: the harness must reproduce round 13b's
rate_family per SAE to within 0.002, else nothing below is interpretable.

P1 (PRIMARY) compensation: carrier activation on A vs C, paired by (SAE,letter).
    CONFIRMED (absorption) if CI lower > 0; FALSIFIED-DIRECTION (loss) if upper < 0.
P2 (PRIMARY) consistency: top-1 carrier share vs a random-direction null.
    CONFIRMED if the paired difference's CI lower > 0.
P3 carrier breadth, P4 concentration, P5 capacity contrast — reported, no bars.
"""
import json, os
import numpy as np

BOOT = 10_000
TOL_REPRO = 0.002
WIDTHS = [2048, 16384]


def boot_ci(x, reps=BOOT, seed=1):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ms = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def main():
    rows = json.load(open(os.environ.get("R14", "round14_results.json")))
    ref = json.load(open(os.environ["R13B"])) if os.environ.get("R13B") else None
    print(f"scored SAEs: {len(rows)}")

    # ---------------- gate 1: reproduce round 13b ----------------
    g1, worst = True, 0.0
    if ref:
        by = {(r["arch"], r["m"], r["seed"]): r["rate_family"] for r in ref}
        for r in rows:
            k = (r["arch"], r["m"], r["seed"])
            if k in by:
                d = abs(by[k] - r["rate_family"])
                worst = max(worst, d)
                if d > TOL_REPRO:
                    g1 = False
        print(f"\n=== gate 1 reproduction === max |d rate_family| = {worst:.4f} "
              f"(tol {TOL_REPRO}) -> {'OK' if g1 else 'FAIL'}")
        if not g1:
            print("  harness does not reproduce round 13b; STOPPING per prereg.")
            return
    else:
        print("\n=== gate 1 === SKIPPED (no R13B reference given)")

    # ---------------- gate 2: set sizes / exclusions ----------------
    print("\n=== gate 2 set sizes ===")
    for m in WIDTHS:
        for a in ("l1", "topk"):
            rs = [r for r in rows if r["arch"] == a and r["m"] == m]
            if not rs:
                continue
            cl = [v for r in rs for v in r["per_letter"].values()
                  if v.get("clean_latent")]
            sc = [v for v in cl if v.get("scored")]
            nA = [v["n_A"] for v in cl]
            print(f"  {a:5s} m={m:6d}: {len(sc)}/{len(cl)} (letter,SAE) cells scored "
                  f"(|A|>=20); median |A|={np.median(nA):.0f} "
                  f"|C|={np.median([v['n_C'] for v in cl]):.0f}")

    def cells(m=None, a=None):
        out = []
        for r in rows:
            if m and r["m"] != m:
                continue
            if a and r["arch"] != a:
                continue
            for L, v in r["per_letter"].items():
                if v.get("scored"):
                    out.append(v)
        return out

    def per_sae(fn, m=None, a=None):
        """Collapse to ONE value per SAE, then bootstrap over SAEs.

        The prereg specifies a bootstrap over SAEs. Resampling (SAE,letter) cells
        directly would treat ~400 cells as independent when they are 16 SAEs x 24
        shared letters -- doubly clustered -- and would report a CI several times
        too narrow. Collapsing first is the conservative reading and matches how
        rounds 12/13a/13b paired by seed.
        """
        vals = []
        for r in rows:
            if m and r["m"] != m:
                continue
            if a and r["arch"] != a:
                continue
            xs = [fn(v) for v in r["per_letter"].values() if v.get("scored")]
            xs = [x for x in xs if np.isfinite(x)]
            if xs:
                vals.append(float(np.mean(xs)))
        return vals

    # ---------------- P1 compensation ----------------
    print("\n=== P1 (PRIMARY): does the carrier take up the slack? ===")
    verdicts = {}
    for m in WIDTHS:
        d = per_sae(lambda x: x["act_A"] - x["act_C"], m=m)
        if not d:
            continue
        lo, hi = boot_ci(d, seed=11)
        print(f"  m={m:6d}: mean(act_A - act_C) = {np.mean(d):+.4f} "
              f"CI [{lo:+.4f},{hi:+.4f}] (n={len(d)} SAEs)")
        verdicts[m] = ("CONFIRMED (carrier compensates -> absorption)" if lo > 0
                       else "FALSIFIED-DIRECTION (carrier does NOT compensate -> loss)"
                       if hi < 0 else "NOT CONFIRMED (straddles 0)")
    for m in WIDTHS:
        if m in verdicts:
            print(f"  P1 @ m={m}: {verdicts[m]}")

    # ---------------- P2 consistency ----------------
    print("\n=== P2 (PRIMARY): is the carrier a repeated latent? ===")
    p2 = {}
    for m in WIDTHS:
        d = per_sae(lambda x: x["share"] - x["share_null"], m=m)
        if not d:
            continue
        sh = per_sae(lambda x: x["share"], m=m)
        nl = per_sae(lambda x: x["share_null"], m=m)
        lo, hi = boot_ci(d, seed=12)
        print(f"  m={m:6d}: share={np.mean(sh):.3f} null={np.mean(nl):.3f} "
              f"diff={np.mean(d):+.3f} CI [{lo:+.3f},{hi:+.3f}] (n={len(d)} SAEs)")
        p2[m] = "CONFIRMED" if lo > 0 else "NOT CONFIRMED"
    for m in WIDTHS:
        if m in p2:
            print(f"  P2 @ m={m}: {p2[m]}")

    # ---------------- P3 / P4 ----------------
    print("\n=== P3 (secondary): carrier breadth vs family ===")
    for m in WIDTHS:
        for a in ("l1", "topk"):
            v = cells(m=m, a=a)
            if not v:
                continue
            print(f"  {a:5s} m={m:6d}: rate(kappa)={np.mean([x['rate_kappa'] for x in v]):.4f} "
                  f"vs rate(F_L)={np.mean([x['rate_fam'] for x in v]):.4f}")

    print("\n=== P4 (secondary): concentration of letter-direction mass ===")
    for m in WIDTHS:
        v = cells(m=m)
        if not v:
            continue
        print(f"  m={m:6d}: conc on A (kappa) = {np.mean([x['conc_A'] for x in v]):.3f} "
              f"| conc on C (top family) = {np.mean([x['conc_C'] for x in v]):.3f}")

    # ---------------- P5 capacity contrast ----------------
    print("\n=== P5 (secondary): capacity contrast ===")
    if all(m in verdicts for m in WIDTHS):
        print(f"  P1 verdict m=2048: {verdicts[2048]}")
        print(f"  P1 verdict m=16384: {verdicts[16384]}")
        if verdicts[2048].split()[0] != verdicts[16384].split()[0]:
            print("  *** WIDTH-DEPENDENT: the 'absorbed' label does not mean the same")
            print("      thing at both widths; rounds 12/13a/13b measure different")
            print("      things at different m. ***")

    print("\n=== REGISTERED VERDICTS ===")
    print(f"  gate 1 reproduction: {g1}")
    for m in WIDTHS:
        if m in verdicts:
            print(f"  P1 @ m={m:5d}: {verdicts[m]}")
            print(f"  P2 @ m={m:5d}: {p2.get(m)}")


if __name__ == "__main__":
    main()
