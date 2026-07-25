"""Round-12 POST-HOC diagnosis: why is P1 a null?

*** EXPLORATORY / POST-HOC. NOT a registered result. ***
The registered primary (analysis/analyze_round12.py, frozen at lock 0722212) stands
as reported: P1 NOT CONFIRMED. Nothing here amends it. This script asks the
follow-up question the registered scorer cannot: *what is the null made of?*, in
order to generate a round-13 pre-registration. Every number below is
development-set / hypothesis-generating and must be re-tested on fresh data.

Inputs: the 16 in-config fl.json (clean set; the pythia-70m contaminant is absent).

Analyses
  A. POWER / EQUIVALENCE. The registered endpoint pairs at the SEED level (n=8).
     Each SAE also scores 24 letters, so a letter-level paired contrast has far
     more resolution. Report both, plus a TOST-style equivalence read: what
     effect sizes does the observed CI actually exclude? A null is only
     informative if it excludes the effect the theory predicts.
  B. HETEROGENEITY. How concentrated is absorption across letters? If a handful
     of letters carry it, the seed-mean is a poor summary and arch contrasts are
     dominated by which letters happen to misbehave.
  C. SPLITTING CONFOUND. SAEBench-style absorption fires when the letter's "main"
     latent fails to activate while the letter is still reconstructed. Feature
     SPLITTING produces exactly that signature without any parent/child
     absorption. `sel` (main-latent selectivity) is the splitting proxy already
     recorded. If rate ~ -sel dominates, the metric is substantially a splitting
     metric and the arch contrast is mediated, not confounded away.
  D. q-DEPENDENCE. Toy theory: eps*(lam,q) is INCREASING in q, so higher-
     frequency features should absorb more readily. `n` (instances scored) is the
     available frequency proxy. Weak test, reported as such.

Pure stdlib + numpy (no scipy on this box): Spearman, OLS and bootstraps are
implemented locally.
"""
import json, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "results", "real")
BOOT = 10_000
ARCHES = ("l1", "topk")


def load(arch):
    rows = []
    for p in sorted(glob.glob(os.path.join(RD, f"sae_pythia-1.4b_L12_{arch}_x8_s*_fl.json"))):
        d = json.load(open(p))
        d["_path"] = os.path.basename(p)
        rows.append(d)
    return rows


def spearman(x, y):
    """Spearman rho + a permutation p-value (no scipy)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _rank(x), _rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(0)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(2000)])
    p = float((np.abs(null) >= abs(rho)).mean())
    return rho, p


def _rank(a):
    order = np.argsort(a)
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    # average ties
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def boot_mean_ci(x, reps=BOOT, seed=1):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    ms = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def ols(X, y):
    """Least squares with HC0-ish SEs. X includes an intercept column."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    # HC0 sandwich (robust; letters are heterogeneous)
    S = (X * (resid ** 2)[:, None]).T @ X
    cov = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(cov))
    return beta, se


def main():
    data = {a: load(a) for a in ARCHES}
    letters = sorted(data["l1"][0]["per_letter"])
    print(f"SAEs: L1={len(data['l1'])} TopK={len(data['topk'])}   letters={len(letters)}")
    print("*** POST-HOC / EXPLORATORY — the registered P1 verdict is unchanged ***")

    # letter x seed x arch tensors of the scored counts
    absd, pres, sel, nn = {}, {}, {}, {}
    for a in ARCHES:
        absd[a] = np.array([[r["per_letter"][L]["absorbed"] for L in letters] for r in data[a]], float)
        pres[a] = np.array([[r["per_letter"][L]["letter_present"] for L in letters] for r in data[a]], float)
        sel[a] = np.array([[r["per_letter"][L]["sel"] for L in letters] for r in data[a]], float)
        nn[a] = np.array([[r["per_letter"][L]["n"] for L in letters] for r in data[a]], float)
    rate = {a: absd[a] / pres[a] for a in ARCHES}          # (seed, letter)

    # ---------------- A. power / equivalence ----------------
    print("\n=== A. POWER & EQUIVALENCE ===")
    # seed level (the registered unit): pooled over letters, as the scorer does
    seed_rate = {a: absd[a].sum(axis=1) / pres[a].sum(axis=1) for a in ARCHES}
    d_seed = seed_rate["l1"] - seed_rate["topk"]
    lo_s, hi_s = boot_mean_ci(d_seed)
    print(f"  seed-level (registered unit, n=8): diff={d_seed.mean():+.4f} CI [{lo_s:+.4f},{hi_s:+.4f}]")

    # letter level: average the per-seed paired diff within each letter -> 24 units
    d_letter = (rate["l1"] - rate["topk"]).mean(axis=0)     # per letter, seed-paired
    lo_l, hi_l = boot_mean_ci(d_letter, seed=2)
    print(f"  letter-level (n=24, seed-paired):  diff={d_letter.mean():+.4f} CI [{lo_l:+.4f},{hi_l:+.4f}]")
    pos = int((d_letter > 0).sum())
    print(f"  letters with L1>TopK: {pos}/24  (sign test is uninformative at {pos}/24)")

    base = (rate["l1"].mean() + rate["topk"].mean()) / 2
    print(f"  pooled base absorption rate = {base:.4f}")
    print("  equivalence read (is the null informative?):")
    for frac in (0.25, 0.50, 1.00):
        margin = frac * base
        excl_s = hi_s < margin and lo_s > -margin
        excl_l = hi_l < margin and lo_l > -margin
        print(f"    +/-{frac*100:3.0f}% of base (={margin:.4f}): "
              f"seed-level {'EXCLUDED' if excl_s else 'not excluded'} | "
              f"letter-level {'EXCLUDED' if excl_l else 'not excluded'}")

    # ---------------- B. heterogeneity ----------------
    print("\n=== B. HETEROGENEITY ACROSS LETTERS ===")
    for a in ARCHES:
        per_L = rate[a].mean(axis=0)
        tot_abs = absd[a].sum(axis=0)
        share = np.sort(tot_abs / tot_abs.sum())[::-1]
        top3 = [letters[i] for i in np.argsort(per_L)[::-1][:3]]
        print(f"  {a:5s}: rate min={per_L.min():.4f} max={per_L.max():.4f} "
              f"(ratio {per_L.max()/max(per_L.min(),1e-9):.0f}x)  "
              f"top-3 letters {top3} carry {share[:3].sum()*100:.0f}% of all absorbed instances")

    # ---------------- C. metric validity: how much of the endpoint IS selectivity? ----------------
    print("\n=== C. METRIC VALIDITY (endpoint vs main-latent selectivity) ===")
    print("  NOTE the construction:  sel[j] = P(fire_j | L) - P(fire_j | not L)")
    print("                          rate   ~ P(main latent MISSES | L present, retained)")
    print("  so rate ~ (1 - FPR) - sel ALGEBRAICALLY. They share the same fires[.,j] term.")
    print("  => a regression of rate on arch CONTROLLING for sel conditions on a function of")
    print("     the outcome and is INVALID. It is deliberately not reported here. Quantify")
    print("     instead how much of the endpoint the identity already accounts for:")
    S = np.concatenate([sel[a].ravel() for a in ARCHES])
    R = np.concatenate([rate[a].ravel() for a in ARCHES])
    slope, intercept = np.polyfit(S, R, 1)
    pred = slope * S + intercept
    r2 = 1 - ((R - pred) ** 2).sum() / ((R - R.mean()) ** 2).sum()
    print(f"  pooled (384 SAE-letter cells): rate = {slope:+.3f}*sel {intercept:+.3f}, R^2={r2:.3f}")
    print(f"    (exact identity would be slope -1.000, R^2 ~ 1; observed leaves {1-r2:.1%} residual)")
    for a in ARCHES:
        rho, p = spearman(sel[a].mean(axis=0), rate[a].mean(axis=0))
        print(f"    {a:5s}: spearman(sel, rate) = {rho:+.3f} (perm p={p:.4f}, n=24 letters)")
    print("  READ: ~2/3 of the endpoint's variance is a re-expression of 'does the top")
    print("        letter-latent fire'. That is what feature SPLITTING also moves. The")
    print("        endpoint does not, on its own, separate absorption from splitting.")

    # a VALID, non-tautological arch contrast: selectivity itself (seed-paired, letter-clustered)
    print("\n  --- valid arch contrast: selectivity itself (not conditioned on the outcome) ---")
    d_sel_letter = (sel["l1"] - sel["topk"]).mean(axis=0)     # per letter, seed-paired
    lo, hi = boot_mean_ci(d_sel_letter, seed=3)
    print(f"  mean sel: L1={sel['l1'].mean():.3f}  TopK={sel['topk'].mean():.3f}")
    print(f"  paired diff (L1-TopK), letter-clustered bootstrap: {d_sel_letter.mean():+.4f} "
          f"CI [{lo:+.4f},{hi:+.4f}]  ({int((d_sel_letter<0).sum())}/24 letters lower for L1)")
    print("  => L1 main latents are LESS selective, i.e. L1 SPLITS the letter feature more.")
    print("     This is a clean architecture difference and it is NOT the absorption endpoint.")
    print("     (consistent with round-11 L1-splitting and with P3 L1 recall >> TopK.)")

    # ---------------- D. q-dependence ----------------
    print("\n=== D. FREQUENCY (q) DEPENDENCE — toy theory: eps* increases with q ===")
    for a in ARCHES:
        rho, p = spearman(nn[a].mean(axis=0), rate[a].mean(axis=0))
        print(f"  {a:5s}: spearman(n_instances, rate) = {rho:+.3f} (perm p={p:.4f})  "
              f"[weak proxy for q; n is corpus letter frequency, not latent firing rate]")

    print("\n=== READ ===")
    print("  Generated hypotheses for round 13 (NOT results):")
    print("   H1 regime: dead%=46-57% at m=16384 => spare-capacity regime; toy theory")
    print("      puts absorption pressure under capacity SCARCITY, so an arch gap is")
    print("      not predicted here. Test: sweep m downward at fixed L0.")
    print("   H2 metric: ~2/3 of endpoint variance is a re-expression of main-latent")
    print("      selectivity, which feature SPLITTING also moves; and L1 demonstrably")
    print("      splits more than TopK. So the registered endpoint cannot separate")
    print("      'L1 absorbs more' from 'L1 splits more'. Test: a FAMILY-based endpoint")
    print("      (score the letter against its whole split family, not one main latent).")
    print("   NB: H1 and H2 both predict a null for round 12 WITHOUT the toy theory being")
    print("       wrong. They are the two ways the registered test could have been")
    print("       under-powered in principle rather than in sample size.")


if __name__ == "__main__":
    main()
