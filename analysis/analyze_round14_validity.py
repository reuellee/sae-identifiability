"""Report the round-14 post-hoc validity diagnostics.

Written before the diagnostic's real output was read, for the same reason every
evaluator in this project is: so the presentation cannot be tuned to the result.

Aggregation follows the round-14 prereg's rule, which the review endorsed: collapse
per SAE first, then bootstrap over SAEs. The (SAE, letter) cells are NOT independent
-- 16 SAEs share the same ~24 letters -- so bootstrapping cells directly would give
intervals several times too narrow.

Reads results/real/round14_validity.json. Usage:
    python3 analysis/analyze_round14_validity.py [path] > results_round14_validity.txt
"""
import json
import sys

import numpy as np

BOOT = 10000
SEED = 0
PRIMARY_M = 16384


def boot_ci(per_sae, rng, n=BOOT):
    """95% percentile CI over SAEs (the independent unit)."""
    v = np.asarray([x for x in per_sae if np.isfinite(x)], float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def collapse(rows, m, field):
    """One number per SAE: the mean over that SAE's scored letters."""
    out = []
    for r in rows:
        if r["m"] != m:
            continue
        v = [c[field] for c in r["per_letter"].values()
             if field in c and np.isfinite(c[field])]
        if v:
            out.append(float(np.mean(v)))
    return out


def collapse_arch(rows, m, arch, field):
    return [float(np.mean([c[field] for c in r["per_letter"].values()
                           if field in c and np.isfinite(c[field])]))
            for r in rows
            if r["m"] == m and r["arch"] == arch and r["per_letter"]]


def line(label, vals, rng, fmt="{:.4f}"):
    if not vals:
        print(f"  {label:<38} (no scored cells)")
        return None
    mu = float(np.mean(vals))
    lo, hi = boot_ci(vals, rng)
    print(f"  {label:<38} {fmt.format(mu)}   95% CI [{fmt.format(lo)}, {fmt.format(hi)}]"
          f"   n_SAE={len(vals)}")
    return mu


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/real/round14_validity.json"
    rows = json.load(open(path))
    rng = np.random.default_rng(SEED)
    widths = sorted({r["m"] for r in rows})

    print("ROUND 14 POST-HOC VALIDITY DIAGNOSTICS")
    print("=" * 78)
    print("EXPLORATORY. Not a registered endpoint. The round-14 registered results in")
    print("results_round14.txt are unchanged by anything below.")
    print(f"SAEs: {len(rows)}   widths: {widths}   bootstrap: {BOOT} over SAEs, seed {SEED}")
    ncell = sum(len(r["per_letter"]) for r in rows)
    print(f"scored (SAE, letter) cells: {ncell}")

    for m in widths:
        sub = [r for r in rows if r["m"] == m]
        cells = sum(len(r["per_letter"]) for r in sub)
        print(f"\n{'-' * 78}\nm = {m}   ({len(sub)} SAEs, {cells} scored cells)"
              f"{'   <-- PRIMARY' if m == PRIMARY_M else '   (underpowered in round 14; read with care)'}")

        print("\n D1  does the selected carrier actually fire?")
        print("     (reviewer's blocking claim: argmax falls through to an INACTIVE latent)")
        line("inactive-carrier frac, letter dir", collapse(rows, m, "d1_inactive_frac_letter"), rng)
        line("inactive-carrier frac, random dir", collapse(rows, m, "d1_inactive_frac_null"), rng)

        print("\n D2  per-trial concentration  [THE ONE THAT DECIDES THE CONCLUSION]")
        print("     high on A => each absorbed trial HAS a carrier, just a varying one")
        print("                  (compositional absorption -- 'diffuse' would be wrong)")
        print("     low  on A => mass is spread thinly => loss / threshold suppression")
        a = line("per-trial top share, absorbed (A)", collapse(rows, m, "d2_pertrial_conc_A"), rng)
        c = line("per-trial top share, control (C)", collapse(rows, m, "d2_pertrial_conc_C"), rng)
        if a is not None and c is not None:
            print(f"  {'A / C ratio':<38} {a / c:.3f}"
                  f"   ({'A comparable to normal trials' if a / c > 0.7 else 'A much less concentrated than normal'})")
        nd = collapse(rows, m, "d2_distinct_carriers")
        na = collapse(rows, m, "d2_n_A")
        if nd and na:
            print(f"  {'distinct carriers / |A|':<38} "
                  f"{np.mean(nd):.1f} / {np.mean(na):.1f} = {np.mean(nd) / max(np.mean(na), 1e-9):.3f}"
                  "   (near 1 = every trial its own carrier)")

        print("\n D3  is the P2 null a fair comparator?   [EXPLORATORY -- see RESPONSE caveat;")
        print("     the self-test does NOT exercise this, so treat as indicative only]")
        u = line("null, family excluded only (frozen)", collapse(rows, m, "d3_null_unfair"), rng)
        f = line("null, symmetric exclusion (fair)", collapse(rows, m, "d3_null_fair"), rng)
        if u is not None and f is not None:
            print(f"  {'inflation from the asymmetry':<38} {u - f:+.4f}")

        for arch in ("l1", "topk"):
            va = collapse_arch(rows, m, arch, "d2_pertrial_conc_A")
            if va:
                print(f"     [{arch:>4}] per-trial conc on A = {np.mean(va):.4f} (n_SAE={len(va)})")

    print(f"\n{'=' * 78}\nHow to read D2: the claim under test is 'absorbed trials are diffuse'.")
    print("The frozen P4 used the GLOBAL modal carrier and so could not distinguish")
    print("diffuse from concentrated-but-trial-specific. D2 uses each trial's OWN top")
    print("non-family latent, which is the statistic that claim requires.")


if __name__ == "__main__":
    main()
