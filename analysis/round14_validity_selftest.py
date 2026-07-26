"""Self-test for analysis/round14_validity.py, run BEFORE looking at its real output.

The diagnostics exist to adjudicate a review objection, so they have to be shown to
discriminate before their numbers are used as an argument. This builds two synthetic
worlds that differ only in the thing D2 claims to measure and checks that D2
separates them:

  "carrier"  each absorbed trial has one strongly aligned latent drawn from a small
             recurring pool -- compositional/hierarchical absorption, the state the
             reviewer says the frozen P4 cannot rule out.
  "diffuse"  no aligned latent; the letter direction's mass is spread over whatever
             happens to fire -- representational loss.

Expected: D2 (per-trial concentration on A) HIGH for "carrier", LOW for "diffuse".
If it does not separate them, D2 cannot settle the objection and must not be cited.

It also measures D1 (fraction of trials whose selected carrier does not fire) in both
worlds. The reviewer's blocking claim was that argmax falls through to an inactive
latent whenever every active latent is negatively aligned, which needs all ~L0 of
them negative at once -- about 2^-L0. Measured here rather than argued.

Run: python3 analysis/round14_validity_selftest.py     (numpy + torch, no sklearn)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import round14_validity as V

D, M, N_A, N_C, L0, POOL = 64, 400, 120, 200, 20, 8
SEED = 0


def build(scenario, rng, Wdec, u):
    """Same background process in both worlds; only the aligned latent differs."""
    F = np.zeros((N_A + N_C, M), np.float32)
    fam = list(range(5))
    for t in range(N_A + N_C):
        act = rng.choice(np.arange(5, M), size=L0, replace=False)
        F[t, act] = np.abs(rng.normal(1.0, 0.3, size=L0))
        if t >= N_A:                                    # control: a family latent fires
            F[t, rng.choice(fam)] = 3.0
        elif scenario == "carrier":
            j = 5 + (t % POOL)                          # recurring carrier pool
            Wdec[:, j] = u + 0.05 * rng.normal(size=D)
            Wdec[:, j] /= np.linalg.norm(Wdec[:, j])
            F[t, j] = 8.0
    return F, fam


def main():
    rng = np.random.default_rng(SEED)
    u = rng.normal(size=D); u /= np.linalg.norm(u)
    out = {}
    for scen in ("diffuse", "carrier"):
        Wdec = rng.normal(size=(D, M))
        Wdec /= np.linalg.norm(Wdec, axis=0, keepdims=True)
        F, fam = build(scen, rng, Wdec, u)
        r = V.diagnostics(F, Wdec, fam, np.arange(N_A),
                          np.arange(N_A, N_A + N_C), u, rng)
        out[scen] = r
        print(f"--- {scen} ---")
        print(f"  D1 inactive-carrier frac : letter={r['d1_inactive_frac_letter']:.4f}"
              f"  null={r['d1_inactive_frac_null']:.4f}")
        print(f"  D2 per-trial conc on A   : {r['d2_pertrial_conc_A']:.3f}"
              f"   (control {r['d2_pertrial_conc_C']:.3f})")
        print(f"  D2 distinct carriers     : {r['d2_distinct_carriers']} / {r['d2_n_A']}")
        print(f"  D3 null unfair / fair    : {r['d3_null_unfair']:.3f}"
              f" / {r['d3_null_fair']:.3f}")

    ok = True
    sep = out["carrier"]["d2_pertrial_conc_A"] - out["diffuse"]["d2_pertrial_conc_A"]
    if sep < 0.30:
        print(f"\nFAIL: D2 separates the two worlds by only {sep:.3f}; it cannot "
              f"settle the objection."); ok = False
    if out["carrier"]["d2_distinct_carriers"] > POOL:
        print(f"\nFAIL: carrier world should use <= {POOL} distinct carriers, got "
              f"{out['carrier']['d2_distinct_carriers']}."); ok = False
    for scen in out:
        if out[scen]["d1_inactive_frac_letter"] > 0.01 or out[scen]["d1_inactive_frac_null"] > 0.01:
            print(f"\nNOTE: inactive-carrier selection is non-negligible in "
                  f"'{scen}' -- the reviewer's argmax concern would then be live.")
    print(f"\nD2 separation = {sep:.3f} (carrier {out['carrier']['d2_pertrial_conc_A']:.3f} "
          f"vs diffuse {out['diffuse']['d2_pertrial_conc_A']:.3f}) -- "
          f"{'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
