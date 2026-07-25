"""Adaptive L1-lambda calibration (round 13b, Amendment 1).

Same job as calibrate_lambda.py -- pick the lambda whose HELD-OUT L0 at the
TRAINING step budget is closest to TARGET -- but with an adaptive bracketing
search instead of a fixed grid.

Why (declared pre-results in notes/prereg-round13b-capacity.md, Amendment 1):
  * The registered fixed grid {2,3,4,4.5,5,6} may not BRACKET L0=32 at small m.
    Measured: at m=2048, lam=4 gives L0 far above 32. A grid that does not
    bracket returns an edge value, the width then fails the matched-L0 gate, and
    the round's most important width is lost.
  * Cost: a fixed 6-point grid is 6 full 15k-step trainings per width.
    Bracket-then-bisect reaches the same target in ~5 and adapts per width.

Outcome-blind by construction: this reads ONLY the reported L0 and never sees any
absorption quantity. Larger lambda => more shrinkage => smaller L0 (monotone),
which is what the bisection assumes; the assumption is CHECKED and reported.

`train_l0` is copied verbatim from calibrate_lambda.py so calibration and the run
still invoke the identical trainer.

Env: ACTS, EVAL_ACTS, EXPANSION, CALIB_STEPS, TARGET (32), LAM0 (4.5),
     MAX_EVALS (6), BAND_LO/BAND_HI (28/36).
Prints "CHOSEN_LAM <lam> L0=<l0> band=<in|widened|none>".
"""
import os, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = os.environ.get("CALIB_STEPS", "15000")
TARGET = float(os.environ.get("TARGET", "32"))
LAM0 = float(os.environ.get("LAM0", "4.5"))
MAX_EVALS = int(os.environ.get("MAX_EVALS", "6"))
LO, HI = float(os.environ.get("BAND_LO", "28")), float(os.environ.get("BAND_HI", "36"))
WLO, WHI = LO - 4, HI + 4


def train_l0(lam):                                   # verbatim from calibrate_lambda.py
    env = dict(os.environ, ARCH="l1", LAM=str(lam), STEPS=STEPS, SEED="0")
    env.setdefault("GPU_ACTS", "1")
    p = subprocess.run([sys.executable, os.path.join(HERE, "real_train_sae.py")],
                       env=env, capture_output=True, text=True)
    l0 = None
    for line in p.stdout.splitlines():
        if line.startswith("STATS "):
            l0 = json.loads(line[6:]).get("l0")
    print(f"  lam={lam}: L0={l0}", flush=True)
    if l0 is None:
        sys.stderr.write(p.stdout[-2000:] + "\n" + p.stderr[-2000:] + "\n")
    return l0


def main():
    assert os.environ.get("ACTS"), "set ACTS=train cache"
    seen = {}                                        # lam -> L0

    def ev(lam):
        lam = round(float(lam), 4)
        if lam not in seen:
            l0 = train_l0(lam)
            if l0 is None:
                sys.exit("calibration trainer failed")
            seen[lam] = l0
        return seen[lam]

    # ---- 1. bracket: find lam_lo with L0 > TARGET and lam_hi with L0 < TARGET ----
    l0 = ev(LAM0)
    lo = hi = None
    if l0 > TARGET:
        lo, l0_lo = LAM0, l0                          # need MORE shrinkage -> larger lam
        lam = LAM0
        while len(seen) < MAX_EVALS:
            lam *= 2.0
            if ev(lam) <= TARGET:
                hi = lam; break
            lo = lam
    else:
        hi, l0_hi = LAM0, l0                          # need LESS shrinkage -> smaller lam
        lam = LAM0
        while len(seen) < MAX_EVALS:
            lam /= 2.0
            if ev(lam) >= TARGET:
                lo = lam; break
            hi = lam

    # ---- 2. bisect on log lambda ----
    if lo is not None and hi is not None:
        while len(seen) < MAX_EVALS:
            mid = (lo * hi) ** 0.5                    # geometric midpoint
            if abs(mid - lo) / lo < 1e-3 or abs(mid - hi) / hi < 1e-3:
                break
            if ev(mid) > TARGET:
                lo = mid
            else:
                hi = mid
    else:
        print("  WARNING: could not bracket the target within MAX_EVALS; "
              "reporting the closest evaluated lambda", flush=True)

    # ---- 3. monotonicity check (the bisection's assumption), reported ----
    xs = sorted(seen)
    ys = [seen[x] for x in xs]
    mono = all(ys[i] >= ys[i + 1] for i in range(len(ys) - 1))
    print(f"  evaluated {dict(zip(xs, ys))}", flush=True)
    print(f"  monotone(L0 decreasing in lambda) = {mono}", flush=True)

    # ---- 4. pick closest to TARGET, preferring in-band ----
    def band(l0):
        return "in" if LO <= l0 <= HI else ("widened" if WLO <= l0 <= WHI else "none")
    best = min(seen, key=lambda L: (band(seen[L]) == "none", abs(seen[L] - TARGET)))
    print(f"CHOSEN_LAM {best} L0={seen[best]} band={band(seen[best])}", flush=True)


if __name__ == "__main__":
    main()
