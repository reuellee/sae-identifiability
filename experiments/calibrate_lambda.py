"""Pick the L1 lambda that matches TopK's L0 (round-12 prereg step).

Trains short L1 SAEs (seed 0) at a lambda grid by subprocess-calling the EXACT
trainer (experiments/real_train_sae.py), reads the reported held-out L0 from its
STATS line, and prints the lambda whose L0 lands in the registered band closest
to the target. Reusing the real trainer keeps calibration and the run identical.

Env: ACTS (train cache), EVAL_ACTS (held-out cache), LAM_GRID (comma-sep,
     default "2,4,8,16,32,64"), CALIB_STEPS (default 8000), TARGET (32),
     BAND_LO/BAND_HI (28/36), EXPANSION (8), GPU_ACTS (1).
Prints "CHOSEN_LAM <lam> L0=<l0> band=<in|widened|none>".
"""
import os, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = [float(x) for x in os.environ.get("LAM_GRID", "2,4,8,16,32,64").split(",")]
STEPS = os.environ.get("CALIB_STEPS", "8000")
TARGET = float(os.environ.get("TARGET", "32"))
LO, HI = float(os.environ.get("BAND_LO", "28")), float(os.environ.get("BAND_HI", "36"))
WLO, WHI = LO - 4, HI + 4                                  # widened band

def train_l0(lam):
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
    results = []                                           # (lam, l0)
    for lam in GRID:
        l0 = train_l0(lam)
        if l0 is not None:
            results.append((lam, l0))
    if not results:
        print("CHOSEN_LAM none L0=none band=none (all calibration runs failed)"); return
    inband = [(lam, l0) for lam, l0 in results if LO <= l0 <= HI]
    widened = [(lam, l0) for lam, l0 in results if WLO <= l0 <= WHI]
    pool, tag = (inband, "in") if inband else ((widened, "widened") if widened else (results, "none"))
    lam, l0 = min(pool, key=lambda t: abs(t[1] - TARGET))
    print(f"grid results: {results}", flush=True)
    print(f"CHOSEN_LAM {lam} L0={l0:.2f} band={tag}", flush=True)

if __name__ == "__main__":
    main()
