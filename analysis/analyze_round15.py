"""Round 15 (FROZEN AT LOCK): evaluator for the Gemma Scope 2 cross-validation.

Pre-registration: notes/prereg-round15-gemmascope-crossval.md. Frozen at the same
commit. Reads the per-SAE rows JSON produced by experiments/gemmascope_crossval.py
and emits the registered verdicts. Do not tune anything here after seeing a result.

Env: ROWS=<round15_rows.json>  OUT=<results_round15.txt>
Registered constants: BOOT=10000, boot seed=1, letters as the statistical unit.
"""
import os, json
import numpy as np

BOOT = 10_000
SEED = 1
W_SERIES = [16384, 65536, 262144]          # layer 13, l0_medium
L0_SERIES = ["small", "medium", "big"]     # layer 13, width 65536
LAYER_SERIES = [7, 13, 17, 22]             # width 16384, l0_medium
PRIMARY_LAYER = 13


def boot_ci_letters(diffs_by_letter, reps=BOOT, seed=SEED):
    """Letter-clustered bootstrap: resample letters; a letter contributes ALL its
    entries (a list of diffs) each time it is drawn."""
    letters = sorted(diffs_by_letter)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(letters), size=len(letters))
        vals = [v for i in pick for v in diffs_by_letter[letters[i]]]
        means[r] = float(np.mean(vals))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(lo, hi):
    if lo > 0:
        return "CONFIRMED"
    if hi < 0:
        return "FALSIFIED-DIRECTION"
    return "NOT CONFIRMED"


def spearman3(xs, ys):
    """Tie-aware rank correlation (scipy ships with scikit-learn on the VM;
    Gemini pre-lock review #2: argsort-ranks fake +1.0 on tied rates)."""
    from scipy.stats import spearmanr
    res = spearmanr(xs, ys)
    r = res.statistic if hasattr(res, "statistic") else res[0]
    return float(r) if r == r else float("nan")


def clean_letters(row):
    return {L for L, v in row["per_letter"].items() if v.get("clean_latent")}


EXPECTED_CELLS = {(13, 16384, "medium"), (13, 65536, "medium"), (13, 262144, "medium"),
                  (13, 65536, "small"), (13, 65536, "big"),
                  (7, 16384, "medium"), (17, 16384, "medium"), (22, 16384, "medium")}


def main():
    rows = json.load(open(os.environ["ROWS"]))
    by = {(r["layer"], r["width"], r["l0_tag"]): r for r in rows}
    out = []

    def w(s):
        out.append(s)
        print(s, flush=True)

    # ---------------------------------------------------------------- gates
    gates = {}
    # Frozen-configuration enforcement (GPT pre-lock P1.2): fail closed if any
    # row was produced under non-registered constants or a registered cell is
    # missing. The scorer's env overrides exist for the SMOKE pilot only.
    cfg_ok = all(r.get("theta") == 0 and r.get("tau") == 0.3 and r.get("fam_cap") == 32
                 for r in rows)
    cells_ok = EXPECTED_CELLS <= set(by.keys())
    gates["g0_frozen_config"] = bool(cfg_ok)
    gates["g0_cells_present"] = bool(cells_ok)
    wrows = {}
    for width in W_SERIES:
        r = by.get((PRIMARY_LAYER, width, "medium"))
        wrows[width] = r
        if r is None:
            gates[f"present_{width}"] = False
            continue
        cfg = r.get("config_l0") or 0
        ok_l0 = cfg > 0 and 0.5 * cfg <= r["measured_l0"] <= 1.5 * cfg
        ok_fvu = r["fvu"] <= 0.5
        gates[f"conform_{width}"] = bool(ok_l0 and ok_fvu)
    g1 = all(gates.get(f"conform_{width}", False) for width in W_SERIES)

    cl = {width: clean_letters(wrows[width]) if wrows[width] else set() for width in W_SERIES}
    gates["letters_per_cell"] = {str(width): len(cl[width]) for width in W_SERIES}
    g2_cells = all(len(cl[width]) >= 15 for width in W_SERIES)
    inter12 = cl[16384] & cl[262144]
    gates["inter_16k_262k"] = len(inter12)
    g2_inter = len(inter12) >= 12
    # Gate 3 (words): probes exist only for letters clearing MIN_WORDS on both
    # sides, so the probed-letter count in any row is the operative check.
    probed = len(wrows[16384]["per_letter"]) if wrows[16384] else 0
    gates["g3_probed_letters"] = probed
    g3 = probed >= 20
    gates["g1_conformance"] = g1
    gates["g2_cells"] = g2_cells
    gates["g2_intersection"] = g2_inter
    gates["g3_words"] = g3
    w(f"gates: {json.dumps(gates)}")

    for width in W_SERIES:
        r = wrows[width]
        if r:
            w(f"cell w={width}: L0={r['measured_l0']}/cfg{r['config_l0']} "
              f"fvu={r['fvu']} rate_single={r['rate_single']} "
              f"rate_family={r['rate_family']} clean_letters={len(cl[width])}")

    fail = []
    if not (cfg_ok and cells_ok):
        fail.append("config")
    if not g1:
        fail.append("conformance")
    if not (g2_cells and g2_inter):
        fail.append("letters")
    if not g3:
        fail.append("words")

    # ---------------------------------------------------------------- P1 / P2
    def paired(field):
        d = {}
        for L in sorted(inter12):
            a = wrows[262144]["per_letter"][L][field]
            b = wrows[16384]["per_letter"][L][field]
            d[L] = [float(a) - float(b)]
        return d

    # P1 is the SOLE primary; P2 is the key secondary (GPT pre-lock P2.11 —
    # no unadjusted dual-primary family). P2 tests splitting on the UNCAPPED
    # family size (GPT pre-lock P2.6 — the capped size is censored at 32).
    for name, label, field in (("P1", "PRIMARY", "rate_family"),
                               ("P2", "KEY SECONDARY", "fam_size_uncapped")):
        if fail:
            w(f"{name} ({label}): NOT CONFIRMED (failing gates: {fail})")
            continue
        d = paired(field)
        vals = [v for vs in d.values() for v in vs]
        lo, hi = boot_ci_letters(d)
        v = verdict(lo, hi)
        w(f"{name} ({label}) {field} 262k-16k: {v} "
          f"(mean={np.mean(vals):+.4f}, CI [{lo:+.4f},{hi:+.4f}], "
          f"n_letters={len(d)}, positive {sum(1 for x in vals if x > 0)}/{len(vals)})")
    if not fail:
        hits = sum(1 for width in W_SERIES
                   for v_ in wrows[width]["per_letter"].values() if v_.get("cap_hit"))
        w(f"   fam cap hits across width-series cells: {hits}")

    # per-letter Spearman across the width series (descriptive)
    inter3 = cl[16384] & cl[65536] & cl[262144]
    if all(wrows[width] for width in W_SERIES) and inter3:
        sp = [spearman3(np.array(W_SERIES, float),
                        np.array([wrows[width]["per_letter"][L]["rate_family"]
                                  for width in W_SERIES]))
              for L in sorted(inter3)]
        w(f"D: per-letter Spearman(rate_family, width) mean={np.nanmean(sp):+.3f} "
          f"over {len(sp)} letters (descriptive)")

    # ---------------------------------------------------------------- P3
    # P3 REDESIGNED pre-lock (GPT review P2.4/P2.5, Gemini #4): the sign of
    # rate_single - rate_family is guaranteed (family contains the argmax), so
    # the registered bar is MATERIAL relative inflation: letter-mean relative
    # inflation over the three-width clean intersection, CI lower > 0.10.
    # Letters with mean rate_family below the registered floor are excluded.
    P3_BAR, P3_FLOOR = 0.10, 0.005
    if fail:
        w(f"P3 (secondary): NOT CONFIRMED (failing gates: {fail})")
    else:
        d3, dropped = {}, 0
        for L in sorted(inter3):
            rs = [wrows[width]["per_letter"][L] for width in W_SERIES]
            mf = float(np.mean([x["rate_family"] for x in rs]))
            ms = float(np.mean([x["rate_single"] for x in rs]))
            if mf < P3_FLOOR:
                dropped += 1
                continue
            d3[L] = [ms / mf - 1.0]
        if len(d3) < 8:
            w(f"P3 (secondary): NOT CONFIRMED (only {len(d3)} letters above the "
              f"rate_family floor {P3_FLOOR}; {dropped} dropped)")
        else:
            lo, hi = boot_ci_letters(d3)
            vals = [v for vs in d3.values() for v in vs]
            v = "CONFIRMED" if lo > P3_BAR else "NOT CONFIRMED"
            w(f"P3 (secondary) relative inflation (bar {P3_BAR:.2f}): {v} "
              f"(mean={np.mean(vals):+.3f}, CI [{lo:+.3f},{hi:+.3f}], "
              f"n_letters={len(d3)}, dropped_below_floor={dropped}; 13a found 0.23-0.33)")
        for width in W_SERIES:
            r = wrows[width]
            rf = r["rate_family"]
            infl = (r["rate_single"] - rf) / rf if rf > 0 else float("nan")
            w(f"   inflation w={width}: single={r['rate_single']} family={rf} "
              f"relative={infl:+.1%}")

    # D4 (sensitivity, GPT review P1.1): eligibility fixed by the 16k BASELINE
    # cell alone so the 262k outcome cannot select letters out; letters that
    # lose their clean latent at 262k contribute the tau-waived rate.
    if not fail:
        d4 = {}
        for L in sorted(cl[16384]):
            hipl = wrows[262144]["per_letter"].get(L, {})
            hi_rate = hipl.get("rate_family", hipl.get("rate_family_waived"))
            lo_rate = wrows[16384]["per_letter"][L]["rate_family"]
            if hi_rate is None:
                continue
            d4[L] = [float(hi_rate) - float(lo_rate)]
        lo4, hi4 = boot_ci_letters(d4)
        vals4 = [v for vs in d4.values() for v in vs]
        churn = sum(1 for L in cl[16384] if L not in cl[262144])
        w(f"D4 (sensitivity) P1 with 16k-fixed eligibility (waived tau at 262k): "
          f"mean={np.mean(vals4):+.4f}, CI [{lo4:+.4f},{hi4:+.4f}], "
          f"n_letters={len(d4)}, letters_losing_clean_at_262k={churn}")

    # ---------------------------------------------------------------- D1-D3
    w("D1 (descriptive) L0 series @65k layer13:")
    for tag in L0_SERIES:
        r = by.get((PRIMARY_LAYER, 65536, tag))
        if r:
            w(f"   l0_{tag}: cfgL0={r['config_l0']} measured={r['measured_l0']} "
              f"rate_family={r['rate_family']} rate_single={r['rate_single']} "
              f"mean_fam_size={np.mean([v['fam_size'] for v in r['per_letter'].values() if v.get('clean_latent')]):.2f}")
    w("D2 (descriptive) layer series @16k medium:")
    for ly in LAYER_SERIES:
        r = by.get((ly, 16384, "medium"))
        if r:
            w(f"   layer_{ly}: rate_family={r['rate_family']} "
              f"clean_letters={len(clean_letters(r))} fvu={r['fvu']}")
    w("D3 (descriptive) theta grid (family rate):")
    for r in rows:
        if r.get("grid_family"):
            w(f"   {r['sae']}: {json.dumps(r['grid_family'])}")

    o = os.environ.get("OUT", "results_round15.txt")
    open(o, "w").write("\n".join(out) + "\n")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
