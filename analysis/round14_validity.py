"""Round 14 POST-HOC VALIDITY DIAGNOSTICS (exploratory; NOT a registered endpoint).

Round 14's registered results stand exactly as reported in results_round14.txt and
SUMMARY_round14.md. Nothing here changes them. This script asks a separate question:
do P2 and P4, as computed by the frozen scorer, actually license the reading
"absorbed trials are diffuse"?

An adversarial review (reviews/GEMINI_round14_2026-07-26.md) raised two objections.
One of them is quantitatively wrong; one is right and I did not see it myself. Both
are tested here, plus a third that I found while checking the first.

D1 -- does the carrier argmax ever select a latent that does not fire?
     The reviewer's mechanism: c = F*align is 0 for inactive latents and negative for
     active ones with negative alignment, so if EVERY active latent has negative
     alignment, argmax returns the first inactive index. Real mechanism, but it needs
     all ~L0 active latents to be negatively aligned at once. Measured directly as the
     fraction of trials whose selected carrier has activation 0.

D2 -- is P4's "diffuse" reading sound? THE IMPORTANT ONE.
     The frozen scorer computes conc_A as the share held by the GLOBAL modal carrier
     kappa, averaged over all A trials. If the letter's mass is picked up by a
     different composite latent on each trial (compositional absorption), each trial
     is individually CONCENTRATED while kappa's average share is still tiny -- kappa
     only wins ~14% of trials, so its mean share is small by construction. So the
     frozen conc_A cannot distinguish "diffuse" from "concentrated but trial-specific".
     D2 recomputes concentration PER TRIAL, using each trial's own top non-family
     latent, which is the statistic the diffuseness claim actually needs.

D3 -- is the P2 null a fair comparator?
     For the letter direction the family (the most letter-selective latents) is
     EXCLUDED before the argmax. For a random direction NOTHING is excluded, so the
     null keeps its most concentrated candidates while the letter direction has had
     its removed by construction. That asymmetry inflates the null. D3 recomputes the
     null while also dropping the top-|fam| latents by alignment with r.

Env: same as round14_carrier.py (WORDS, SAES, OUT).
CPU-only.
"""
import os, json
import numpy as np

import round14_carrier as R14           # frozen scorer; sets A/C come from it verbatim

NULL_DRAWS = R14.NULL_DRAWS
THETA, TAU, FAM_CAP, MIN_A = R14.THETA, R14.TAU, R14.FAM_CAP, R14.MIN_A


def _rand_dir(rng, d):
    r = rng.normal(size=d)
    return r / max(np.linalg.norm(r), 1e-12)


def diagnostics(F, Wdec, fam, rows_A, rows_C, u, rng):
    align = Wdec.T @ u
    mask = np.ones(len(align), bool); mask[list(fam)] = False
    idx = np.where(mask)[0]

    FA = F[np.ix_(rows_A, idx)]                     # (|A|, non-family)
    cA = FA * align[idx]
    bestA = cA.argmax(axis=1)
    ar = np.arange(len(rows_A))

    # ---- D1: does the selected carrier actually fire? -----------------------
    d1_letter = float((FA[ar, bestA] <= THETA).mean())
    d1_null = []
    for _ in range(NULL_DRAWS):
        a = Wdec.T @ _rand_dir(rng, len(u))
        b = (FA * a[idx]).argmax(axis=1)
        d1_null.append(float((FA[ar, b] <= THETA).mean()))

    # ---- D2: per-trial concentration, not global-modal concentration --------
    posA = np.clip(cA, 0, None)
    totA = posA.sum(axis=1)
    live = totA > 1e-9                              # trials with any positive mass
    pertrial_A = float((posA.max(axis=1)[live] / totA[live]).mean()) if live.any() else float("nan")
    # how many DISTINCT latents serve as per-trial top carrier (compositional => many)
    n_distinct = int(len(np.unique(idx[bestA])))
    # comparator: same per-trial statistic on control trials, all latents allowed
    cC = F[rows_C] * align
    posC = np.clip(cC, 0, None)
    totC = posC.sum(axis=1)
    liveC = totC > 1e-9
    pertrial_C = float((posC.max(axis=1)[liveC] / totC[liveC]).mean()) if liveC.any() else float("nan")

    # ---- D3: null with a matched exclusion ---------------------------------
    nfam = len(fam)
    fair, unfair = [], []
    for _ in range(NULL_DRAWS):
        a = Wdec.T @ _rand_dir(rng, len(u))
        # unfair (frozen scorer's version): exclude the LETTER family only
        car = idx[(FA * a[idx]).argmax(axis=1)]
        _, cn = np.unique(car, return_counts=True)
        unfair.append(cn.max() / len(car))
        # fair: also drop this direction's own top-|fam| aligned latents
        drop = np.argsort(a)[::-1][:nfam]
        mk = mask.copy(); mk[drop] = False
        ix2 = np.where(mk)[0]
        car2 = ix2[(F[np.ix_(rows_A, ix2)] * a[ix2]).argmax(axis=1)]
        _, cn2 = np.unique(car2, return_counts=True)
        fair.append(cn2.max() / len(car2))

    return dict(
        d1_inactive_frac_letter=d1_letter,
        d1_inactive_frac_null=float(np.mean(d1_null)),
        d2_pertrial_conc_A=pertrial_A,
        d2_pertrial_conc_C=pertrial_C,
        d2_distinct_carriers=n_distinct,
        d2_n_A=int(len(rows_A)),
        d3_null_unfair=float(np.mean(unfair)),
        d3_null_fair=float(np.mean(fair)),
    )


def score_one(path, Xr, letters, probes, rng):
    """Mirrors R14.score_one's set construction EXACTLY; only the stats differ."""
    s = R14.safe_load(path)
    F, Xhat, arch, k, Wdec = R14.encode(s, Xr)
    fires = F > THETA
    out = {}
    for L, (yL, present_all, lrf) in probes.items():
        retained_all = lrf.predict_proba(Xhat)[:, 1] > 0.5
        sel = fires[yL == 1].mean(0) - fires[yL == 0].mean(0)
        j = int(sel.argmax())
        if sel[j] < TAU:
            continue
        fam = np.where(sel >= TAU)[0]
        if len(fam) > FAM_CAP:
            fam = fam[np.argsort(sel[fam])[::-1][:FAM_CAP]]
        Lw = np.where(yL == 1)[0]
        present = present_all[Lw]; retained = retained_all[Lw]
        famfire = fires[np.ix_(Lw, fam)].any(axis=1)
        miss_fam = present & (~famfire) & retained
        ctrl = present & famfire
        if int(miss_fam.sum()) >= MIN_A and int(ctrl.sum()) > 0:
            u = lrf.coef_[0].astype(np.float64)
            u = u / max(np.linalg.norm(u), 1e-12)
            out[L] = diagnostics(F, Wdec, fam, Lw[miss_fam], Lw[ctrl], u, rng)
    return dict(sae=os.path.basename(path), arch=arch,
                m=int(s["W_enc"].shape[0]), seed=s.get("seed"), per_letter=out)


def main():
    W = R14.safe_load(os.environ["WORDS"])
    Xr = W["acts"].numpy().astype("float32")
    letters = np.array(W["letters"])
    print(f"words={Xr.shape}", flush=True)
    probes = R14.build_probes(Xr, letters)
    rng = np.random.default_rng(R14.SEED)
    rows = []
    for p in os.environ["SAES"].split(","):
        p = p.strip()
        if not p:
            continue
        rows.append(score_one(p, Xr, letters, probes, rng))
        print(f"  diag {os.path.basename(p)} cells={len(rows[-1]['per_letter'])}", flush=True)
    json.dump(rows, open(os.environ.get("OUT", "round14_validity.json"), "w"))
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
