"""CPU pilot for the pair-identification detector: CALIBRATION ONLY.

8 absorbed-regime SAEs (rho x sigma x 2 seeds, Arm A generative model + trainer
verbatim via import) to measure the pair-level statistics Arm A did not record:
decoder cos(parent, composite), co-firing rates/lift/overlap at theta=0.05,
parent-fires-on-joint and composite-fires-on-host-only rates, child residual
direction recovery. These numbers set the detector thresholds in
notes/prereg-pair-identification.md BEFORE the confirmatory GPU run; this pilot
is disclosed there and its runs are never counted as confirmatory evidence.
"""
import torch, math, csv, time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_bimodality_armA as A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "prereg_pairid", "pilot.csv")
THETA = 0.05
N_EV = 100_000

runs = [dict(rhos=(rho, 0, 0), sigma=sig, seed=s)
        for rho in (0.10, 0.20) for sig in (0.0, 0.1) for s in (0, 1)]

t0 = time.time()
sae, A_bg_s, spec_s, rec = A.train_program(runs, A.M_LAT, "pilot")
vp, c1, c2, c3 = spec_s
rows = []
for i, run in enumerate(runs):
    vp1, vc1 = vp[i], c1[i]
    w_true = (vp1 + vc1) / math.sqrt(2)
    D = sae.D.detach()[i]
    cos_w = (D * w_true.unsqueeze(1)).sum(0)
    comp = int(cos_w.abs().argmax()); cos_comp = float(cos_w.abs().max())
    cos_p = (D * vp1.unsqueeze(1)).sum(0)
    par = int(cos_p.abs().argmax()); cos_par = float(cos_p.abs().max())
    cos_pc = float((D[:, par] * D[:, comp]).sum().abs())
    # child residual from the pair alone (what the label-free detector gets)
    u = D[:, comp] - (D[:, comp] * D[:, par]).sum() * D[:, par]
    u = u / u.norm().clamp_min(1e-8)
    child_res_cos = float((u * vc1).sum().abs())
    g = torch.Generator(device=A.dev).manual_seed(7000 + run["seed"])
    x, host, chs = A.sample_labeled(N_EV, g, A_bg_s[i], vp1, [c1[i], c2[i], c3[i]],
                                    run["rhos"], run["sigma"])
    f = A.encode(sae, i, x)
    fp_ = f[:, par] > THETA; fc_ = f[:, comp] > THETA
    h = host.cpu().numpy() > 0.5; c = chs[0].cpu().numpy() > 0.5
    Pp, Pc, Pand = fp_.mean(), fc_.mean(), (fp_ & fc_).mean()
    lift = Pand / (Pp * Pc) if Pp * Pc > 0 else float("nan")
    ovl = Pand / min(Pp, Pc) if min(Pp, Pc) > 0 else float("nan")
    # full pair-matrix context: how unusual is (cos band, low lift) globally?
    fires = f > THETA
    rates = fires.mean(0)
    keep = (rates > 5e-4) & (rates < 0.6)
    Dn = D.cpu().numpy()
    C = np.abs(Dn.T @ Dn)
    Pj = (fires.astype(np.float32).T @ fires.astype(np.float32)) / len(f)
    L = Pj / np.maximum(np.outer(rates, rates), 1e-12)
    iu = np.triu_indices(D.shape[1], 1)
    mask = keep[iu[0]] & keep[iu[1]]
    band = (C[iu] > 0.55) & (C[iu] < 0.95) & mask
    rows.append(dict(rho=run["rhos"][0], sigma=run["sigma"], seed=run["seed"],
        absorbed=int(cos_comp > 0.98), cos_comp=round(cos_comp, 4),
        cos_par=round(cos_par, 4), cos_pc=round(cos_pc, 4),
        child_res_cos=round(child_res_cos, 4),
        P_par=round(float(Pp), 5), P_comp=round(float(Pc), 5),
        P_and=round(float(Pand), 6), lift=round(float(lift), 4),
        overlap=round(float(ovl), 4),
        par_fire_joint=round(float(fp_[h & c].mean()), 4),
        par_fire_hostonly=round(float(fp_[h & ~c].mean()), 4),
        comp_fire_joint=round(float(fc_[h & c].mean()), 4),
        comp_fire_hostonly=round(float(fc_[h & ~c].mean()), 4),
        n_pairs_in_band=int(band.sum()),
        pair_in_band=int((C[par, comp] > 0.55) & (C[par, comp] < 0.95)),
        pair_lift_rank=int((L[iu][band] <= L[par, comp]).sum()) if band.sum() else -1,
        rec=round(float(rec[i]), 4)))
    print(f"[pilot {i+1}/8] rho={run['rhos'][0]} sig={run['sigma']} s={run['seed']} "
          f"abs={rows[-1]['absorbed']} cos_pc={cos_pc:.3f} lift={lift:.3f} "
          f"ovl={ovl:.3f} band_pairs={int(band.sum())}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"wrote {OUT} total {time.time()-t0:.0f}s", flush=True)
