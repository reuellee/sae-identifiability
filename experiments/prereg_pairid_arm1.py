"""Arm 1 (confirmatory) of the pair-identification pre-registration.

Implements notes/prereg-pair-identification.md: detector = decoder-cosine band
+ low binarized co-firing lift; conditions A (absorbed), CD (correlated
independent — the discrimination control), CDX (exclusive-correlated, the
disclosed equivalence class, exploratory), F (faithful), N (null).
Trainer/generative model = Arm A verbatim (imported). SAE weights are saved.

Detector thresholds below are the ones LOCKED in the prereg note, calibrated on
the disclosed CPU pilot (results/prereg_pairid/pilot.csv). Do not change after
the confirmatory run.

SMOKE=1 runs a tiny end-to-end version.
Outputs: results/prereg_pairid/arm1_runs.csv + weights_<tag>.pt
"""
import torch, math, csv, time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_bimodality_armA as A
from prereg_bimodality_armA import BSAE, train_batched, make_feature_sets

dev = A.dev
SMOKE = A.SMOKE
RT2 = math.sqrt(2)

# ---- detector constants (LOCKED from pilot; keep in sync with the note) ----
# Two-sided lift rule (pilot discovery): absorbed pairs sit at lift ~0 (sigma=0,
# clean gating) or ~3 (sigma=0.1, both gates leak); independent correlated
# features sit at lift ~1. Flag = cos band AND lift far from 1 either way.
C_LO, C_HI = 0.45, 0.90
L_LO, L_HI = 0.5, 2.0
THETA = 0.05
RATE_LO, RATE_HI = 5e-4, 0.6
N_EV = 6000 if SMOKE else 200_000
# ----------------------------------------------------------------------------

P_HOST, LAM = A.P_HOST, A.LAM
EPS_F = 0.0152          # 2.5 * eps*(lam=0.2, q = P_HOST*0.10) — faithful side
SEEDS = 2 if SMOKE else 16
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "results", "prereg_pairid")

def build_runs():
    runs = []
    a_cells = [(0.10, 0.0)] if SMOKE else \
        [(r, s) for r in (0.05, 0.10, 0.20) for s in (0.0, 0.1)]
    for rho, sig in a_cells:
        for s in range(SEEDS):
            runs.append(dict(cond="A", rho=rho, eps=0.0, b_rate=0.0,
                             b_anti=0, sigma=sig, seed=s))
    for br in ((0.03,) if SMOKE else (0.03, 0.10)):
        for s in range(SEEDS):
            runs.append(dict(cond="CD", rho=0.0, eps=0.0, b_rate=br,
                             b_anti=0, sigma=0.0, seed=s))
    for s in range(SEEDS):
        runs.append(dict(cond="CDX", rho=0.0, eps=0.0, b_rate=0.03,
                         b_anti=1, sigma=0.0, seed=s))
    for s in range(SEEDS):
        runs.append(dict(cond="F", rho=0.10, eps=EPS_F, b_rate=0.0,
                         b_anti=0, sigma=0.0, seed=s))
    for s in range(SEEDS):
        runs.append(dict(cond="N", rho=0.0, eps=0.0, b_rate=0.0,
                         b_anti=0, sigma=0.0, seed=s))
    return runs

def sampler(runs, Abg, vp, vc, vb):
    """Vectorized over runs: host/child/child-solo/b-feature per-run rates."""
    rho = torch.tensor([r["rho"] for r in runs], device=dev).unsqueeze(1)
    eps = torch.tensor([r["eps"] for r in runs], device=dev).unsqueeze(1)
    brt = torch.tensor([r["b_rate"] for r in runs], device=dev).unsqueeze(1)
    anti = torch.tensor([float(r["b_anti"]) for r in runs],
                        device=dev).unsqueeze(1)
    sig = torch.tensor([r["sigma"] for r in runs], device=dev).view(-1, 1, 1)
    host_on = torch.tensor([1.0 if r["cond"] != "N" else 0.0 for r in runs],
                           device=dev).unsqueeze(1)
    Rn = len(runs)
    def fn(Bs, gen, return_labels=False):
        coeff = torch.empty(Rn, Bs, A.N_BG, device=dev).uniform_(
            0.8, 1.2, generator=gen)
        mask = (torch.rand(Rn, Bs, A.N_BG, device=dev, generator=gen)
                < A.BG_RATE).float()
        x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
        host = (torch.rand(Rn, Bs, device=dev, generator=gen)
                < P_HOST).float() * host_on
        child = host * (torch.rand(Rn, Bs, device=dev, generator=gen)
                        < rho).float()
        # child-solo (F): independent Bernoulli; overlap w/ host events is
        # negligible (P_HOST*eps) and disclosed in the prereg note
        solo = (torch.rand(Rn, Bs, device=dev, generator=gen) < eps).float()
        child = torch.clamp(child + solo, max=1.0)
        b = (torch.rand(Rn, Bs, device=dev, generator=gen) < brt).float()
        b = b * (1.0 - anti * host)          # CDX: b only when host silent
        x = (x + host.unsqueeze(-1) * vp.unsqueeze(1)
               + child.unsqueeze(-1) * vc.unsqueeze(1)
               + b.unsqueeze(-1) * vb.unsqueeze(1))
        x = x + sig * torch.randn(Rn, Bs, A.D_AMB, device=dev, generator=gen)
        return (x, host, child, b) if return_labels else x
    return fn

def detect(Dn, fires):
    """The registered detector. Dn:[d,m] numpy, fires:[N,m] bool.
    Returns (flag list of (i,j), C, L, rates)."""
    rates = fires.mean(0)
    keep = (rates > RATE_LO) & (rates < RATE_HI)
    C = np.abs(Dn.T @ Dn)
    F32 = fires.astype(np.float32)
    Pj = (F32.T @ F32) / len(fires)
    L = Pj / np.maximum(np.outer(rates, rates), 1e-12)
    m = Dn.shape[1]
    flags = []
    for i in range(m):
        if not keep[i]: continue
        for j in range(i + 1, m):
            if not keep[j]: continue
            if C_LO < C[i, j] < C_HI and (L[i, j] <= L_LO or L[i, j] >= L_HI):
                flags.append((i, j))
    return flags, C, L, rates

if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    runs = build_runs()
    Rn = len(runs)
    A_bg_s, spec_s = make_feature_sets([100 + r["seed"] for r in runs],
                                       n_special=5)
    vp_s, vc_s, _, _, eo_s = spec_s
    vb_s = (vp_s + eo_s) / RT2                     # 45 deg from v_p, unit norm
    torch.manual_seed(11)
    sae = BSAE(Rn, A.D_AMB, A.M_LAT).to(dev)
    fn = sampler(runs, A_bg_s, vp_s, vc_s, vb_s)
    rec = train_batched(sae, fn, torch.full((Rn,), LAM, device=dev),
                        A.STEPS, tag=f"pairid R={Rn}")
    torch.save(dict(W=sae.W.detach().cpu(), b=sae.b.detach().cpu(),
                    D=sae.D.detach().cpu(), runs=runs),
               os.path.join(OUTDIR, "weights_arm1.pt"))

    rows = []
    for i, run in enumerate(runs):
        vp1, vc1, vb1 = vp_s[i], vc_s[i], vb_s[i]
        D = sae.D.detach()[i]
        w_true = (vp1 + vc1) / RT2
        par = int((D * vp1.unsqueeze(1)).sum(0).abs().argmax())
        comp = int((D * w_true.unsqueeze(1)).sum(0).abs().argmax())
        cos_comp = float((D[:, comp] * w_true).sum().abs())
        chl = int((D * vc1.unsqueeze(1)).sum(0).abs().argmax())
        cos_child = float((D[:, chl] * vc1).sum().abs())
        b_idx = int((D * vb1.unsqueeze(1)).sum(0).abs().argmax())
        cos_b = float((D[:, b_idx] * vb1).sum().abs())
        # eval set from the run's own distribution
        g = torch.Generator(device=dev).manual_seed(9000 + run["seed"])
        # single-run labeled sampling via the vectorized fn would waste Rn x N;
        # index into a chunked eval of this run only
        one = sampler([run], A_bg_s[i:i+1], vp_s[i:i+1], vc_s[i:i+1],
                      vb_s[i:i+1])
        x, host, child, bmask = one(N_EV, g, return_labels=True)
        f = A.encode(sae, i, x[0])
        fires = f > THETA
        flags, C, L, rates = detect(D.cpu().numpy(), fires)
        if run["cond"] in ("CD", "CDX"):
            tp = tuple(sorted((par, b_idx)))
        else:
            tp = tuple(sorted((par, comp)))
        tp_flagged = int(tp in flags)
        fp = [p for p in flags if p != tp]
        # pair-level stats + estimator on the DETECTED orientation
        row = dict(cond=run["cond"], rho=run["rho"], eps=run["eps"],
                   b_rate=run["b_rate"], sigma=run["sigma"], seed=run["seed"],
                   absorbed=int(cos_comp > 0.98), cos_comp=round(cos_comp, 4),
                   cos_child=round(cos_child, 4), cos_b=round(cos_b, 4),
                   n_flagged=len(flags), tp_flagged=tp_flagged,
                   fp_count=len(fp),
                   tp_cos=round(float(C[tp]), 4), tp_lift=round(float(L[tp]), 4),
                   rate_i=round(float(rates[tp[0]]), 5),
                   rate_j=round(float(rates[tp[1]]), 5),
                   rec=round(float(rec[i]), 4))
        if tp_flagged:
            lo, hi = (tp if rates[tp[0]] < rates[tp[1]] else (tp[1], tp[0]))
            det_comp, det_par = lo, hi          # composite = rarer
            u = D[:, det_comp] - (D[:, det_comp] * D[:, det_par]).sum() * D[:, det_par]
            u = u / u.norm().clamp_min(1e-8)
            row["orient_ok"] = int(det_comp == (comp if run["cond"] in ("A", "F")
                                                else b_idx))
            row["child_res_cos"] = round(float((u * (vc1 if run["cond"] in ("A", "F")
                                                     else eo_s[i])).sum().abs()), 4)
            either = fires[:, det_comp] | fires[:, det_par]
            row["rho_hat"] = round(float(fires[:, det_comp].mean()
                                         / max(either.mean(), 1e-9)), 5)
        rows.append(row)
        if i % 16 == 15:
            print(f"  [eval {i+1}/{Rn}]", flush=True)

    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    out = os.path.join(OUTDIR, "arm1_runs.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows) total {time.time()-t0:.0f}s",
          flush=True)
