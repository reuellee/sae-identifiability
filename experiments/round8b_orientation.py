"""S2 of notes/prereg-containment-orientation.md, Amendment 1: confirmatory
GPU stress test for the amended pair-orientation rule (magnitude
cross-delta primary, firing-containment fallback, rarity final fallback)
vs the round-8 rarity rule alone. Reuses the round8_synthetic E3 harness
(m=32, d=64, n_bg=30, sigma=0.1, 15k steps) at FOUR prevalence cells,
24 seeds each (round 8 used 8). All rules computed head-to-head on the
same fires/activation arrays per run. Weights ARE saved this time
(results/round8/weights_r8b_orient.pt) -- round 8's E2/E3 weight loss is
the gap this closes.

SMOKE=1 tiny end-to-end. Outputs results/round8/r8b_orient_runs.csv + weights.
"""
import torch, math, csv, time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import round8_synthetic as R8S

A, dev, SMOKE = R8S.A, R8S.dev, R8S.SMOKE
OUTDIR = R8S.OUTDIR
MARGIN, MARGIN_MAG, MIN_N = 0.02, 0.3, 30
CELLS = [0.10, 0.50, 0.60, 0.80]
N_SEEDS = 2 if SMOKE else 24


def orient(fires, act, i, j):
    """Amendment 1: magnitude cross-delta primary, firing-containment
    fallback, rarity final fallback. Returns (rarity, containment, amended)."""
    fi, fj = fires[:, i], fires[:, j]
    ai, aj = act[:, i], act[:, j]
    ni, nj = fi.sum(), fj.sum()
    cij = float((fi & fj).sum() / nj) if nj > 0 else float("nan")   # P(i | j)
    cji = float((fi & fj).sum() / ni) if ni > 0 else float("nan")   # P(j | i)
    if cji > cij + MARGIN:
        cont = i
    elif cij > cji + MARGIN:
        cont = j
    else:
        cont = None
    rarity = i if ni < nj else j

    i_off, i_on = ai[fi & ~fj], ai[fi & fj]
    j_off, j_on = aj[fj & ~fi], aj[fj & fi]
    if min(i_off.size, i_on.size, j_off.size, j_on.size) >= MIN_N:
        delta_i = float(i_off.mean() - i_on.mean())
        delta_j = float(j_off.mean() - j_on.mean())
        if delta_i > delta_j + MARGIN_MAG:
            amended = i
        elif delta_j > delta_i + MARGIN_MAG:
            amended = j
        else:
            amended = cont
    else:
        amended = cont
    if amended is None:
        amended = rarity
    return rarity, cont, amended, cij, cji


def eval_run(sae_D, sae_W, sae_b, run, Abg1, vp1, vc1):
    d = Abg1.shape[0]
    w_true = vp1 + vc1
    w_true = w_true / w_true.norm().clamp_min(1e-8)
    D = sae_D
    par = int((D * vp1.unsqueeze(1)).sum(0).abs().argmax())
    comp = int((D * w_true.unsqueeze(1)).sum(0).abs().argmax())
    cos_comp = float((D[:, comp] * w_true).sum().abs())
    g = torch.Generator(device=dev).manual_seed(9000 + run["seed"])
    one = R8S.sampler([run], Abg1.unsqueeze(0), vp1.unsqueeze(0), vc1.unsqueeze(0))
    x = one(R8S.N_EV, g)[0]
    f = torch.relu(x @ sae_W.T + sae_b)
    fires_t = f > R8S.THETA
    fires = fires_t.cpu().numpy()
    act = f.cpu().numpy()
    flags, C, L, rates = R8S.detect_v11(D.cpu().numpy(), fires)
    tp = tuple(sorted((par, comp)))
    tp_flagged = int(tp in flags and par != comp)
    row = dict(rho=run["rho"], seed=run["seed"],
               absorbed=int(cos_comp > 0.98), cos_comp=round(cos_comp, 4),
               tp_flagged=tp_flagged)
    if tp_flagged:
        rarity_ch, cont_ch, amended_ch, cij, cji = orient(fires, act, *tp)
        row["rarity_ok"] = int(rarity_ch == comp)
        row["cont_indet"] = int(cont_ch is None)
        row["cont_ok"] = float("nan") if cont_ch is None else int(cont_ch == comp)
        row["amended_ok"] = int(amended_ch == comp)
        row["c_par_giv_comp"] = round(cij if comp == tp[0] else cji, 4)
        row["c_comp_giv_par"] = round(cji if comp == tp[0] else cij, 4)
    return row


if __name__ == "__main__":
    t0 = time.time()
    runs = [dict(exp="E3orient", rho=rho, sigma=0.1, seed=s)
            for rho in CELLS for s in range(N_SEEDS)]
    feats = [R8S.features(64, 30, r["seed"]) for r in runs]
    Abg = torch.stack([f[0] for f in feats])
    vp = torch.stack([f[1] for f in feats])
    vc = torch.stack([f[2] for f in feats])
    torch.manual_seed(11)
    sae = R8S.BSAEK(len(runs), 64, 32).to(dev)
    rec, wall = R8S.train(sae, R8S.sampler(runs, Abg, vp, vc),
                          torch.full((len(runs),), R8S.LAM, device=dev),
                          R8S.STEPS, tag="E3orient")
    torch.save(dict(W=sae.W.detach().cpu(), b=sae.b.detach().cpu(),
                    D=sae.D.detach().cpu(), Abg=Abg.cpu(), vp=vp.cpu(),
                    vc=vc.cpu(), runs=runs),
               os.path.join(OUTDIR, "weights_r8b_orient.pt"))
    rows = []
    for i, r in enumerate(runs):
        row = eval_run(sae.D.detach()[i], sae.W.detach()[i], sae.b.detach()[i],
                       r, Abg[i], vp[i], vc[i])
        row["wall_s"] = round(wall, 1)
        row["rec"] = round(float(rec[i]), 4)
        rows.append(row)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    out = os.path.join(OUTDIR, "r8b_orient_runs.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows) total {time.time()-t0:.0f}s", flush=True)

    for rho in CELLS:
        c = [r for r in rows if r["rho"] == rho]
        ab = [r for r in c if r["absorbed"]]
        fl = [r for r in ab if r["tp_flagged"]]
        if not fl:
            print(f"rho={rho}: formed {len(ab)}/{len(c)}, flagged 0 -- no orientation data")
            continue
        rar = np.mean([r["rarity_ok"] for r in fl])
        det = [r for r in fl if not r["cont_indet"]]
        cont = np.mean([r["cont_ok"] for r in det]) if det else float("nan")
        amd = np.mean([r["amended_ok"] for r in fl])
        print(f"rho={rho}: formed {len(ab)}/{len(c)}, flagged {len(fl)}/{len(ab)} | "
              f"rarity acc {rar:.3f} | containment acc {cont:.3f} "
              f"(indeterminate {len(fl)-len(det)}/{len(fl)}) | amended acc {amd:.3f}")
