"""Round 9: gating-corrected rho estimators (prereg notes/prereg-gating-corrected-rho.md).

Estimators, per ORACLE-LOCATED criterion-qualified absorbed pair (par, comp),
theta = 0.05 (see prereg for frozen definitions + edge-case rules):
  rho_count (incumbent, descriptive): P(b_lo)/P(b_lo OR b_hi), lo = lower-rate
      latent (targets min(rho,1-rho); scoped rho<=0.5).
  rho_C (oracle-comp count; leak-bias baseline): P(b_comp)/P(either).
  rho_X (exclusive-ratio):    n01 / (n01 + n10)             [01 = comp only]
  rho_D (dominance-partition): 01->J, 10->S, 11->J iff act_comp > act_par
      (ties -> S); rho_D = nJ / (nJ + nS). Valid under S1 (unit decoder
      norms, asserted at eval).
Each estimator is scored at TWO endpoints: O (all tokens, operational) and
M (oracle J-union-S tokens only, mechanism; "m_" columns). Scoring uses
ORACLE orientation (registered); auto (rarity) is descriptive.

Cells (confirmatory, fresh seeds, 24/cell per standing constraint, weights
saved):
  SC: synthetic round-8 E3 family (d=64, n_bg=30, m=32, P_HOST=0.25, lam=0.2),
      rho in {0.1,0.3,0.5,0.7} x sigma in {0,0.05,0.1} x seeds 40-63
      (288 runs, one batched training).
  RC: real GPT-2 layer-6 round-8 E1 protocol (m=128, Q+P0=0.4, eps=0.002,
      20k steps), rho in {0.1,0.3,0.5,0.7} -> (Q,P0) in {(0.04,0.36),
      (0.12,0.28),(0.2,0.2),(0.28,0.12)}, seeds 32-55 per cell (matched
      pairs across cells, disjoint from all prior rounds' seeds); one
      batched training per cell.

Qualification (frozen): par = argmax|cos(D_j, v_p)|, comp = argmax
|cos(D_j, w_true)|; absorbed iff cos_comp > 0.98 AND par != comp (both
harnesses; RC = E1's criterion). Non-formed runs excluded + disclosed;
cell scoreable iff >= 16/24 qualify. No exclusion may be based on F1-F4
diagnostics or estimator output. Oracle labels locate/orient/score only -
never inside estimators. SMOKE=1 runs a tiny end-to-end version on CPU.

Outputs: results/round9/r9_runs.csv, weights_r9_syn.pt, weights_r9_real.pt.
"""
import torch, math, csv, time, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import prereg_bimodality_armA as A
import round8_synthetic as R8
from rho_estimators_lib import (rho_estimators, theta_sensitivity,
                                oracle_diags)

dev = A.dev
SMOKE = A.SMOKE
OUTDIR = os.path.join(HERE, "..", "results", "round9")
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------------- SC part
def run_sc(rows):
    rhos = [0.1, 0.3, 0.5, 0.7]
    sigmas = [0.0, 0.05, 0.1]
    seeds = list(range(40, 42 if SMOKE else 64))
    if SMOKE: rhos, sigmas = [0.3], [0.1]
    runs = [dict(exp="SC", rho=r, sigma=sg, seed=s)
            for r in rhos for sg in sigmas for s in seeds]
    feats = [R8.features(64, 30, r["seed"]) for r in runs]
    Abg = torch.stack([f[0] for f in feats])
    vp = torch.stack([f[1] for f in feats])
    vc = torch.stack([f[2] for f in feats])
    torch.manual_seed(11)
    sae = R8.BSAEK(len(runs), 64, 32).to(dev)
    rec, wall = R8.train(sae, R8.sampler(runs, Abg, vp, vc),
                         torch.full((len(runs),), R8.LAM, device=dev),
                         R8.STEPS, tag="R9-SC")
    torch.save(dict(W=sae.W.detach().cpu(), b=sae.b.detach().cpu(),
                    D=sae.D.detach().cpu(), runs=runs),
               os.path.join(OUTDIR, "weights_r9_syn.pt"))
    for i, r in enumerate(runs):
        D = sae.D.detach()[i]
        vp1, vc1 = vp[i], vc[i]
        w_true = vp1 + vc1; w_true = w_true / w_true.norm().clamp_min(1e-8)
        par = int((D * vp1.unsqueeze(1)).sum(0).abs().argmax())
        comp = int((D * w_true.unsqueeze(1)).sum(0).abs().argmax())
        cos_comp = float((D[:, comp] * w_true).sum().abs())
        row = dict(exp="SC", rho=r["rho"], sigma=r["sigma"], seed=r["seed"],
                   m=32, absorbed=int(cos_comp > 0.98 and par != comp),
                   cos_comp=round(cos_comp, 4), rec=round(float(rec[i]), 4),
                   wall_s=round(wall, 1))
        if row["absorbed"] or os.environ.get("EVAL_ALWAYS"):  # shakeout only
            # labeled eval stream (fresh generator, masks retained)
            g = torch.Generator(device=dev).manual_seed(9500 + int(r["seed"]))
            n_ev = 6000 if SMOKE else 150_000
            coeff = torch.empty(n_ev, 30, device=dev).uniform_(0.8, 1.2, generator=g)
            mask = (torch.rand(n_ev, 30, device=dev, generator=g) < R8.BG_RATE).float()
            x = torch.einsum('bn,dn->bd', coeff * mask, Abg[i])
            host = (torch.rand(n_ev, device=dev, generator=g) < R8.P_HOST).float()
            ch = host * (torch.rand(n_ev, device=dev, generator=g) < r["rho"]).float()
            x = x + host.unsqueeze(-1) * vp1 + ch.unsqueeze(-1) * vc1
            x = x + r["sigma"] * torch.randn(n_ev, 64, device=dev, generator=g)
            f = torch.relu(x @ sae.W.detach()[i].T + sae.b.detach()[i])
            # S1 precondition: unit decoder norms on the scored pair
            for k in (par, comp):
                assert abs(float(D[:, k].norm()) - 1.0) < 1e-3, \
                    f"S1 violated: ||D[{k}]|| != 1"
            ap_np = f[:, par].cpu().numpy(); ac_np = f[:, comp].cpu().numpy()
            hostn = host.cpu().numpy() > 0.5; chn = ch.cpu().numpy() > 0.5
            mJ = hostn & chn; mS = hostn & ~chn; mB = ~hostn
            row.update(rho_estimators(ap_np, ac_np))              # O endpoint
            mJS = mJ | mS                                         # M endpoint
            row.update({f"m_{k}": v for k, v in
                        rho_estimators(ap_np[mJS], ac_np[mJS]).items()})
            row.update(theta_sensitivity(ap_np, ac_np))
            row.update(oracle_diags(ap_np, ac_np, mJ, mS, mB))
            # auto orientation (rarity rule, descriptive): comp := rarer latent
            if row["lo_is_comp"]:
                row["rho_d_auto"] = row["rho_d"]
            else:
                sw = rho_estimators(ac_np, ap_np)
                row["rho_d_auto"] = sw["rho_d"]
        rows.append(row)
    del sae
    if dev == "cuda": torch.cuda.empty_cache()
    print("[SC] done", flush=True)

# ------------------------------------------------------------------- RC part
def run_rc(rows):
    D_MODEL, BATCH = 768, 2048
    STEPS = 100 if SMOKE else 20000
    N_EV = 6000 if SMOKE else 200_000
    LAM, AMP, EPS, M_LAT = 1.0, 5.0, 0.002, 128
    CELLS = [(0.04, 0.36), (0.12, 0.28), (0.20, 0.20), (0.28, 0.12)]
    if SMOKE: CELLS = [(0.20, 0.20)]
    SEEDS = list(range(32, 34 if SMOKE else 56))
    ACT = os.path.join(HERE, "activations_l6.pt")
    if not os.path.exists(ACT):
        try:
            import transformers  # noqa
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "--user",
                            "-q", "transformers"], check=True)
        os.chdir(HERE)                    # extractor writes to CWD
        import extract_activations
        extract_activations.main()
    bg = torch.load(ACT, weights_only=True).float()
    mu = bg.mean(0, keepdim=True)
    bg = ((bg - mu) * math.sqrt(D_MODEL) / (bg - mu).norm(dim=1).mean()).half().to(dev)
    N = bg.shape[0]
    pairs = {}
    for s in SEEDS:
        g = torch.Generator().manual_seed(300 + s)
        Qm, _ = torch.linalg.qr(torch.randn(D_MODEL, 2, generator=g))
        pairs[s] = Qm.T.to(dev)
    for qq, pp0 in CELLS:
        run_rc_cell(rows, qq, pp0, SEEDS, pairs, bg, N,
                    D_MODEL, BATCH, STEPS, N_EV, LAM, AMP, EPS, M_LAT)
        write_csv(rows)                    # preemption-safe per-cell flush

def run_rc_cell(rows, qq, pp0, SEEDS, pairs, bg, N,
                D_MODEL, BATCH, STEPS, N_EV, LAM, AMP, EPS, M_LAT):
    runs = [(qq, pp0, s) for s in SEEDS]
    Rn = len(runs)
    ap = torch.stack([pairs[s][0] for _, _, s in runs])
    ac = torch.stack([pairs[s][1] for _, _, s in runs])
    bound = 1 / math.sqrt(D_MODEL)
    torch.manual_seed(51)
    W = torch.nn.Parameter(torch.empty(Rn, M_LAT, D_MODEL, device=dev).uniform_(-bound, bound))
    b = torch.nn.Parameter(torch.zeros(Rn, M_LAT, device=dev))
    Dd = torch.nn.Parameter(torch.randn(Rn, D_MODEL, M_LAT, device=dev) / math.sqrt(D_MODEL))
    with torch.no_grad(): Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
    opt = torch.optim.Adam([W, b, Dd], lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(STEPS):
        idx = torch.randint(0, N, (Rn, BATCH), device=dev, generator=gen)
        x = bg[idx].float()
        u = torch.rand(Rn, BATCH, device=dev, generator=gen)
        j = (u < qq).float()
        p = ((u >= qq) & (u < qq + pp0)).float()
        cs = ((u >= qq + pp0) & (u < qq + pp0 + EPS)).float()
        x = (x + AMP * (j + p).unsqueeze(-1) * ap.unsqueeze(1)
               + AMP * (j + cs).unsqueeze(-1) * ac.unsqueeze(1))
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, W) + b.unsqueeze(1))
        xh = torch.einsum('rbm,rdm->rbd', f, Dd)
        loss = (((x - xh) ** 2).sum(-1) + LAM * f.sum(-1)).mean(-1).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
        if t == STEPS // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 4000 == 0:
            print(f"  [RC q={qq} {t}/{STEPS}] ({time.time()-t0:.0f}s)", flush=True)
    torch.save(dict(W=W.detach().cpu(), b=b.detach().cpu(), D=Dd.detach().cpu(),
                    runs=runs),
               os.path.join(OUTDIR, f"weights_r9_real_q{int(round(qq*100)):02d}.pt"))
    for i, (qq, pp0, s) in enumerate(runs):
        ap_, ac_ = pairs[s][0], pairs[s][1]
        comp_dir = (ap_ + ac_) / math.sqrt(2)
        Di = Dd[i].detach()
        par = int(torch.einsum('dm,d->m', Di, ap_).abs().argmax())
        comp = int(torch.einsum('dm,d->m', Di, comp_dir).abs().argmax())
        cos_comp = float(torch.einsum('dm,d->m', Di, comp_dir).abs().max())
        cos_child = float(torch.einsum('dm,d->m', Di, ac_).abs().max())
        rho_true = qq / (qq + pp0)
        row = dict(exp="RC", rho=round(rho_true, 4), sigma=-1.0, seed=s, m=M_LAT,
                   q=qq, p0=pp0,
                   absorbed=int(cos_comp > 0.98 and par != comp),
                   cos_comp=round(cos_comp, 4), cos_child=round(cos_child, 4),
                   rec=0.0, wall_s=round(time.time() - t0, 1))
        if row["absorbed"] or os.environ.get("EVAL_ALWAYS"):  # shakeout only
            g2 = torch.Generator(device=dev).manual_seed(9500 + s)
            apar_chunks, acomp_chunks, labels = [], [], []
            for c0 in range(0, N_EV, 50_000):
                n = min(50_000, N_EV - c0)
                idx = torch.randint(0, N, (n,), device=dev, generator=g2)
                x = bg[idx].float()
                u = torch.rand(n, device=dev, generator=g2)
                jj = (u < qq).float()
                pp = ((u >= qq) & (u < qq + pp0)).float()
                cc = ((u >= qq + pp0) & (u < qq + pp0 + EPS)).float()
                x = (x + AMP * (jj + pp).unsqueeze(-1) * ap_
                       + AMP * (jj + cc).unsqueeze(-1) * ac_)
                f = torch.relu(x @ W[i].detach().T + b[i].detach())
                apar_chunks.append(f[:, par].cpu().numpy())
                acomp_chunks.append(f[:, comp].cpu().numpy())
                labels.append(torch.stack([jj, pp]).cpu().numpy())
            for k in (par, comp):
                assert abs(float(Di[:, k].norm()) - 1.0) < 1e-3, \
                    f"S1 violated: ||D[{k}]|| != 1"
            ap_np = np.concatenate(apar_chunks)
            ac_np = np.concatenate(acomp_chunks)
            jj, pp = np.concatenate(labels, axis=1)
            mJ = jj > 0.5; mS = pp > 0.5; mB = ~(mJ | mS)
            row.update(rho_estimators(ap_np, ac_np))              # O endpoint
            mJS = mJ | mS                                         # M endpoint
            row.update({f"m_{k}": v for k, v in
                        rho_estimators(ap_np[mJS], ac_np[mJS]).items()})
            row.update(theta_sensitivity(ap_np, ac_np))
            row.update(oracle_diags(ap_np, ac_np, mJ, mS, mB))
            if row["lo_is_comp"]:
                row["rho_d_auto"] = row["rho_d"]
            else:
                sw = rho_estimators(ac_np, ap_np)
                row["rho_d_auto"] = sw["rho_d"]
        rows.append(row)
        print(f"[RC q={qq} s{s}] abs={row['absorbed']} "
              f"rho_d={row.get('rho_d')} rho_x={row.get('rho_x')} "
              f"rho_count={row.get('rho_count')}", flush=True)
    del W, b, Dd
    if dev == "cuda": torch.cuda.empty_cache()

def write_csv(rows):
    """Incremental full rewrite after each phase - preemption-safe on spot
    instances (partial results survive)."""
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    out = os.path.join(OUTDIR, "r9_runs.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)", flush=True)

if __name__ == "__main__":
    t0 = time.time()
    rows = []
    run_sc(rows)
    write_csv(rows)
    ACT_PRESENT = os.path.exists(os.path.join(HERE, "activations_l6.pt"))
    if SMOKE and dev != "cuda" and not ACT_PRESENT:
        print("SMOKE on CPU without activations_l6.pt: skipping RC part "
              "(extractor needs CUDA)", flush=True)
    else:
        run_rc(rows)
    write_csv(rows)
    print(f"total {time.time()-t0:.0f}s", flush=True)
