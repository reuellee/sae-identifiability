"""Round 8 E2 (width-scaling null calibration) + E3 (robustness cells).
Pre-registered in notes/prereg-round8-scaling-robustness.md. Detector v1.1
UNCHANGED (synthetic thresholds incl. overlap veto).

E2: scales (d,n_bg,m) in (64,30,32),(128,62,64),(256,126,128),(512,254,256);
    null (bg only, 16 seeds) + absorbed (rho=0.10, sigma=0.1, 8 seeds).
E3: m=32 cells, 8 seeds, sigma=0.1: angle cos(vp,vc') in {0.3,0.5};
    prevalence rho=0.6 (orientation stress); TopK k=4 lam=0 (exploratory).

SMOKE=1 tiny end-to-end. Outputs results/round8/r8syn_runs.csv + weights.
"""
import torch, math, csv, time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_bimodality_armA as A

dev = A.dev
SMOKE = A.SMOKE
RT2 = math.sqrt(2)

# detector v1.1 (locked; identical to the Arm-2 constants + overlap veto)
C_LO, C_HI = 0.45, 0.90
L_LO, L_HI = 0.5, 2.0
OVL_MAX = 0.9
THETA = 0.05
RATE_LO, RATE_HI = 5e-4, 0.6

P_HOST, LAM, BG_RATE = 0.25, 0.2, 0.08
STEPS = 100 if SMOKE else 15000
N_EV = 6000 if SMOKE else 150_000
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "results", "round8")
os.makedirs(OUTDIR, exist_ok=True)

SCALES = [(64, 30, 32), (128, 62, 64)] if SMOKE else \
         [(64, 30, 32), (128, 62, 64), (256, 126, 128), (512, 254, 256)]
N_NULL = 2 if SMOKE else 16
N_ABS = 2 if SMOKE else 8

def features(d, n_bg, seed):
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(100 + seed)))
    return Q[:, :n_bg].contiguous().to(dev), Q[:, n_bg].contiguous().to(dev), \
           Q[:, n_bg + 1].contiguous().to(dev)

class BSAEK(A.BSAE):
    """BSAE with optional per-forward TopK (round-2 style)."""
    def forward(self, x, topk=None):
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, self.W) + self.b.unsqueeze(1))
        if topk is not None:
            thr = f.topk(topk, dim=-1).values[..., -1:].clamp_min(1e-12)
            f = f * (f >= thr).float()
        return torch.einsum('rbm,rdm->rbd', f, self.D), f

def train(sae, fn, lam_vec, steps, topk=None, tag=""):
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(steps):
        x = fn(A.BATCH, gen)
        xh, f = sae(x, topk=topk)
        rec = ((x - xh) ** 2).sum(-1).mean(-1)
        loss = (rec + lam_vec * f.sum(-1).mean(-1)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == steps // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 2000 == 0:
            print(f"  [{tag} {t}/{steps}] rec={float(rec.mean()):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return rec.detach(), time.time() - t0

def detect_v11(Dn, fires):
    rates = fires.mean(0)
    keep = (rates > RATE_LO) & (rates < RATE_HI)
    C = np.abs(Dn.T @ Dn)
    F32 = fires.astype(np.float32)
    Pj = (F32.T @ F32) / len(fires)
    L = Pj / np.maximum(np.outer(rates, rates), 1e-12)
    O = Pj / np.maximum(np.minimum.outer(rates, rates), 1e-12)
    m = Dn.shape[1]
    flags = []
    kidx = np.where(keep)[0]
    for a in range(len(kidx)):
        for b_ in range(a + 1, len(kidx)):
            i, j = kidx[a], kidx[b_]
            if (C_LO < C[i, j] < C_HI
                    and (L[i, j] <= L_LO or L[i, j] >= L_HI)
                    and O[i, j] < OVL_MAX):
                flags.append((int(i), int(j)))
    return flags, C, L, rates

def sampler(runs, Abg, vp, vc):
    rho = torch.tensor([r["rho"] for r in runs], device=dev).unsqueeze(1)
    host_on = torch.tensor([1.0 if r["rho"] > 0 else 0.0 for r in runs],
                           device=dev).unsqueeze(1)
    sig = torch.tensor([r["sigma"] for r in runs], device=dev).view(-1, 1, 1)
    Rn, d, n_bg = Abg.shape[0], Abg.shape[1], Abg.shape[2]
    def fn(Bs, gen):
        coeff = torch.empty(Rn, Bs, n_bg, device=dev).uniform_(0.8, 1.2, generator=gen)
        mask = (torch.rand(Rn, Bs, n_bg, device=dev, generator=gen) < BG_RATE).float()
        x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
        host = (torch.rand(Rn, Bs, device=dev, generator=gen) < P_HOST).float() * host_on
        ch = host * (torch.rand(Rn, Bs, device=dev, generator=gen) < rho).float()
        x = (x + host.unsqueeze(-1) * vp.unsqueeze(1)
               + ch.unsqueeze(-1) * vc.unsqueeze(1))
        return x + sig * torch.randn(Rn, Bs, d, device=dev, generator=gen)
    return fn

def eval_run(sae_D, sae_W, sae_b, run, Abg1, vp1, vc1, extra):
    """Detector eval for one run; returns CSV row dict."""
    d = Abg1.shape[0]
    w_true = vp1 + vc1
    w_true = w_true / w_true.norm().clamp_min(1e-8)
    D = sae_D
    par = int((D * vp1.unsqueeze(1)).sum(0).abs().argmax())
    comp = int((D * w_true.unsqueeze(1)).sum(0).abs().argmax())
    cos_comp = float((D[:, comp] * w_true).sum().abs())
    g = torch.Generator(device=dev).manual_seed(9000 + run["seed"])
    one = sampler([run], Abg1.unsqueeze(0), vp1.unsqueeze(0), vc1.unsqueeze(0))
    x = one(N_EV, g)[0]
    f = torch.relu(x @ sae_W.T + sae_b)
    if run.get("topk"):
        thr = f.topk(run["topk"], dim=-1).values[..., -1:].clamp_min(1e-12)
        f = f * (f >= thr).float()
    fires = (f > THETA).cpu().numpy()
    flags, C, L, rates = detect_v11(D.cpu().numpy(), fires)
    tp = tuple(sorted((par, comp)))
    is_null = run["rho"] == 0
    tp_flagged = 0 if is_null else int(tp in flags and par != comp)
    fp = len(flags) if is_null else len([p for p in flags if p != tp])
    row = dict(exp=run["exp"], d=d, m=D.shape[1], rho=run["rho"],
               sigma=run["sigma"], ccos=run.get("ccos", 0.0),
               topk=run.get("topk", 0), seed=run["seed"],
               absorbed=0 if is_null else int(cos_comp > 0.98),
               cos_comp=round(cos_comp, 4), n_flagged=len(flags),
               tp_flagged=tp_flagged, fp_count=fp,
               tp_cos=round(float(C[tp]), 4) if not is_null else 0.0,
               tp_lift=round(float(L[tp]), 4) if not is_null else 0.0)
    row.update(extra)
    if tp_flagged:
        lo, hi = (tp if rates[tp[0]] < rates[tp[1]] else (tp[1], tp[0]))
        row["orient_ok"] = int(lo == comp)
        u = D[:, lo] - (D[:, lo] * D[:, hi]).sum() * D[:, hi]
        u = u / u.norm().clamp_min(1e-8)
        # registered D3-analogue target: orthogonalized child
        ustar = vc1 - (vc1 * vp1).sum() * vp1
        ustar = ustar / ustar.norm().clamp_min(1e-8)
        row["child_res_cos"] = round(float((u * ustar).sum().abs()), 4)
    return row

if __name__ == "__main__":
    t0 = time.time()
    rows = []
    # ------------------------------------------------------------- E2 scales
    for (d, n_bg, m) in SCALES:
        runs = ([dict(exp="E2null", rho=0.0, sigma=0.1, seed=s) for s in range(N_NULL)]
                + [dict(exp="E2abs", rho=0.10, sigma=0.1, seed=s) for s in range(N_ABS)])
        feats = [features(d, n_bg, r["seed"]) for r in runs]
        Abg = torch.stack([f[0] for f in feats])
        vp = torch.stack([f[1] for f in feats])
        vc = torch.stack([f[2] for f in feats])
        torch.manual_seed(11)
        sae = BSAEK(len(runs), d, m).to(dev)
        rec, wall = train(sae, sampler(runs, Abg, vp, vc),
                          torch.full((len(runs),), LAM, device=dev), STEPS,
                          tag=f"E2 d={d} m={m}")
        mem = (torch.cuda.max_memory_allocated() / 2**20
               if dev == "cuda" else float("nan"))
        for i, r in enumerate(runs):
            rows.append(eval_run(sae.D.detach()[i], sae.W.detach()[i],
                                 sae.b.detach()[i], r, Abg[i], vp[i], vc[i],
                                 dict(wall_s=round(wall, 1), mem_mb=round(mem, 0),
                                      rec=round(float(rec[i]), 4))))
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        print(f"[E2 d={d} m={m}] done", flush=True)
        del sae
        if dev == "cuda": torch.cuda.empty_cache()
    # ------------------------------------------------------------- E3 cells
    cells = ([dict(exp="E3angle", ccos=0.3), dict(exp="E3angle", ccos=0.5),
              dict(exp="E3prev", rho=0.6)] if not SMOKE else
             [dict(exp="E3angle", ccos=0.3)])
    runs, vcs = [], []
    for cell in cells:
        for s in range(N_ABS):
            r = dict(exp=cell["exp"], rho=cell.get("rho", 0.10), sigma=0.1,
                     ccos=cell.get("ccos", 0.0), seed=s)
            runs.append(r)
    feats = [features(64, 30, r["seed"]) for r in runs]
    Abg = torch.stack([f[0] for f in feats])
    vp = torch.stack([f[1] for f in feats])
    vc0 = torch.stack([f[2] for f in feats])
    c = torch.tensor([r["ccos"] for r in runs], device=dev).unsqueeze(1)
    vc = c * vp + (1 - c ** 2).sqrt() * vc0          # nonorthogonal child
    torch.manual_seed(11)
    sae = BSAEK(len(runs), 64, 32).to(dev)
    rec, wall = train(sae, sampler(runs, Abg, vp, vc),
                      torch.full((len(runs),), LAM, device=dev), STEPS, tag="E3")
    for i, r in enumerate(runs):
        rows.append(eval_run(sae.D.detach()[i], sae.W.detach()[i],
                             sae.b.detach()[i], r, Abg[i], vp[i], vc[i],
                             dict(wall_s=round(wall, 1), mem_mb=0.0,
                                  rec=round(float(rec[i]), 4))))
    del sae
    # TopK cell (lam=0, k=4)
    runs_k = [dict(exp="E3topk", rho=0.10, sigma=0.1, topk=4, seed=s)
              for s in range(N_ABS)]
    feats = [features(64, 30, r["seed"]) for r in runs_k]
    Abg = torch.stack([f[0] for f in feats])
    vp = torch.stack([f[1] for f in feats])
    vc = torch.stack([f[2] for f in feats])
    torch.manual_seed(11)
    sae = BSAEK(len(runs_k), 64, 32).to(dev)
    rec, wall = train(sae, sampler(runs_k, Abg, vp, vc),
                      torch.zeros(len(runs_k), device=dev), STEPS, topk=4,
                      tag="E3topk")
    for i, r in enumerate(runs_k):
        rows.append(eval_run(sae.D.detach()[i], sae.W.detach()[i],
                             sae.b.detach()[i], r, Abg[i], vp[i], vc[i],
                             dict(wall_s=round(wall, 1), mem_mb=0.0,
                                  rec=round(float(rec[i]), 4))))
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    out = os.path.join(OUTDIR, "r8syn_runs.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows) total {time.time()-t0:.0f}s", flush=True)
