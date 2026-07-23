"""Round 10: feature absorption under TopK (and JumpReLU) SAEs.

Theory: theory/topk_absorption.md (verified in theory/verify_topk_absorption.py).
Prereg: notes/prereg-topk-absorption.md.

Measures the child-recovery angle phi (0=parent, 45=composite/absorbed,
90=faithful child) for TopK SAEs trained on the 2D hierarchical toy, across
budget k and child-solo rate eps. Batched (R parallel SAEs per training,
grouped by k so the TopK budget is uniform within a batch).

Arms:
  M  (isolated pair; near-zero low-amplitude background): tests T1 (k=1
     crossover on the 2q scale, q-scaling) and T2 (capacity collapse: k=2
     recovers the faithful child for every eps>0 where k=1 absorbs).
  C  (realistic background, sweep global k at fixed eps): T3 (capacity gating).
  J  (JumpReLU threshold gate, exploratory, descriptive only).

SMOKE=1 runs a tiny CPU end-to-end version. Weights saved per (arm,k).
Outputs: results/round10/r10_runs.csv (+ weights_r10_*.pt).
"""
import torch, math, csv, time, os

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = bool(int(os.environ.get("SMOKE", "0")))
OUTDIR = os.path.join(HERE, "..", "results", "round10")
os.makedirs(OUTDIR, exist_ok=True)
BATCH = 2048
STEPS = 200 if SMOKE else 15000

def features(d, n_bg, seed):
    g = torch.Generator().manual_seed(100 + seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return (Q[:, :n_bg].contiguous().to(dev),
            Q[:, n_bg].contiguous().to(dev),        # v_p
            Q[:, n_bg + 1].contiguous().to(dev))    # v_c

class TopKSAE(torch.nn.Module):
    """Batched R parallel SAEs; hard TopK or JumpReLU gate, nonneg (ReLU) codes."""
    def __init__(self, R, d, m):
        super().__init__()
        bnd = 1.0 / math.sqrt(d)
        self.W = torch.nn.Parameter(torch.empty(R, m, d).uniform_(-bnd, bnd))
        self.b = torch.nn.Parameter(torch.zeros(R, m))
        self.D = torch.nn.Parameter(torch.randn(R, d, m) / math.sqrt(d))
        self.renorm()
    def renorm(self):
        with torch.no_grad():
            self.D.div_(self.D.norm(dim=1, keepdim=True).clamp_min(1e-8))
    def forward(self, x, topk=None, jump=None):
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, self.W) + self.b.unsqueeze(1))
        if topk is not None:
            thr = f.topk(topk, dim=-1).values[..., -1:].clamp_min(1e-12)
            f = f * (f >= thr).float()
        if jump is not None:                     # jump: [R,1,1] threshold gate
            f = f * (f >= jump).float()
        return torch.einsum('rbm,rdm->rbd', f, self.D), f

def sampler(runs, Abg, vp, vc):
    """Per-run event partition (joint q, parent-solo p, child-solo eps) + bg."""
    R = len(runs)
    qv = torch.tensor([r["q"] for r in runs], device=dev).view(R, 1)
    pv = torch.tensor([r["p"] for r in runs], device=dev).view(R, 1)
    ev = torch.tensor([r["eps"] for r in runs], device=dev).view(R, 1)
    amp = torch.tensor([r["bg_amp"] for r in runs], device=dev).view(R, 1, 1)
    rate = torch.tensor([r["bg_rate"] for r in runs], device=dev).view(R, 1, 1)
    n_bg = Abg.shape[2]
    def fn(B, gen):
        if n_bg > 0:
            coeff = torch.empty(R, B, n_bg, device=dev).uniform_(0.5, 1.0, generator=gen) * amp
            mask = (torch.rand(R, B, n_bg, device=dev, generator=gen) < rate).float()
            x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
        else:
            x = torch.zeros(R, B, Abg.shape[1], device=dev)
        u = torch.rand(R, B, device=dev, generator=gen)
        joint = (u < qv).float()
        psolo = ((u >= qv) & (u < qv + pv)).float()
        csolo = ((u >= qv + pv) & (u < qv + pv + ev)).float()
        x = (x + (joint + psolo).unsqueeze(-1) * vp.unsqueeze(1)
               + (joint + csolo).unsqueeze(-1) * vc.unsqueeze(1))
        return x
    return fn

def train(sae, fn, steps, topk=None, jump=None, lam=0.0, tag=""):
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(steps):
        x = fn(BATCH, gen)
        xh, f = sae(x, topk=topk, jump=jump)
        rec = ((x - xh) ** 2).sum(-1).mean(-1)
        loss = (rec + lam * f.sum(-1).mean(-1)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == steps // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 4000 == 0:
            print(f"  [{tag} {t}/{steps}] rec={float(rec.mean()):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return rec.detach()

def phi_child(Di, vp, vc):
    """Child-side decoder angle in the (vp,vc) plane. 0=parent,45=composite,90=child."""
    cp = torch.einsum('dm,d->m', Di, vp)
    cc = torch.einsum('dm,d->m', Di, vc)
    rho = torch.sqrt(cp ** 2 + cc ** 2)
    phi = torch.rad2deg(torch.atan2(cc, cp))
    inplane = [(float(rho[j]), float(phi[j])) for j in range(Di.shape[1]) if rho[j] > 0.5]
    child = [(r, a) for r, a in inplane if 20.0 < a < 120.0]
    phic = max(child, key=lambda ra: ra[1])[1] if child else float("nan")
    has_par = any(abs(a) < 20.0 and r > 0.8 for r, a in inplane)
    # composite cosine: best |cos| to (vp+vc)/sqrt2
    w = (vp + vc) / math.sqrt(2)
    cos_comp = float(torch.einsum('dm,d->m', Di, w).abs().max())
    cos_child = float(cc.abs().max())
    return phic, int(has_par), cos_comp, cos_child

# ------------------------------------------------------------------ run lists
def arm_M_runs():
    epss = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    seeds = range(2 if SMOKE else 24)
    runs = []
    for k in (1, 2):
        for q in (0.1, 0.2):
            for e in epss:
                for s in seeds:
                    runs.append(dict(arm="M", k=k, q=q, p=q, eps=e, seed=s,
                                     d=32, n_bg=6, m=16, bg_rate=0.05, bg_amp=0.5,
                                     jump=None, lam=0.0))
    return runs

def arm_C_runs():
    ks = [1, 2, 4] if SMOKE else [1, 2, 3, 4, 6, 8, 12, 16, 24]
    seeds = range(2 if SMOKE else 24)
    runs = []
    for k in ks:
        for s in seeds:
            runs.append(dict(arm="C", k=k, q=0.2, p=0.2, eps=0.10, seed=s,
                             d=64, n_bg=30, m=32, bg_rate=0.08, bg_amp=1.0,
                             jump=None, lam=0.0))
    return runs

def arm_J_runs():                       # exploratory JumpReLU
    seeds = range(2 if SMOKE else 8)
    runs = []
    for th in ([0.3] if SMOKE else [0.1, 0.3, 0.5, 0.7]):
        for e in [0.05, 0.15]:
            for s in seeds:
                runs.append(dict(arm="J", k=None, q=0.2, p=0.2, eps=e, seed=s,
                                 d=32, n_bg=6, m=16, bg_rate=0.05, bg_amp=0.5,
                                 jump=th, lam=0.0))
    return runs

def run_group(all_runs, rows):
    """Train one batched SAE per group of runs sharing (arm,k,jump,d,n_bg,m)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_runs:
        groups[(r["arm"], r["k"], r["jump"], r["d"], r["n_bg"], r["m"])].append(r)
    for gkey, runs in groups.items():
        arm, k, jump, d, n_bg, m = gkey
        feats = [features(d, n_bg, r["seed"]) for r in runs]
        Abg = torch.stack([f[0] for f in feats]) if n_bg > 0 else \
              torch.zeros(len(runs), d, 0, device=dev)
        vp = torch.stack([f[1] for f in feats])
        vc = torch.stack([f[2] for f in feats])
        torch.manual_seed(11)
        sae = TopKSAE(len(runs), d, m).to(dev)
        jvec = None if jump is None else torch.full((len(runs), 1, 1), float(jump), device=dev)
        rec = train(sae, sampler(runs, Abg, vp, vc), STEPS, topk=k, jump=jvec,
                    lam=runs[0]["lam"], tag=f"{arm} k={k} j={jump}")
        Dl = sae.D.detach()
        wtag = f"{arm}_k{k}" + ("" if jump is None else f"_j{int(jump*10):02d}")
        torch.save(dict(D=Dl.cpu(), W=sae.W.detach().cpu(), b=sae.b.detach().cpu(),
                        runs=runs), os.path.join(OUTDIR, f"weights_r10_{wtag}.pt"))
        for i, r in enumerate(runs):
            phic, hp, ccomp, cchild = phi_child(Dl[i], vp[i], vc[i])
            row = dict(r); row.pop("bg_amp", None); row.pop("bg_rate", None)
            row.update(phi_child=round(phic, 3) if phic == phic else float("nan"),
                       has_parent=hp, cos_comp=round(ccomp, 4), cos_child=round(cchild, 4),
                       rec=round(float(rec[i]), 4),
                       eps_star_topk=round(2 * r["q"], 3))
            rows.append(row)
        del sae
        if dev == "cuda": torch.cuda.empty_cache()
        print(f"[{arm} k={k} jump={jump}] {len(runs)} runs done", flush=True)
        write_csv(rows)

def write_csv(rows):
    if not rows: return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    out = os.path.join(OUTDIR, "r10_runs.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    t0 = time.time()
    rows = []
    run_group(arm_M_runs(), rows)
    run_group(arm_C_runs(), rows)
    run_group(arm_J_runs(), rows)
    write_csv(rows)
    print(f"wrote r10_runs.csv ({len(rows)} rows) total {time.time()-t0:.0f}s", flush=True)
