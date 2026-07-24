"""Train a wide L1 or TopK SAE on cached real-model activations.

Real-model scaling (round 11 infra). Standard SAE recipe: center by a learned
pre-decoder bias, unit-norm decoder columns, ReLU codes; L1 (shrinkage) or
TopK (hard budget) sparsity. Anthropic-style dead-latent resampling.

Env: ACTS (path to acts_*.pt from real_extract.py), ARCH (l1|topk, default topk),
     EXPANSION (default 8 -> m = 8*d), LAM (L1 coeff, default 5.0 on normalized
     acts), K (TopK budget, default 32), STEPS (default 30000; SMOKE tiny),
     BATCH (default 4096), LR (default 4e-4), OUT (default results/real/sae_...).
Saves weights + stats (FVU, L0, dead%). Reports enough to judge SAE quality.
"""
import os, math, time, json
import torch

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results", "real")
os.makedirs(OUTDIR, exist_ok=True)
ACTS = os.environ.get("ACTS", "")
ARCH = os.environ.get("ARCH", "topk")
EXPANSION = int(os.environ.get("EXPANSION", "8"))
LAM = float(os.environ.get("LAM", "5.0"))
K = int(os.environ.get("K", "32"))
STEPS = 300 if SMOKE else int(os.environ.get("STEPS", "30000"))
BATCH = int(os.environ.get("BATCH", "4096"))
LR = float(os.environ.get("LR", "4e-4"))
SEED = int(os.environ.get("SEED", "0"))
GPU_ACTS = bool(int(os.environ.get("GPU_ACTS", "0")))   # hold acts on GPU -> fast
RESAMPLE_EVERY = 5000
dev = "cuda" if torch.cuda.is_available() else "cpu"

def safe_load(p):
    try: return torch.load(p, weights_only=True)
    except Exception: return torch.load(p)

class SAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.W_enc = torch.nn.Parameter(torch.empty(m, d).uniform_(-1, 1) / math.sqrt(d))
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        W = torch.randn(d, m); W /= W.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(W)
        self.renorm()
    def renorm(self):
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))
    def encode(self, x, arch, k):
        f = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        if arch == "topk":
            thr = f.topk(k, dim=-1).values[:, -1:].clamp_min(1e-12)
            f = f * (f >= thr).float()
        return f
    def forward(self, x, arch, k):
        f = self.encode(x, arch, k)
        return f @ self.W_dec.T + self.b_dec, f

def main():
    assert ACTS, "set ACTS=path to acts_*.pt"
    blob = safe_load(ACTS)
    Xh = blob["acts"]                             # [N, d] fp16 on CPU (keep fp16!)
    N, d = Xh.shape
    m = EXPANSION * d
    # normalization stats from a subsample (avoid a full float32 copy -> OOM)
    gsub = torch.Generator().manual_seed(2)
    sub = Xh[torch.randperm(N, generator=gsub)[:min(200000, N)]].float()
    mu = sub.mean(0)
    scale = math.sqrt(d) / (sub - mu).norm(dim=1).mean().item()
    var = (((sub - mu) * scale) ** 2).sum(1).mean().item()
    del sub
    mu_dev = mu.to(dev)
    Xg = Xh.to(dev) if (GPU_ACTS and dev == "cuda") else None   # acts resident on GPU (fast)
    def batch_norm(rows):                          # fp16 rows -> normalized on GPU
        return (rows.float().to(dev) - mu_dev) * scale
    def get_batch(gen):
        if Xg is not None:
            idx = torch.randint(0, N, (BATCH,), generator=gen, device=dev)
            return (Xg[idx].float() - mu_dev) * scale
        return batch_norm(Xh[torch.randint(0, N, (BATCH,), generator=gen)])
    print(f"acts {tuple(Xh.shape)} fp16 model={blob.get('model')} layer={blob.get('layer')} "
          f"-> m={m} arch={ARCH} (lam={LAM} k={K}) seed={SEED} gpu_acts={Xg is not None}", flush=True)
    torch.manual_seed(SEED)
    sae = SAE(d, m).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    g = (torch.Generator(device=dev) if Xg is not None else torch.Generator()).manual_seed(1000 + SEED)
    fired = torch.zeros(m, device=dev)            # steps-since-fired tracker
    t0 = time.time()
    for t in range(STEPS):
        x = get_batch(g)
        xh, f = sae(x, ARCH, K)
        rec = ((x - xh) ** 2).sum(-1).mean()
        if ARCH == "l1":
            loss = rec + LAM * f.abs().sum(-1).mean()
        else:
            loss = rec
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        with torch.no_grad():
            active = (f > 0).any(0)
            fired = torch.where(active, torch.zeros_like(fired), fired + 1)
        if t == STEPS // 2:
            for gp in opt.param_groups: gp["lr"] = LR / 3
        # dead-latent resample (Anthropic-style, simplified)
        if t > 0 and t % RESAMPLE_EVERY == 0 and t < STEPS - RESAMPLE_EVERY:
            dead = torch.where(fired > RESAMPLE_EVERY // 2)[0]
            if len(dead) > 0:
                with torch.no_grad():
                    xb = get_batch(g)
                    resid = (xb - (sae(xb, ARCH, K)[0])).norm(dim=1)
                    pick = torch.multinomial(resid ** 2 + 1e-6, min(len(dead), xb.shape[0]))
                    dirs = (xb[pick] - sae.b_dec)
                    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    nd = min(len(dead), len(pick))
                    sae.W_dec[:, dead[:nd]] = dirs[:nd].T
                    sae.W_enc[dead[:nd]] = dirs[:nd] * 0.2
                    sae.b_enc[dead[:nd]] = 0.0
                    fired[dead[:nd]] = 0
                sae.renorm()
        if t % 4000 == 0:
            with torch.no_grad():
                fvu = (rec / var).item(); l0 = float((f > 0).float().sum(-1).mean())
                dead_pct = float((fired > RESAMPLE_EVERY // 2).float().mean())
            print(f"  [{ARCH} {t}/{STEPS}] FVU={fvu:.4f} L0={l0:.1f} dead={dead_pct:.2%} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # final eval: HELD-OUT if EVAL_ACTS given (a doc-separated cache), else the
    # honest in-cache Monte Carlo FVU. var recomputed on the eval source.
    EVAL_ACTS = os.environ.get("EVAL_ACTS", "")
    with torch.no_grad():
        if EVAL_ACTS:
            Ev = safe_load(EVAL_ACTS)["acts"]
            ei = torch.randperm(Ev.shape[0])[:min(50000, Ev.shape[0])]
            ev = batch_norm(Ev[ei]) if Xg is None else (Ev[ei].float().to(dev) - mu_dev) * scale
            eval_src = "held-out:" + os.path.basename(EVAL_ACTS)
        else:
            ei = torch.randint(0, N, (min(50000, N),), generator=torch.Generator().manual_seed(7))
            ev = batch_norm(Xh[ei])
            eval_src = "in-cache (NOT held-out)"
        var_ev = (ev ** 2).sum(-1).mean().item()
        xh, f = sae(ev, ARCH, K)
        fvu = float(((ev - xh) ** 2).sum(-1).mean() / var_ev)
        l0 = float((f > 0).float().sum(-1).mean())
        dead_pct = float((fired > STEPS // 4).float().mean())
    tag = f"{blob.get('model','m').split('/')[-1]}_L{blob.get('layer')}_{ARCH}_x{EXPANSION}_s{SEED}"
    out = os.path.join(OUTDIR, f"sae_{tag}.pt")
    torch.save(dict(W_enc=sae.W_enc.detach().cpu(), b_enc=sae.b_enc.detach().cpu(),
                    W_dec=sae.W_dec.detach().cpu(), b_dec=sae.b_dec.detach().cpu(),
                    mu=mu, scale=scale, d=d, m=m, arch=ARCH, lam=LAM, k=K, seed=SEED,
                    model=blob.get("model"), layer=blob.get("layer"),
                    stats=dict(fvu=fvu, l0=l0, dead_pct=dead_pct, eval_src=eval_src)), out)
    print(f"saved {out}: FVU={fvu:.4f} ({eval_src}) L0={l0:.1f} dead={dead_pct:.2%} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print("STATS " + json.dumps(dict(tag=tag, seed=SEED, fvu=round(fvu,4), l0=round(l0,1),
                                     dead_pct=round(dead_pct,4))), flush=True)

if __name__ == "__main__":
    main()
