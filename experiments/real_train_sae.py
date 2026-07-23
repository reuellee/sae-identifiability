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
    X = blob["acts"].float()                     # [N, d] on CPU
    N, d = X.shape
    m = EXPANSION * d
    mu = X.mean(0)
    scale = math.sqrt(d) / (X - mu).norm(dim=1).mean().item()
    Xn = ((X - mu) * scale)                       # normalized; keep on CPU, stream to GPU
    print(f"acts {tuple(X.shape)} model={blob.get('model')} layer={blob.get('layer')} "
          f"-> m={m} arch={ARCH} (lam={LAM} k={K})", flush=True)
    torch.manual_seed(0)
    sae = SAE(d, m).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)
    fired = torch.zeros(m, device=dev)            # steps-since-fired tracker
    t0 = time.time()
    var = ((X - mu) ** 2).sum(1).mean().item() * (scale ** 2)   # var of normalized acts
    for t in range(STEPS):
        idx = torch.randint(0, N, (BATCH,), generator=g)
        x = Xn[idx].to(dev)
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
                    xb = Xn[torch.randint(0, N, (8192,), generator=g)].to(dev)
                    resid = (xb - (sae(xb, ARCH, K)[0])).norm(dim=1)
                    pick = torch.multinomial(resid ** 2 + 1e-6, min(len(dead), 8192))
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
    # final eval on a held-out random slab
    with torch.no_grad():
        ev = Xn[torch.randint(0, N, (min(50000, N),), generator=g)].to(dev)
        xh, f = sae(ev, ARCH, K)
        fvu = float(((ev - xh) ** 2).sum(-1).mean() / var)
        l0 = float((f > 0).float().sum(-1).mean())
        dead_pct = float((fired > STEPS // 4).float().mean())
    tag = f"{blob.get('model','m').split('/')[-1]}_L{blob.get('layer')}_{ARCH}_x{EXPANSION}"
    out = os.path.join(OUTDIR, f"sae_{tag}.pt")
    torch.save(dict(W_enc=sae.W_enc.detach().cpu(), b_enc=sae.b_enc.detach().cpu(),
                    W_dec=sae.W_dec.detach().cpu(), b_dec=sae.b_dec.detach().cpu(),
                    mu=mu, scale=scale, d=d, m=m, arch=ARCH, lam=LAM, k=K,
                    model=blob.get("model"), layer=blob.get("layer"),
                    stats=dict(fvu=fvu, l0=l0, dead_pct=dead_pct)), out)
    print(f"saved {out}: FVU={fvu:.4f} L0={l0:.1f} dead={dead_pct:.2%} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print("STATS " + json.dumps(dict(tag=tag, fvu=round(fvu,4), l0=round(l0,1),
                                     dead_pct=round(dead_pct,4))), flush=True)

if __name__ == "__main__":
    main()
