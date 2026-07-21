"""Capacity-limited real-data test: does harmful absorption appear when capacity
is scarce, and does event-weighting then fix it? (The decisive method question
identified after the disambiguation round.)

Same injected-pair setup on real GPT-2 layer-6 activations, but m = 128
(vs 1536 before): latent slots are now scarce relative to the background's
effective feature load, so the pair must COMPETE for representation -- the
regime of Theorem 2 and of production SAEs on full LLMs.

PRE-REGISTERED PREDICTIONS:
  P1 (harmful absorption appears): vanilla at eps=0.002, small m -> absorbed
      configuration: composite-aligned latent present, child-aligned latent
      weak or absent (child max-cos well below the m=1536 value ~0.82 and
      below composite), routing low.
  P2 (weighting fixes it where it bites): residual (and oracle) weighting at
      the same capacity recovers the child latent: higher child max-cos and
      routing than vanilla at eps=0.002.
  P3 (positive control): vanilla at eps=0.05 keeps a child latent even at
      small m (frequent-enough events earn a slot).
Failure of P1 = real-data injected pairs do not reproduce capacity-limited
absorption at these settings (honest negative; regime still untested).

Grid: m = 128 x {vanilla, oracle, residual} x eps {0.002, 0.05} x 4
seeds = 48 runs (two batched groups). Metrics: decoder max-cos to child/comp/
parent + routing score P(best-child latent fires | realistic child-solo probe).
Output: SUMMARY4.md, results_bgrel_v2.csv, bgrel_done.flag.
"""
import torch, math, csv, time

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
D_MODEL, BATCH, STEPS, WARMUP = 768, 2048, 20000, 4000
LAM, Q, P0, AMP = 1.0, 0.2, 0.2, 5.0
out = ["# Capacity-limited test SUMMARY3\n"]
def log(s): print(s, flush=True); out.append(s)

bg = torch.load("activations_l6.pt").float()
mu = bg.mean(0, keepdim=True)
bg = ((bg - mu) * math.sqrt(D_MODEL) / (bg - mu).norm(dim=1).mean()).half().to(dev)
N = bg.shape[0]

# --- stage-1 background SAE (no injected events), for bg-relative novelty weights
BG_M = 128
_bound = 1 / math.sqrt(D_MODEL)
torch.manual_seed(99)
bW = torch.empty(BG_M, D_MODEL, device=dev).uniform_(-_bound, _bound).requires_grad_()
bb = torch.zeros(BG_M, device=dev, requires_grad=True)
bD = (torch.randn(D_MODEL, BG_M, device=dev) / math.sqrt(D_MODEL)).requires_grad_()
with torch.no_grad(): bD.div_(bD.norm(dim=0, keepdim=True))
bopt = torch.optim.Adam([bW, bb, bD], lr=1e-3)
bgen = torch.Generator(device=dev).manual_seed(3)
for t in range(10000):
    idx = torch.randint(0, N, (2048,), device=dev, generator=bgen)
    xb = bg[idx].float()
    fb = torch.relu(xb @ bW.T + bb)
    xhb = fb @ bD.T
    bl = ((xb - xhb) ** 2).sum(-1).mean() + LAM * fb.sum(-1).mean()
    bopt.zero_grad(); bl.backward(); bopt.step()
    with torch.no_grad(): bD.div_(bD.norm(dim=0, keepdim=True).clamp_min(1e-8))
    if t == 5000:
        for gp in bopt.param_groups: gp["lr"] = 1e-3 / 3
bW, bb, bD = bW.detach(), bb.detach(), bD.detach()
# median background residual on clean pool
with torch.no_grad():
    _idx = torch.randint(0, N, (16384,), device=dev, generator=bgen)
    _x = bg[_idx].float()
    _r = ((_x - torch.relu(_x @ bW.T + bb) @ bD.T) ** 2).sum(-1)
    BG_MED = float(_r.median())
print(f"[bg-sae] trained; median clean residual {BG_MED:.1f}", flush=True)

pairs = {}
for s in range(4):
    g = torch.Generator().manual_seed(300 + s)
    Qm, _ = torch.linalg.qr(torch.randn(D_MODEL, 2, generator=g))
    pairs[s] = Qm.T.to(dev)

rows_csv = []
for M_LAT in (128,):
    runs = [(c, e, s) for c in ("vanilla", "oracle", "bgrel")
            for e in (0.002, 0.05) for s in range(4)]
    Rn = len(runs)
    ap = torch.stack([pairs[s][0] for _, _, s in runs])
    ac = torch.stack([pairs[s][1] for _, _, s in runs])
    eps_vec = torch.tensor([e for _, e, _ in runs], device=dev)
    is_or = torch.tensor([c == "oracle" for c, _, _ in runs], device=dev)
    is_re = torch.tensor([c == "bgrel" for c, _, _ in runs], device=dev)
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
        e = eps_vec.view(-1, 1)
        j = (u < Q).float(); p = ((u >= Q) & (u < Q + P0)).float()
        cs = ((u >= Q + P0) & (u < Q + P0 + e)).float()
        x = (x + AMP * (j + p).unsqueeze(-1) * ap.unsqueeze(1)
               + AMP * (j + cs).unsqueeze(-1) * ac.unsqueeze(1))
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, W) + b.unsqueeze(1))
        xh = torch.einsum('rbm,rdm->rbd', f, Dd)
        res2 = ((x - xh) ** 2).sum(-1)
        per = res2 + LAM * f.sum(-1)
        w = torch.ones_like(per)
        pn = (1 - Q - P0 - e).clamp_min(1e-6)
        nn = 1 - j - p - cs
        wor = (j / Q + p / P0 + cs / e.clamp_min(1e-6) + nn / pn) / 4
        w = torch.where(is_or.view(-1, 1).expand_as(w), wor, w)
        with torch.no_grad():
            xf = x.reshape(-1, D_MODEL)
            br = ((xf - torch.relu(xf @ bW.T + bb) @ bD.T) ** 2).sum(-1).reshape(x.shape[0], -1)
            wbg = (br / BG_MED).clamp(1.0, 50.0)
        w = torch.where(is_re.view(-1, 1).expand_as(w), wbg, w)
        loss = (w * per).mean(-1).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
        if t == STEPS // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 4000 == 0: print(f"  [m={M_LAT} {t}/{STEPS}] ({time.time()-t0:.0f}s)", flush=True)
    gen2 = torch.Generator(device=dev).manual_seed(7)
    pidx = torch.randint(0, N, (256,), device=dev, generator=gen2)
    log(f"\n## m = {M_LAT}")
    for i, (c, e, s) in enumerate(runs):
        ap_, ac_ = pairs[s][0], pairs[s][1]
        comp = (ap_ + ac_) / math.sqrt(2)
        cc = torch.einsum('dm,d->m', Dd[i].detach(), ac_)
        best = int(cc.argmax())
        xp = bg[pidx].float() + AMP * ac_
        f = torch.relu(xp @ W[i].detach().T + b[i].detach())
        route = float((f[:, best] > 0).float().mean())
        row = dict(m=M_LAT, cond=c, eps=e, seed=s,
                   cos_child=round(float(cc.max()), 3),
                   cos_comp=round(float(torch.einsum('dm,d->m', Dd[i].detach(), comp).max()), 3),
                   cos_parent=round(float(torch.einsum('dm,d->m', Dd[i].detach(), ap_).max()), 3),
                   route=round(route, 2))
        rows_csv.append(row)
        log(f"- {c} eps={e} s{s}: child={row['cos_child']} comp={row['cos_comp']} "
            f"parent={row['cos_parent']} route={row['route']}")
    del W, b, Dd, opt
    torch.cuda.empty_cache()

with open("results_bgrel_v2.csv", "w", newline="") as fh:
    wcsv = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
    wcsv.writeheader(); wcsv.writerows(rows_csv)
open("SUMMARY4.md", "w").write("\n".join(out))
open("bgrel_done.flag", "w").write("ok\n")
print("done", flush=True)
