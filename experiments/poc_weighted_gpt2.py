"""POC (round 6): does event-weighting beat absorption on REAL activations?

Setup: GPT-2-small layer-6 residual activations (500k tokens) as background,
centered and scaled to E||x||^2 = d (d=768). A known hierarchical pair
(a_p, a_c: random orthonormal directions, amplitude A=5) is injected per token:
joint q=0.2, parent-solo p0=0.2, child-solo eps (swept). SAE: m=1536 (2x
overcomplete), ReLU encoder, unit-norm decoder, lam=1.0 (soft-threshold-to-
amplitude ratio matched to the toy: lam/2A = 0.1), 20k steps, batch 2048.

PRE-REGISTERED PREDICTIONS:
  P1 (theory transfers): vanilla absorbs at small eps (no clean child latent;
      composite-aligned latent present), recovers at eps=0.05 positive control.
  P2 (oracle weighting works on real data): inverse-density weighting recovers
      the child at all eps incl. 0.002, as in the toy (round 4).
  P3 (practical estimator): residual-weighting (w = clamp(res^2/mean res^2,
      1, 50) after 4k-step warmup -- oracle-free) recovers the child at small
      eps in a majority of seeds: hard-sample upweighting approximates
      inverse-event-density weighting exactly where absorption bites (child-
      solo events carry the largest residual under an absorbed dictionary).

Conditions: {vanilla, oracle, residual} x eps {0.002, 0.01, 0.05} x 4 seeds
= 36 runs, one batched program. Output: results_poc.csv, done_poc.flag.
"""
import torch, math, csv, time, json, os

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
SMOKE = os.environ.get("SMOKE") == "1"

D_MODEL, M_LAT, BATCH = 768, 1536, 2048
STEPS = 60 if SMOKE else 20000
WARMUP = 10 if SMOKE else 4000
LAM, Q, P0, AMP = 1.0, 0.2, 0.2, 5.0

def main():
    bg = torch.load("activations_l6.pt").float()
    bg = bg - bg.mean(0, keepdim=True)
    bg = bg * math.sqrt(D_MODEL) / bg.norm(dim=1).mean()
    bg = bg.half().to(dev)
    N = bg.shape[0]
    print(f"background pool {bg.shape}, E||x||~{bg.float().norm(dim=1).mean():.1f}",
          flush=True)

    conds = ["vanilla", "oracle", "residual"]
    epss = (0.002, 0.01, 0.05)
    n_seeds = 2 if SMOKE else 4
    if SMOKE: conds, epss = ["vanilla", "residual"], (0.002,)
    runs = [(c, e, s) for c in conds for e in epss for s in range(n_seeds)]
    Rn = len(runs)
    # per-seed injected pair directions (orthonormal, shared across conds/eps)
    pairs = {}
    for s in range(n_seeds):
        g = torch.Generator().manual_seed(300 + s)
        Qm, _ = torch.linalg.qr(torch.randn(D_MODEL, 2, generator=g))
        pairs[s] = Qm.T.to(dev)          # [2, d]: a_p, a_c
    ap = torch.stack([pairs[s][0] for _, _, s in runs])   # [R, d]
    ac = torch.stack([pairs[s][1] for _, _, s in runs])
    eps_vec = torch.tensor([e for _, e, _ in runs], device=dev)
    is_oracle = torch.tensor([c == "oracle" for c, _, _ in runs], device=dev)
    is_resid = torch.tensor([c == "residual" for c, _, _ in runs], device=dev)

    bound = 1 / math.sqrt(D_MODEL)
    W = torch.nn.Parameter(torch.empty(Rn, M_LAT, D_MODEL, device=dev).uniform_(-bound, bound))
    b = torch.nn.Parameter(torch.zeros(Rn, M_LAT, device=dev))
    Dd = torch.nn.Parameter(torch.randn(Rn, D_MODEL, M_LAT, device=dev) / math.sqrt(D_MODEL))
    with torch.no_grad():
        Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
    opt = torch.optim.Adam([W, b, Dd], lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(STEPS):
        idx = torch.randint(0, N, (Rn, BATCH), device=dev, generator=gen)
        x = bg[idx].float()                                   # [R,B,d]
        u = torch.rand(Rn, BATCH, device=dev, generator=gen)
        e = eps_vec.view(-1, 1)
        joint = (u < Q).float()
        psolo = ((u >= Q) & (u < Q + P0)).float()
        csolo = ((u >= Q + P0) & (u < Q + P0 + e)).float()
        x = (x + AMP * (joint + psolo).unsqueeze(-1) * ap.unsqueeze(1)
               + AMP * (joint + csolo).unsqueeze(-1) * ac.unsqueeze(1))
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, W) + b.unsqueeze(1))
        xh = torch.einsum('rbm,rdm->rbd', f, Dd)
        res2 = ((x - xh) ** 2).sum(-1)                        # [R,B]
        per_sample = res2 + LAM * f.sum(-1)
        # weights
        w = torch.ones_like(per_sample)
        p_none = (1 - Q - P0 - e).clamp_min(1e-6)
        none = 1 - joint - psolo - csolo
        w_or = (joint / Q + psolo / P0 + csolo / e.clamp_min(1e-6) + none / p_none) / 4
        w = torch.where(is_oracle.view(-1, 1).expand_as(w), w_or, w)
        if t >= WARMUP:
            w_res = (res2 / res2.mean(dim=1, keepdim=True)).clamp(1.0, 50.0).detach()
            w = torch.where(is_resid.view(-1, 1).expand_as(w), w_res, w)
        loss = (w * per_sample).mean(-1).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
        if t == STEPS // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 2000 == 0:
            print(f"  [POC step {t}/{STEPS}] mean_res={float(res2.mean()):.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # metrics
    Dm = Dd.detach()
    comp = (ap + ac) / math.sqrt(2)
    cos_c = torch.einsum('rdm,rd->rm', Dm, ac)
    cos_p = torch.einsum('rdm,rd->rm', Dm, ap)
    cos_x = torch.einsum('rdm,rd->rm', Dm, comp)
    # functional child latent: argmax encoder response on child-solo probe
    probe = AMP * ac
    fch = torch.relu(torch.einsum('rmd,rd->rm', W.detach(), probe) + b.detach())
    act, fidx = fch.max(dim=1)
    rows = []
    for i, (c, e, s) in enumerate(runs):
        j = fidx[i]
        rows.append(dict(
            cond=c, eps=e, seed=s,
            max_cos_child=round(float(cos_c[i].max()), 3),
            max_cos_parent=round(float(cos_p[i].max()), 3),
            max_cos_comp=round(float(cos_x[i].max()), 3),
            func_cos_child=round(float(cos_c[i][j]), 3),
            func_cos_comp=round(float(cos_x[i][j]), 3),
            func_act=round(float(act[i]), 2)))
        print(rows[-1], flush=True)
    with open("results_poc.csv", "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"[POC] {Rn} runs in {time.time()-t0:.0f}s", flush=True)
    if not SMOKE:
        open("done_poc.flag", "w").write("ok\n")

if __name__ == "__main__":
    main()
