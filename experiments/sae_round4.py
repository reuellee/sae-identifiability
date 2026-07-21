"""Round 4 (batched): GPU validation of the event-weighted remedy (report sec 12).

Sharp pre-registered predictions from theory/remedy_weighted.py:
  P1: an inverse-density-weighted SAE recovers the faithful child latent at eps
      as low as 5e-4 -- 100x below the vanilla boundary eps*=0.0486 and 20x
      below the best any coherence penalty achieves (0.0112) -- because under
      weighting the absorbed configuration is not even a local minimum.
  P2: recovery is RELIABLE (no seed multistability), unlike the coherence
      penalty whose anti/merged traps persist.
  P3: unweighted controls at the same eps stay absorbed (phi ~ 45deg).

Weighting: per-sample w(x) = (1/P(event class of x)) / Z, event classes
{joint, parent-solo, child-solo, background-only} with oracle labels (we
generate the data, so labels are known; estimating them from data is future
work and noted in the report). Z normalizes E[w] = 1 (= number of classes).
Weight multiplies the whole per-sample loss (recon + L1), matching the theory.

SMOKE=1 for a pipeline check. Output: results_weighted.csv, done_r4.flag.
"""
import torch, math, csv, time, json, os

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE") == "1"
print(f"device={dev} smoke={SMOKE}", flush=True)

RT2 = math.sqrt(2)
D_AMB, N_BG, M_LAT, BG_RATE, BATCH = 64, 30, 32, 0.08, 2048
STEPS = 60 if SMOKE else 15000
LAM, Q, P0 = 0.2, 0.2, 0.2

class BSAE(torch.nn.Module):
    def __init__(self, R, d, m):
        super().__init__()
        bound = 1 / math.sqrt(d)
        self.W = torch.nn.Parameter(torch.empty(R, m, d).uniform_(-bound, bound))
        self.b = torch.nn.Parameter(torch.empty(R, m).uniform_(-bound, bound))
        self.D = torch.nn.Parameter(torch.randn(R, d, m) / math.sqrt(d))
        self.renorm()
    def renorm(self):
        with torch.no_grad():
            self.D.div_(self.D.norm(dim=1, keepdim=True).clamp_min(1e-8))

def make_feature_sets(seed_bases, n_special):
    Ms = torch.stack([torch.randn(D_AMB, D_AMB,
                                  generator=torch.Generator().manual_seed(b))
                      for b in seed_bases])
    Qm, _ = torch.linalg.qr(Ms)
    A_bg = Qm[:, :, :N_BG].to(dev)
    spec = [Qm[:, :, N_BG + i].contiguous().to(dev) for i in range(n_special)]
    return A_bg, spec

def sample_with_weights(Bs, gen, Abg, ap, ac, eps_vec, weighted_mask):
    """Returns x [R,B,d] and per-sample weights w [R,B] (1.0 where unweighted)."""
    Rn = Abg.shape[0]
    coeff = torch.empty(Rn, Bs, N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(Rn, Bs, N_BG, device=dev, generator=gen) < BG_RATE).float()
    x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
    u = torch.rand(Rn, Bs, device=dev, generator=gen)
    e = eps_vec.view(-1, 1)
    joint = (u < Q).float()
    psolo = ((u >= Q) & (u < Q + P0)).float()
    csolo = ((u >= Q + P0) & (u < Q + P0 + e)).float()
    x = (x + (joint + psolo).unsqueeze(-1) * ap.unsqueeze(1)
           + (joint + csolo).unsqueeze(-1) * ac.unsqueeze(1))
    # inverse-density weights from oracle event labels, E[w] normalized
    p_none = (1.0 - Q - P0 - e).clamp_min(1e-6)
    none = 1.0 - joint - psolo - csolo
    w = (joint / Q + psolo / P0 + csolo / e.clamp_min(1e-6) + none / p_none) / 4.0
    w = torch.where(weighted_mask.view(-1, 1).expand_as(w), w, torch.ones_like(w))
    return x, w

def inventory(sae, ap, ac):
    Dm = sae.D.detach()
    cp = torch.einsum('rdm,rd->rm', Dm, ap)
    cc = torch.einsum('rdm,rd->rm', Dm, ac)
    rho = (cp ** 2 + cc ** 2).sqrt()
    phi = torch.rad2deg(torch.atan2(cc, cp))
    fch = torch.relu(torch.einsum('rmd,rd->rm', sae.W.detach(), ac) + sae.b.detach())
    act, idx = fch.max(dim=1)
    out = []
    for r in range(Dm.shape[0]):
        order = torch.argsort(rho[r], descending=True)[:6]
        inv = [(round(float(rho[r][i]), 3), round(float(phi[r][i]), 1))
               for i in order if float(rho[r][i]) > 0.3]
        out.append(dict(phi_func=round(float(phi[r][idx[r]]), 1),
                        rho_func=round(float(rho[r][idx[r]]), 3),
                        act_child=round(float(act[r]), 3),
                        inv=json.dumps(inv)))
    return out

def main():
    eps_grid = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    n_seeds = 2 if SMOKE else 8
    if SMOKE: eps_grid = eps_grid[:2]
    seeds = list(range(n_seeds))
    A_bg_s, (ap_s, ac_s) = make_feature_sets([100 + s for s in seeds], 2)
    runs = [(w, e, s) for w in (True, False) for e in eps_grid for s in seeds]
    Rn = len(runs)
    sidx = torch.tensor([s for _, _, s in runs])
    Abg, ap, ac = A_bg_s[sidx], ap_s[sidx], ac_s[sidx]
    eps_vec = torch.tensor([e for _, e, _ in runs], device=dev, dtype=torch.float32)
    wmask = torch.tensor([w for w, _, _ in runs], device=dev)
    torch.manual_seed(17)
    sae = BSAE(Rn, D_AMB, M_LAT).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(STEPS):
        x, w = sample_with_weights(BATCH, gen, Abg, ap, ac, eps_vec, wmask)
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, sae.W) + sae.b.unsqueeze(1))
        xh = torch.einsum('rbm,rdm->rbd', f, sae.D)
        per_sample = ((x - xh) ** 2).sum(-1) + LAM * f.sum(-1)      # [R,B]
        loss = (w * per_sample).mean(-1).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == STEPS // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 3000 == 0:
            print(f"  [R4 step {t}/{STEPS}] ({time.time()-t0:.0f}s)", flush=True)
    inv = inventory(sae, ap, ac)
    rows = [dict(weighted=int(w), eps=e, seed=s, **inv[i])
            for i, (w, e, s) in enumerate(runs)]
    with open("results_weighted.csv", "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"[round4] {Rn} runs done in {time.time()-t0:.0f}s", flush=True)
    if not SMOKE:
        open("done_r4.flag", "w").write("ok\n")

if __name__ == "__main__":
    main()
