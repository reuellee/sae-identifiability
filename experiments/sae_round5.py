"""Round 5 (batched): GPU test of the critical-ratio corollary (general_no_go.md).

PRE-REGISTERED PREDICTIONS (from p0*(lam=0.2, q=0.2) = 0.294):
  P1: p0=0.35 (> p0*), beta >= beta*: SGD recovers the FAITHFUL child even at
      eps=0.002 (far below the p0=q no-go floor) -- majority of seeds faithful.
  P2: p0=0.35, beta=0: absorbed (vanilla boundary eps*=0.0486 does not depend
      on p0; both eps tested are below it) -- the rescue in P1 is the penalty's.
  P3: p0=0.20 (< p0*), beta >= beta*: anti/absorbed persists at eps=0.002
      (replicates round 3) -- the contrast with P1 is the corollary's signature.
Multistability caveat pre-stated: SGD basins may retain some anti seeds at
p0=0.35; prediction is majority faithful, vs near-zero in the p0=0.2 control.

Design: p0 in {0.20, 0.35} x beta in {0, 1, 4}*beta* x eps in {0.002, 0.01}
x 8 seeds = 96 runs, one batched program. Output: results_p0.csv, done_r5.flag.
"""
import torch, math, csv, time, json, os

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE") == "1"
print(f"device={dev} smoke={SMOKE}", flush=True)

RT2 = math.sqrt(2)
D_AMB, N_BG, M_LAT, BG_RATE, BATCH = 64, 30, 32, 0.08, 2048
STEPS = 60 if SMOKE else 15000
LAM, Q = 0.2, 0.2
BSTAR = LAM * Q * (8 - 4 * RT2 - LAM) / 2

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
    return (Qm[:, :, :N_BG].to(dev),
            [Qm[:, :, N_BG + i].contiguous().to(dev) for i in range(n_special)])

def sample(Bs, gen, Abg, ap, ac, eps_vec, p0_vec):
    Rn = Abg.shape[0]
    coeff = torch.empty(Rn, Bs, N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(Rn, Bs, N_BG, device=dev, generator=gen) < BG_RATE).float()
    x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
    u = torch.rand(Rn, Bs, device=dev, generator=gen)
    e = eps_vec.view(-1, 1); p0 = p0_vec.view(-1, 1)
    joint = (u < Q).float()
    psolo = ((u >= Q) & (u < Q + p0)).float()
    csolo = ((u >= Q + p0) & (u < Q + p0 + e)).float()
    return (x + (joint + psolo).unsqueeze(-1) * ap.unsqueeze(1)
              + (joint + csolo).unsqueeze(-1) * ac.unsqueeze(1))

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
                        inv=json.dumps(inv)))
    return out

def main():
    p0s = (0.20, 0.35)
    betas = (0.0, 1.0, 4.0)
    epss = (0.002, 0.01)
    n_seeds = 2 if SMOKE else 8
    if SMOKE: p0s, betas, epss = (0.35,), (1.0,), (0.002,)
    seeds = list(range(n_seeds))
    A_bg_s, (ap_s, ac_s) = make_feature_sets([100 + s for s in seeds], 2)
    runs = [(p0, bm, e, s) for p0 in p0s for bm in betas for e in epss for s in seeds]
    Rn = len(runs)
    sidx = torch.tensor([s for _, _, _, s in runs])
    Abg, ap, ac = A_bg_s[sidx], ap_s[sidx], ac_s[sidx]
    eps_vec = torch.tensor([e for _, _, e, _ in runs], device=dev, dtype=torch.float32)
    p0_vec = torch.tensor([p for p, _, _, _ in runs], device=dev, dtype=torch.float32)
    beta_vec = torch.tensor([bm * BSTAR for _, bm, _, _ in runs], device=dev,
                            dtype=torch.float32)
    torch.manual_seed(23)
    sae = BSAE(Rn, D_AMB, M_LAT).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    eye = torch.eye(M_LAT, device=dev)
    t0 = time.time()
    for t in range(STEPS):
        x = sample(BATCH, gen, Abg, ap, ac, eps_vec, p0_vec)
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, sae.W) + sae.b.unsqueeze(1))
        xh = torch.einsum('rbm,rdm->rbd', f, sae.D)
        rec = ((x - xh) ** 2).sum(-1).mean(-1)
        G = torch.einsum('rdi,rdj->rij', sae.D, sae.D) - eye
        loss = (rec + LAM * f.sum(-1).mean(-1)
                + beta_vec * (G ** 2).sum(dim=(1, 2)) / 2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == STEPS // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 3000 == 0:
            print(f"  [R5 step {t}/{STEPS}] ({time.time()-t0:.0f}s)", flush=True)
    inv = inventory(sae, ap, ac)
    rows = [dict(p0=p0, beta_mult=bm, eps=e, seed=s, **inv[i])
            for i, (p0, bm, e, s) in enumerate(runs)]
    with open("results_p0.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[round5] {Rn} runs in {time.time()-t0:.0f}s", flush=True)
    if not SMOKE:
        open("done_r5.flag", "w").write("ok\n")

if __name__ == "__main__":
    main()
