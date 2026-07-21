"""Round 3 (batched): validate the CORRECTED coherence-penalty theory + rich C3.

C1-boundary: round-1's pre-registered C1 prediction was REFUTED; exact re-analysis
  (theory_merged.py) found the global optimum at beta>=beta*, small eps, is the
  ANTI-ROTATED absorbed pair {~-40deg, ~46deg} (composite keeps absorbing; parent
  rotates negative so the pair is ~orthogonal -> penalty ~0). Corrected boundary
  (lam=q=p0=0.2): faithful wins only for eps > eps**(beta), with eps**(beta*)~0.0112,
  eps**(4beta*)~0.0141, saturating ~0.0224 as beta->inf. THIS experiment sweeps eps
  finely through the predicted boundary at beta in {1,4}beta* x 8 seeds, logging a
  full in-plane inventory (incl. NEGATIVE angles round 1's metric window missed)
  + functional child metric. Success = empirical faithful-basin fraction crosses
  50% near eps**(beta), and anti-rotated geometry visible below it.

C3-rich: two-child vanilla vs matryoshka with continuous metrics (round-1's 0.9
  binary classifier returned all-zeros on matryoshka): per-canonical max cosine
  over {p, c1, c2, comp1, comp2, comp3=(p+c1+c2)/sqrt3, cc=(c1+c2)/sqrt2},
  per-latent in-subspace fraction, top in-subspace latents.

SMOKE=1 for pipeline check. Outputs: results_c1_boundary.csv, results_c3_rich.csv,
done_r3.flag.
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
BSTAR = LAM * Q * (8 - 4 * RT2 - LAM) / 2
PREFIXES = [1, 2, 4, 8, 16, 32]

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

def train_batched(sae, sample_fn, lam, steps, beta_vec=None, matryoshka=False,
                  lr=1e-3, tag=""):
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator(device=dev).manual_seed(1)
    eye = torch.eye(M_LAT, device=dev)
    t0 = time.time()
    for t in range(steps):
        x = sample_fn(BATCH, gen)
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, sae.W) + sae.b.unsqueeze(1))
        if matryoshka:
            rec = 0.0
            for s in PREFIXES:
                xh = torch.einsum('rbm,rdm->rbd', f[:, :, :s], sae.D[:, :, :s])
                rec = rec + ((x - xh) ** 2).sum(-1).mean(-1)
            rec = rec / len(PREFIXES)
        else:
            xh = torch.einsum('rbm,rdm->rbd', f, sae.D)
            rec = ((x - xh) ** 2).sum(-1).mean(-1)
        loss_vec = rec + lam * f.sum(-1).mean(-1)
        if beta_vec is not None:
            G = torch.einsum('rdi,rdj->rij', sae.D, sae.D) - eye
            loss_vec = loss_vec + beta_vec * (G ** 2).sum(dim=(1, 2)) / 2
        loss = loss_vec.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == steps // 2:
            for gp in opt.param_groups: gp["lr"] = lr / 3
        if t % 3000 == 0:
            print(f"  [{tag} step {t}/{steps}] mean_rec={float(rec.mean()):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return rec.detach()

def make_feature_sets(seed_bases, n_special):
    Ms = torch.stack([torch.randn(D_AMB, D_AMB,
                                  generator=torch.Generator().manual_seed(b))
                      for b in seed_bases])
    Qm, _ = torch.linalg.qr(Ms)
    A_bg = Qm[:, :, :N_BG].to(dev)
    spec = [Qm[:, :, N_BG + i].contiguous().to(dev) for i in range(n_special)]
    return A_bg, spec

# ------------------------------------------------------------- C1-boundary

def sample_single(Bs, gen, Abg, ap, ac, eps_vec):
    Rn = Abg.shape[0]
    coeff = torch.empty(Rn, Bs, N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(Rn, Bs, N_BG, device=dev, generator=gen) < BG_RATE).float()
    x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
    u = torch.rand(Rn, Bs, device=dev, generator=gen)
    e = eps_vec.view(-1, 1)
    joint = (u < Q).float()
    psolo = ((u >= Q) & (u < Q + P0)).float()
    csolo = ((u >= Q + P0) & (u < Q + P0 + e)).float()
    return (x + (joint + psolo).unsqueeze(-1) * ap.unsqueeze(1)
              + (joint + csolo).unsqueeze(-1) * ac.unsqueeze(1))

def inventory(sae, ap, ac):
    """Full in-plane inventory per run: top-8 by rho as (rho, phi) incl. negatives,
    plus functional child latent (argmax encoder response on child-solo)."""
    Dm = sae.D.detach()
    cp = torch.einsum('rdm,rd->rm', Dm, ap)
    cc = torch.einsum('rdm,rd->rm', Dm, ac)
    rho = (cp ** 2 + cc ** 2).sqrt()
    phi = torch.rad2deg(torch.atan2(cc, cp))
    fch = torch.relu(torch.einsum('rmd,rd->rm', sae.W.detach(), ac) + sae.b.detach())
    act, idx = fch.max(dim=1)
    out = []
    for r in range(Dm.shape[0]):
        order = torch.argsort(rho[r], descending=True)[:8]
        inv = [(round(float(rho[r][i]), 3), round(float(phi[r][i]), 1))
               for i in order if float(rho[r][i]) > 0.3]
        out.append(dict(
            phi_func=round(float(phi[r][idx[r]]), 1),
            rho_func=round(float(rho[r][idx[r]]), 3),
            act_child=round(float(act[r]), 3),
            inv=json.dumps(inv)))
    return out

def run_c1_boundary():
    eps_grid = [0.004, 0.007, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.040]
    betas = [1.0, 4.0]
    n_seeds = 8
    if SMOKE:
        eps_grid, betas, n_seeds = eps_grid[:2], betas[:1], 2
    seeds = list(range(n_seeds))
    A_bg_s, (ap_s, ac_s) = make_feature_sets([100 + s for s in seeds], 2)
    runs = [(bm, e, s) for bm in betas for e in eps_grid for s in seeds]
    Rn = len(runs)
    sidx = torch.tensor([s for _, _, s in runs])
    Abg, ap, ac = A_bg_s[sidx], ap_s[sidx], ac_s[sidx]
    eps_vec = torch.tensor([e for _, e, _ in runs], device=dev, dtype=torch.float32)
    beta_vec = torch.tensor([bm * BSTAR for bm, _, _ in runs], device=dev,
                            dtype=torch.float32)
    torch.manual_seed(11)
    sae = BSAE(Rn, D_AMB, M_LAT).to(dev)
    fn = lambda Bs, g: sample_single(Bs, g, Abg, ap, ac, eps_vec)
    rec = train_batched(sae, fn, LAM, STEPS, beta_vec=beta_vec,
                        tag=f"C1b R={Rn}")
    inv = inventory(sae, ap, ac)
    rows = [dict(beta_mult=bm, eps=e, seed=s, rec=round(float(rec[i]), 4), **inv[i])
            for i, (bm, e, s) in enumerate(runs)]
    with open("results_c1_boundary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[C1-boundary] {Rn} runs done", flush=True)

# ------------------------------------------------------------- C3-rich

def sample_two(Bs, gen, Abg, ap, c1, c2, eps_vec):
    Rn = Abg.shape[0]
    coeff = torch.empty(Rn, Bs, N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(Rn, Bs, N_BG, device=dev, generator=gen) < BG_RATE).float()
    x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
    u = torch.rand(Rn, Bs, device=dev, generator=gen)
    e = eps_vec.view(-1, 1)
    j1 = (u < Q).float()
    j2 = ((u >= Q) & (u < 2 * Q)).float()
    ps = ((u >= 2 * Q) & (u < 2 * Q + P0)).float()
    s1 = ((u >= 2 * Q + P0) & (u < 2 * Q + P0 + e)).float()
    s2 = ((u >= 2 * Q + P0 + e) & (u < 2 * Q + P0 + 2 * e)).float()
    return (x + (j1 + j2 + ps).unsqueeze(-1) * ap.unsqueeze(1)
              + (j1 + s1).unsqueeze(-1) * c1.unsqueeze(1)
              + (j2 + s2).unsqueeze(-1) * c2.unsqueeze(1))

def run_c3_rich():
    n_seeds = 2 if SMOKE else 6
    modes = ["vanilla", "matryoshka"]
    eps_list = [0.0, 0.01]
    rows = []
    for mode in modes:
        seeds = list(range(n_seeds))
        A_bg_s, (ap_s, c1_s, c2_s) = make_feature_sets([200 + s for s in seeds], 3)
        runs = [(e, s) for e in eps_list for s in seeds]
        Rn = len(runs)
        sidx = torch.tensor([s for _, s in runs])
        Abg, ap, c1, c2 = A_bg_s[sidx], ap_s[sidx], c1_s[sidx], c2_s[sidx]
        eps_vec = torch.tensor([e for e, _ in runs], device=dev, dtype=torch.float32)
        torch.manual_seed(13)
        sae = BSAE(Rn, D_AMB, M_LAT).to(dev)
        fn = lambda Bs, g: sample_two(Bs, g, Abg, ap, c1, c2, eps_vec)
        rec = train_batched(sae, fn, LAM, STEPS, matryoshka=(mode == "matryoshka"),
                            tag=f"C3r {mode} R={Rn}")
        Dm = sae.D.detach()
        canon = dict(parent=ap, child1=c1, child2=c2,
                     comp1=(ap + c1) / RT2, comp2=(ap + c2) / RT2,
                     comp3=(ap + c1 + c2) / math.sqrt(3),
                     cc=(c1 + c2) / RT2)
        cos = {k: torch.einsum('rdm,rd->rm', Dm, v) for k, v in canon.items()}
        # in-subspace fraction: projection onto span(p, c1, c2) (orthonormal)
        pr = sum(torch.einsum('rdm,rd->rm', Dm, v) ** 2 for v in (ap, c1, c2))
        rho_s = pr.sqrt()
        for i, (e, s) in enumerate(runs):
            maxcos = {k: round(float(v[i].abs().max()), 3) for k, v in cos.items()}
            order = torch.argsort(rho_s[i], descending=True)[:6]
            top = [dict(rho=round(float(rho_s[i][j]), 3),
                        p=round(float(cos['parent'][i][j]), 2),
                        c1=round(float(cos['child1'][i][j]), 2),
                        c2=round(float(cos['child2'][i][j]), 2))
                   for j in order if float(rho_s[i][j]) > 0.3]
            rows.append(dict(mode=mode, eps=e, seed=s, rec=round(float(rec[i]), 4),
                             **{f"max_{k}": v for k, v in maxcos.items()},
                             top=json.dumps(top)))
        print(f"[C3-rich {mode}] {Rn} runs done", flush=True)
    with open("results_c3_rich.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    t0 = time.time()
    run_c1_boundary()
    run_c3_rich()
    print(f"round3 total {time.time()-t0:.0f}s", flush=True)
    if not SMOKE:
        open("done_r3.flag", "w").write("ok\n")
