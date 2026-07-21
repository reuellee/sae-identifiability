"""Round 2 (batched): SAE-identifiability follow-ups. Runs on dev-gpu (L4).

B-fine: fine eps sweep around predicted eps*(lam,q), 16 seeds/cell, ALL runs of
  a cell trained simultaneously as one batched tensor program (run dim folded
  into the einsums), so a cell of 208 runs costs ~one round-1 run's wall-clock.
  Child latent identified FUNCTIONALLY (argmax encoder response on a child-solo
  probe) alongside round-1's max-angle metric, which is biased toward 90 deg
  near the transition. Goal: eps_c per (q,lam) with seed CIs -> regress against
  eps*(lam,q) = lam*q*(8-4*sqrt(2)-lam) / (2*(1-(2-sqrt(2))*lam)).

A-oracle: recovery experiment with trainability controls per (n,k):
  random init vs random init + dead-latent reinit vs ORACLE init (decoder = true
  dictionary). If oracle init stays at the dictionary where random init fails,
  round-1's low recovery at small k was optimization failure, not
  non-identifiability (round 1 was non-monotone in k, peaking at k=8).

SMOKE=1 env var runs a tiny version of everything (shapes + pipeline check).
Outputs: results_b_fine.csv, results_a_oracle.csv, done_r2.flag
"""
import torch, math, csv, time, json, os

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE") == "1"
print(f"device={dev} smoke={SMOKE}", flush=True)

RT2 = math.sqrt(2)
D_AMB, N_BG, M_LAT, BG_RATE, BATCH = 64, 30, 32, 0.08, 2048
STEPS_B = 60 if SMOKE else 15000
STEPS_A = 60 if SMOKE else 12000
P0 = 0.2

def eps_star(lam, q):
    return lam * q * (8 - 4 * RT2 - lam) / (2 * (1 - (2 - RT2) * lam))

# ------------------------------------------------------------- batched SAE

class BSAE(torch.nn.Module):
    """R independent SAEs trained as one program. W:[R,m,d] D:[R,d,m]."""
    def __init__(self, R, d, m):
        super().__init__()
        bound = 1 / math.sqrt(d)                      # nn.Linear default init
        self.W = torch.nn.Parameter(torch.empty(R, m, d).uniform_(-bound, bound))
        self.b = torch.nn.Parameter(torch.empty(R, m).uniform_(-bound, bound))
        self.D = torch.nn.Parameter(torch.randn(R, d, m) / math.sqrt(d))
        self.renorm()
    def renorm(self):
        with torch.no_grad():
            self.D.div_(self.D.norm(dim=1, keepdim=True).clamp_min(1e-8))
    def forward(self, x, topk=None):
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, self.W) + self.b.unsqueeze(1))
        if topk is not None:
            thr = f.topk(topk, dim=-1).values[..., -1:].clamp_min(1e-12)
            f = f * (f >= thr).float()
        return torch.einsum('rbm,rdm->rbd', f, self.D), f

def train_batched(sae, sample_fn, lam_vec, steps, topk=None, lr=1e-3,
                  every1k_cb=None, tag=""):
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(steps):
        x = sample_fn(BATCH, gen)
        xh, f = sae(x, topk=topk)
        rec = ((x - xh) ** 2).sum(-1).mean(-1)        # [R]
        l1 = f.sum(-1).mean(-1)                       # [R]
        loss = (rec + lam_vec * l1).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == steps // 2:
            for gp in opt.param_groups: gp["lr"] = lr / 3
        if every1k_cb is not None and t % 1000 == 999:
            every1k_cb(sae, f, x, xh)
        if t % 2000 == 0:
            print(f"  [{tag} step {t}/{steps}] mean_rec={float(rec.mean()):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return rec.detach()

# ------------------------------------------------------------- B-fine

def make_feature_sets(seed_bases, n_special):
    """Batched QR, same construction as round 1's make_features(100+seed)."""
    Ms = torch.stack([torch.randn(D_AMB, D_AMB,
                                  generator=torch.Generator().manual_seed(b))
                      for b in seed_bases])
    Q, _ = torch.linalg.qr(Ms)
    A_bg = Q[:, :, :N_BG].to(dev)
    spec = [Q[:, :, N_BG + i].contiguous().to(dev) for i in range(n_special)]
    return A_bg, spec

def sample_cell(Bs, gen, Abg, ap, ac, q, eps_vec):
    Rn = Abg.shape[0]
    coeff = torch.empty(Rn, Bs, N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(Rn, Bs, N_BG, device=dev, generator=gen) < BG_RATE).float()
    x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
    u = torch.rand(Rn, Bs, device=dev, generator=gen)
    e = eps_vec.view(-1, 1)
    joint = (u < q).float()
    psolo = ((u >= q) & (u < q + P0)).float()
    csolo = ((u >= q + P0) & (u < q + P0 + e)).float()
    return (x + (joint + psolo).unsqueeze(-1) * ap.unsqueeze(1)
              + (joint + csolo).unsqueeze(-1) * ac.unsqueeze(1))

def analyze_cell(sae, ap, ac):
    Dm = sae.D.detach()
    cp = torch.einsum('rdm,rd->rm', Dm, ap)
    cc = torch.einsum('rdm,rd->rm', Dm, ac)
    rho = (cp ** 2 + cc ** 2).sqrt()
    phi = torch.rad2deg(torch.atan2(cc, cp))
    # functional child: latent with max encoder response to a child-solo input
    fch = torch.relu(torch.einsum('rmd,rd->rm', sae.W.detach(), ac) + sae.b.detach())
    act, idx = fch.max(dim=1)
    phi_func = phi.gather(1, idx.unsqueeze(1)).squeeze(1)
    rho_func = rho.gather(1, idx.unsqueeze(1)).squeeze(1)
    # round-1-compatible max-angle metric
    cand = (rho > 0.5) & (phi > 20) & (phi < 120)
    phi_masked = torch.where(cand, phi, torch.full_like(phi, float('-inf')))
    phi_max = phi_masked.max(dim=1).values
    has_parent = ((phi.abs() < 20) & (rho > 0.8)).any(dim=1)
    return phi_func, rho_func, act, phi_max, has_parent

def run_B_fine():
    cells = [(0.1, 0.2), (0.2, 0.05), (0.2, 0.1), (0.2, 0.2), (0.2, 0.3)]
    n_eps, n_seeds = (3, 2) if SMOKE else (12, 16)
    rows = []
    for ci, (q, lam) in enumerate(cells if not SMOKE else cells[:1]):
        es = eps_star(lam, q)
        # geometric grid es/4 .. 2.5*es, plus eps=0 control
        grid = [0.0] + [es / 4 * (10.0 ** (i / (n_eps - 1))) for i in range(n_eps)]
        seeds = list(range(n_seeds))
        A_bg_s, (ap_s, ac_s) = make_feature_sets([100 + s for s in seeds], 2)
        runs = [(e, s) for e in grid for s in seeds]
        Rn = len(runs)
        sidx = torch.tensor([s for _, s in runs])
        Abg, ap, ac = A_bg_s[sidx], ap_s[sidx], ac_s[sidx]
        eps_vec = torch.tensor([e for e, _ in runs], device=dev, dtype=torch.float32)
        lam_vec = torch.full((Rn,), lam, device=dev)
        torch.manual_seed(7 + ci)
        sae = BSAE(Rn, D_AMB, M_LAT).to(dev)
        fn = lambda Bs, g: sample_cell(Bs, g, Abg, ap, ac, q, eps_vec)
        rec = train_batched(sae, fn, lam_vec, STEPS_B, tag=f"B q={q} lam={lam} R={Rn}")
        phi_f, rho_f, act, phi_m, has_p = analyze_cell(sae, ap, ac)
        for i, (e, s) in enumerate(runs):
            rows.append(dict(q=q, lam=lam, eps_star=es, eps=e, seed=s,
                             phi_func=float(phi_f[i]), rho_func=float(rho_f[i]),
                             act_child=float(act[i]), phi_maxangle=float(phi_m[i]),
                             has_parent=int(has_p[i]), rec=float(rec[i])))
        print(f"[B-fine cell {ci+1}] q={q} lam={lam} eps*={es:.4f} done", flush=True)
    with open("results_b_fine.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------- A-oracle

def run_A_oracle():
    ks = (2, 32) if SMOKE else (2, 4, 8, 16, 24, 32)
    ns = (128,) if SMOKE else (128, 256)
    n_seeds = 2 if SMOKE else 4
    conds = ["random", "reinit", "oracle"]
    rows = []
    for n in ns:
        for k in ks:
            run_list = [(c, s) for c in conds for s in range(n_seeds)]
            Rn = len(run_list)
            As = {}
            for s in range(n_seeds):
                g = torch.Generator().manual_seed(1000 + s)
                A = torch.randn(D_AMB, n, generator=g)
                As[s] = (A / A.norm(dim=0, keepdim=True)).to(dev)
            A_runs = torch.stack([As[s] for _, s in run_list])       # [R,d,n]
            torch.manual_seed(50 + k)
            sae = BSAE(Rn, D_AMB, n).to(dev)
            with torch.no_grad():
                for i, (c, s) in enumerate(run_list):
                    if c == "oracle":
                        sae.D[i] = As[s]
                        sae.W[i] = As[s].T
                        sae.b[i] = 0.0
            sae.renorm()
            reinit_mask = torch.tensor([c == "reinit" for c, _ in run_list], device=dev)
            def fnA(Bs, gen):
                noise = torch.rand(Rn, Bs, n, device=dev, generator=gen)
                thr = noise.topk(k, dim=-1).values[..., -1:]
                sc = ((noise >= thr).float()
                      * torch.empty(Rn, Bs, n, device=dev).uniform_(0.5, 1.5, generator=gen))
                return torch.einsum('rbn,rdn->rbd', sc, A_runs)
            def reinit_cb(sae, f, x, xh):
                with torch.no_grad():
                    mx = f.max(dim=1).values                          # [R,m]
                    dead = (mx < 1e-6) & reinit_mask.unsqueeze(1)
                    r_i, m_i = dead.nonzero(as_tuple=True)
                    if len(r_i) == 0: return
                    res = (x - xh)
                    b_i = torch.randint(0, x.shape[1], (len(r_i),), device=dev)
                    v = res[r_i, b_i]
                    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    sae.D[r_i, :, m_i] = v
                    sae.W[r_i, m_i, :] = v
                    sae.b[r_i, m_i] = 0.0
            lam_vec = torch.zeros(Rn, device=dev)
            rec = train_batched(sae, fnA, lam_vec, STEPS_A, topk=k,
                                every1k_cb=reinit_cb, tag=f"A n={n} k={k}")
            C = torch.einsum('rdm,rdn->rmn', sae.D.detach(), A_runs).abs()
            mx = C.max(dim=1).values                                  # [R,n]
            for i, (c, s) in enumerate(run_list):
                rows.append(dict(n=n, k=k, cond=c, seed=s,
                                 mmcs=float(mx[i].mean()),
                                 frac_recovered=float((mx[i] > 0.9).float().mean()),
                                 rec=float(rec[i])))
            print(f"[A-oracle n={n} k={k}] " + "  ".join(
                f"{c}:{sum(float(mx[i].mean()) for i,(cc,_) in enumerate(run_list) if cc==c)/n_seeds:.3f}"
                for c in conds), flush=True)
    with open("results_a_oracle.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    t0 = time.time()
    run_B_fine()
    run_A_oracle()
    print(f"round2 total {time.time()-t0:.0f}s", flush=True)
    if not SMOKE:
        open("done_r2.flag", "w").write("ok\n")
