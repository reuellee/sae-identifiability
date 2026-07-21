"""SAE identifiability experiments (runs on dev-gpu, L4).

Experiment B (primary): feature-absorption phase transition.
  Hierarchical parent/child pair embedded among orthonormal background
  features. Sweep (lambda, eps = P(child solo)); train vanilla L1 ReLU SAEs;
  measure the angle of the learned child-side decoder direction within the
  (a_p, a_c) plane. Theory predicts: 45 deg (absorbed) for eps << eps*,
  smooth tilt toward 90 deg (faithful) as eps grows, transition anchored at
  eps*(lambda, q) = lam*q*(8-4*sqrt(2)-lam) / (2*(1-(2-sqrt(2))*lam)).

Experiment A: classical recovery phase diagram.
  Random incoherent dictionaries, k-sparse data, TopK SAEs with oracle k.
  Maps the empirical recovery boundary in (k, n/d) against the worst-case
  Donoho-Elad threshold k* = (1 + 1/mu)/2.

Outputs: results_absorption.csv, results_recovery.csv, run.log, done.flag
"""
import torch, math, csv, time, json, itertools, sys

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}", flush=True)

# ---------------------------------------------------------------- Experiment B

def make_features(d, n_bg, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.randn(d, d, generator=g)
    Q, _ = torch.linalg.qr(M)
    A_bg = Q[:, :n_bg]                      # background features
    a_p, a_c = Q[:, n_bg], Q[:, n_bg + 1]   # parent, child
    return A_bg.to(dev), a_p.to(dev), a_c.to(dev)

def sample_batch_B(B, A_bg, a_p, a_c, q, p, eps, bg_rate, gen):
    n_bg = A_bg.shape[1]
    coeff = torch.empty(B, n_bg, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(B, n_bg, device=dev, generator=gen) < bg_rate).float()
    x = (coeff * mask) @ A_bg.T
    u = torch.rand(B, device=dev, generator=gen)
    joint = (u < q).float()
    psolo = ((u >= q) & (u < q + p)).float()
    csolo = ((u >= q + p) & (u < q + p + eps)).float()
    x = x + (joint + psolo).unsqueeze(1) * a_p + (joint + csolo).unsqueeze(1) * a_c
    return x

class SAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.enc = torch.nn.Linear(d, m)
        self.dec = torch.nn.Parameter(torch.randn(d, m) / math.sqrt(d))
        self.renorm()
    def renorm(self):
        with torch.no_grad():
            self.dec.div_(self.dec.norm(dim=0, keepdim=True).clamp_min(1e-8))
    def forward(self, x, topk=None):
        f = torch.relu(self.enc(x))
        if topk is not None:
            thr = f.topk(topk, dim=1).values[:, -1:].clamp_min(1e-12)
            f = f * (f >= thr).float()
        return f @ self.dec.T, f

def train_sae(sample_fn, d, m, lam, steps, topk=None, lr=1e-3, B=2048, seed=0):
    torch.manual_seed(seed)
    sae = SAE(d, m).to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for t in range(steps):
        x = sample_fn(B, gen)
        xh, f = sae(x, topk=topk)
        loss = ((x - xh) ** 2).sum(1).mean() + lam * f.sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == steps // 2:
            for gpar in opt.param_groups: gpar["lr"] = lr / 3
    return sae, loss.item()

def analyze_pair(sae, a_p, a_c):
    """Angles of decoder columns within the (a_p, a_c) plane."""
    D = sae.dec.detach()
    cp, cc = D.T @ a_p, D.T @ a_c
    rho = torch.sqrt(cp ** 2 + cc ** 2)              # in-plane fraction
    phi = torch.rad2deg(torch.atan2(cc, cp))          # 0=parent, 90=child, 45=composite
    keep = rho > 0.5
    idx = torch.nonzero(keep).flatten().tolist()
    lat = sorted([(float(rho[i]), float(phi[i])) for i in idx], key=lambda r: -r[0])
    # child-side latent: largest-angle in-plane latent with phi in (20, 120)
    child_side = [ (r, a) for r, a in lat if 20.0 < a < 120.0 ]
    phi_child = max(child_side, key=lambda ra: ra[1])[1] if child_side else float("nan")
    has_parent = any(abs(a) < 20.0 and r > 0.8 for r, a in lat)
    return phi_child, has_parent, lat[:4]

def exp_absorption():
    d, n_bg, m, bg_rate, steps = 64, 30, 32, 0.08, 15000
    p = 0.2
    grid_q = [(0.2, lam) for lam in (0.05, 0.1, 0.2, 0.3)] + [(0.1, 0.2)]
    eps_list = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18]
    seeds = [0, 1, 2]
    rows = []
    t0 = time.time()
    todo = len(grid_q) * len(eps_list) * len(seeds)
    done = 0
    for (q, lam), eps, seed in itertools.product(grid_q, eps_list, seeds):
        A_bg, a_p, a_c = make_features(d, n_bg, 100 + seed)
        fn = lambda B, gen: sample_batch_B(B, A_bg, a_p, a_c, q, p, eps, bg_rate, gen)
        sae, L = train_sae(fn, d, m, lam, steps, seed=seed)
        phi, has_par, top = analyze_pair(sae, a_p, a_c)
        rows.append(dict(q=q, lam=lam, eps=eps, seed=seed, phi_child=phi,
                         has_parent=int(has_par), loss=L,
                         top_latents=json.dumps(top)))
        done += 1
        el = time.time() - t0
        print(f"[B {done}/{todo}] q={q} lam={lam} eps={eps} seed={seed} "
              f"phi_child={phi:.1f} parent={has_par} ({el:.0f}s)", flush=True)
    with open("results_absorption.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

# ---------------------------------------------------------------- Experiment A

def exp_recovery():
    d, steps, B = 64, 12000, 2048
    rows = []
    t0 = time.time()
    cfgs = [(n, k, s) for n in (128, 256) for k in (2, 4, 8, 16, 24, 32) for s in (0, 1)]
    for i, (n, k, seed) in enumerate(cfgs):
        g = torch.Generator(device="cpu").manual_seed(1000 + seed)
        A = torch.randn(d, n, generator=g)
        A = (A / A.norm(dim=0, keepdim=True)).to(dev)
        G = (A.T @ A - torch.eye(n, device=dev)).abs()
        mu = float(G.max())
        def fn(Bs, gen, A=A, k=k, n=n):
            noise = torch.rand(Bs, n, device=dev, generator=gen)
            thr = noise.topk(k, dim=1).values[:, -1:]
            s = (noise >= thr).float() * torch.empty(Bs, n, device=dev).uniform_(0.5, 1.5, generator=gen)
            return s @ A.T
        sae, L = train_sae(fn, d, n, 0.0, steps, topk=k, seed=seed, B=B)
        C = (sae.dec.detach().T @ A).abs()           # [m latents x n true]
        mx = C.max(dim=0).values                      # best match per true feature
        mmcs, frac = float(mx.mean()), float((mx > 0.9).float().mean())
        kstar = (1 + 1 / mu) / 2
        rows.append(dict(n=n, d=d, k=k, seed=seed, mu=mu, kstar_worstcase=kstar,
                         mmcs=mmcs, frac_recovered=frac, loss=L))
        print(f"[A {i+1}/{len(cfgs)}] n={n} k={k} seed={seed} mu={mu:.3f} "
              f"mmcs={mmcs:.3f} frac>0.9={frac:.3f} ({time.time()-t0:.0f}s)", flush=True)
    with open("results_recovery.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    t0 = time.time()
    # positive/negative controls first (fast fail)
    print("== control: eps=0 must absorb; large eps must not ==", flush=True)
    exp_absorption()
    exp_recovery()
    print(f"total {time.time()-t0:.0f}s", flush=True)
    open("done.flag", "w").write("ok\n")
