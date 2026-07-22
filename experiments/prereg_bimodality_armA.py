"""Arm A of the pre-registered bimodality-estimator experiment (CPU-feasible).

Implements notes/prereg-bimodality-estimator.md exactly: H1 (binarized no-go),
H2 (within-composite bimodality estimator), H3 (SNR boundary), H4 (multi-child).
Trainer = round-2 BSAE/train_batched (same hyperparameters: BATCH=2048,
STEPS=15000, Adam 1e-3 with lr/3 at half, decoder renorm, loss SUM over d).

ANALYSIS CHOICES LOCKED BEFORE RUNNING (pre-reg left them open; fixed here so
outcomes cannot be rationalized):
  - Generative model: bg = 30 orthonormal distractors, iid Bernoulli(0.08),
    coeff U(0.8,1.2) (round-2 construction, QR of seed 100+s); host event w.p.
    P_HOST=0.25 adds v_p; child i co-fires w.p. rho_i GIVEN host (child-solo
    rate = 0 < eps* -> absorption predicted); additive N(0, sigma^2 I) noise.
  - Binarize latent j "fires" at f_j > THETA = 0.05 (absolute; activations O(1)).
  - M1 primary = NULL-CALIBRATED excess TV: TV(eval@rho=0.02, eval@rho=0.20)
    minus TV(two independent eval sets both @rho=0.02), same N, same frozen SAE.
    Raw empirical TV has O(sqrt(K/N)) positive bias (K ~ thousands of distinct
    bg signatures), so raw TV < 0.02 is unattainable even under exact H1; the
    null term has identical bias. Also reported: in-plane TV (signature
    restricted to {parent-latent, composite-latent, child-latent} — tiny
    signature space, negligible bias) and the labeled diagnostic
    TV(host-only sigs, host+child sigs) which tests Section-2's mechanism
    directly.
  - M2 estimator exactly as registered: 2-component 1-D GMM (EM, 10 restarts,
    200 iters) on composite activations > 1e-6 from an eval set at the run's
    own training rho; rho_hat = weight of the higher-mean component.
  - M3 binarized-route estimator: rho_hat_bin = P(sig contains composite)
    / P(sig contains composite OR parent-latent). Under H1 (both
    sub-populations fire the composite identically) this is degenerate (-> 1);
    if a dedicated parent latent leaks child presence it becomes the natural
    signature-count estimator. Charitable to the binarized route either way.
  - Composite latent = argmax_j |cos(D_j, w_true)|, w_true = (v_p + sum_i
    v_ci)/sqrt(1+m); ABSORBED (registered criterion) iff that cos > 0.98.
    Non-absorbing seeds are EXCLUDED from confirmatory metrics and disclosed
    with counts (round-3 precedent). Oracle directions are used only to
    IDENTIFY/score, never inside the estimator (Arm A scope, per pre-reg).

BLOCKS (16 seeds each; R,S,M registered; X exploratory, disclosed):
  R: m_lat=32, sigma=0,    rho in {0.02,0.05,0.10,0.20}          (M1,M2,M3)
  S: m_lat=32, rho=0.10,   sigma in {0,0.05,0.1,0.2,0.4}         (M4)
  M: m_lat=32, sigma=0.1,  m_children in {1,2,3}, rates
     {0.10} / {0.06,0.14} / {0.04,0.10,0.16}                     (M5)
  X: m_lat=31 (capacity-forced: 30 bg + 1 leaves ONE in-plane latent, the
     regime Section 2 idealizes — host-only events must reuse the composite),
     sigma=0, rho grid as R.                                     (exploratory)

SMOKE=1 runs a tiny end-to-end version. Outputs results/prereg_armA/armA_runs.csv.
"""
import torch, math, csv, time, os
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE") == "1"
torch.set_num_threads(os.cpu_count() or 4)
print(f"device={dev} threads={torch.get_num_threads()} smoke={SMOKE}", flush=True)

RT2 = math.sqrt(2)
D_AMB, N_BG, BG_RATE, BATCH = 64, 30, 0.08, 2048
M_LAT = 32
STEPS = 100 if SMOKE else 15000
P_HOST = 0.25
LAM = 0.2
THETA = 0.05          # binarization threshold
N_EV_M1 = 4000 if SMOKE else 150_000   # per eval set (x3 sets)
N_EV_M2 = 6000 if SMOKE else 200_000
GMM_MAX = 50_000      # subsample cap for EM
SEEDS = 2 if SMOKE else 16
OUTDIR = os.path.join(os.path.dirname(__file__) or ".", "..", "results", "prereg_armA")

# ------------------------------------------------- round-2 trainer (verbatim)

class BSAE(torch.nn.Module):
    """R independent SAEs trained as one program. W:[R,m,d] D:[R,d,m]."""
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
    def forward(self, x):
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, self.W) + self.b.unsqueeze(1))
        return torch.einsum('rbm,rdm->rbd', f, self.D), f

def train_batched(sae, sample_fn, lam_vec, steps, lr=1e-3, tag=""):
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(steps):
        x = sample_fn(BATCH, gen)
        xh, f = sae(x)
        rec = ((x - xh) ** 2).sum(-1).mean(-1)        # [R] (SUM over d)
        l1 = f.sum(-1).mean(-1)
        loss = (rec + lam_vec * l1).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == steps // 2:
            for gp in opt.param_groups: gp["lr"] = lr / 3
        if t % 2000 == 0:
            print(f"  [{tag} step {t}/{steps}] mean_rec={float(rec.mean()):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return rec.detach()

# ------------------------------------------------- data model (per pre-reg)

def make_feature_sets(seed_bases, n_special=4):
    """Round-2 construction: QR of randn(seed 100+s); 30 bg + up to 4 special."""
    Ms = torch.stack([torch.randn(D_AMB, D_AMB,
                                  generator=torch.Generator().manual_seed(b))
                      for b in seed_bases])
    Q, _ = torch.linalg.qr(Ms)
    A_bg = Q[:, :, :N_BG].contiguous().to(dev)
    spec = [Q[:, :, N_BG + i].contiguous().to(dev) for i in range(n_special)]
    return A_bg, spec           # spec = [v_p, c1, c2, c3]

def sample_runs(Bs, gen, Abg, vp, vcs, rho_mat, sig_vec):
    """x for R runs. rho_mat:[R,3] child co-fire rates given host; sig_vec:[R]."""
    Rn = Abg.shape[0]
    coeff = torch.empty(Rn, Bs, N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(Rn, Bs, N_BG, device=dev, generator=gen) < BG_RATE).float()
    x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
    host = (torch.rand(Rn, Bs, device=dev, generator=gen) < P_HOST).float()
    x = x + host.unsqueeze(-1) * vp.unsqueeze(1)
    for i, vc in enumerate(vcs):
        ch = host * (torch.rand(Rn, Bs, device=dev, generator=gen)
                     < rho_mat[:, i].unsqueeze(1)).float()
        x = x + ch.unsqueeze(-1) * vc.unsqueeze(1)
    x = x + sig_vec.view(-1, 1, 1) * torch.randn(Rn, Bs, D_AMB, device=dev,
                                                 generator=gen)
    return x

def sample_labeled(Bs, gen, Abg1, vp1, vcs1, rho_row, sig):
    """Single-run version returning (x, host, child masks) for diagnostics."""
    coeff = torch.empty(Bs, N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(Bs, N_BG, device=dev, generator=gen) < BG_RATE).float()
    x = torch.einsum('bn,dn->bd', coeff * mask, Abg1)
    host = (torch.rand(Bs, device=dev, generator=gen) < P_HOST).float()
    x = x + host.unsqueeze(-1) * vp1
    chs = []
    for i, vc in enumerate(vcs1):
        ch = host * (torch.rand(Bs, device=dev, generator=gen) < rho_row[i]).float()
        chs.append(ch)
        x = x + ch.unsqueeze(-1) * vc
    x = x + sig * torch.randn(Bs, D_AMB, device=dev, generator=gen)
    return x, host, chs

# ------------------------------------------------- analysis primitives

def gmm_fit(x, k=2, restarts=10, iters=200, seed=0):
    """1-D k-component GMM by EM; returns (means, weights) best-loglik."""
    rng = np.random.default_rng(seed)
    if len(x) > GMM_MAX:
        x = rng.choice(x, GMM_MAX, replace=False)
    best_ll, best = -np.inf, None
    for r in range(restarts):
        mu = rng.choice(x, k, replace=False).astype(np.float64)
        var = np.full(k, x.var() + 1e-6); w = np.full(k, 1.0 / k)
        for _ in range(iters):
            logp = (-0.5 * ((x[:, None] - mu) ** 2 / var + np.log(2 * np.pi * var))
                    + np.log(w))
            m = logp.max(1, keepdims=True)
            p = np.exp(logp - m); s = p.sum(1, keepdims=True)
            ll = (np.log(s) + m[:, 0:1]).sum()
            p /= s
            n = p.sum(0) + 1e-12
            w = n / len(x)
            mu = (p * x[:, None]).sum(0) / n
            var = np.maximum((p * (x[:, None] - mu) ** 2).sum(0) / n, 1e-8)
        if ll > best_ll:
            best_ll, best = ll, (mu.copy(), w.copy())
    return best

def signatures(f, theta=THETA):
    """Pack binarized [N,m] activation pattern into hashable rows."""
    bits = (f > theta).astype(np.uint8)
    return np.packbits(bits, axis=1)

def tv_dist(sa, sb):
    """Total variation between two empirical signature distributions."""
    allsig = np.concatenate([sa, sb])
    uniq, inv = np.unique(allsig, axis=0, return_inverse=True)
    ca = np.bincount(inv[:len(sa)], minlength=len(uniq)) / len(sa)
    cb = np.bincount(inv[len(sa):], minlength=len(uniq)) / len(sb)
    return 0.5 * np.abs(ca - cb).sum()

def encode(sae, i, x):
    W = sae.W.detach()[i]; b = sae.b.detach()[i]
    return torch.relu(x @ W.T + b).cpu().numpy()

# ------------------------------------------------- run + evaluate one block

def build_runs():
    runs = []   # dict per run
    rho_grid = [0.02, 0.05, 0.10, 0.20]
    if SMOKE:
        rho_grid = [0.02, 0.20]
    for rho in rho_grid:
        for s in range(SEEDS):
            runs.append(dict(block="R", rhos=(rho, 0, 0), sigma=0.0, seed=s))
    for sig in ([0.0, 0.4] if SMOKE else [0.0, 0.05, 0.1, 0.2, 0.4]):
        for s in range(SEEDS):
            runs.append(dict(block="S", rhos=(0.10, 0, 0), sigma=sig, seed=s))
    mc_rates = {1: (0.10, 0, 0), 2: (0.06, 0.14, 0), 3: (0.04, 0.10, 0.16)}
    for m in ([1, 2] if SMOKE else [1, 2, 3]):
        for s in range(SEEDS):
            runs.append(dict(block="M", rhos=mc_rates[m], sigma=0.1, seed=s))
    return runs

def build_runs_X():
    rho_grid = [0.02, 0.20] if SMOKE else [0.02, 0.05, 0.10, 0.20]
    return [dict(block="X", rhos=(rho, 0, 0), sigma=0.0, seed=s)
            for rho in rho_grid for s in range(SEEDS)]

def train_program(runs, m_lat, tag):
    Rn = len(runs)
    A_bg_s, spec_s = make_feature_sets([100 + r["seed"] for r in runs])
    vp, c1, c2, c3 = spec_s
    rho_mat = torch.tensor([r["rhos"] for r in runs], dtype=torch.float32,
                           device=dev)
    sig_vec = torch.tensor([r["sigma"] for r in runs], dtype=torch.float32,
                           device=dev)
    torch.manual_seed(11)
    sae = BSAE(Rn, D_AMB, m_lat).to(dev)
    fn = lambda Bs, g: sample_runs(Bs, g, A_bg_s, vp, [c1, c2, c3], rho_mat, sig_vec)
    rec = train_batched(sae, fn, torch.full((Rn,), LAM, device=dev), STEPS,
                        tag=f"{tag} R={Rn} m={m_lat}")
    return sae, A_bg_s, spec_s, rec

def eval_run(sae, i, run, Abg1, vp1, vcs_all, rec_i):
    """All per-run metrics -> dict (CSV row)."""
    m_ch = sum(1 for r in run["rhos"] if r > 0)
    vcs = vcs_all[:m_ch]
    w_true = (vp1 + sum(vcs)) / math.sqrt(1 + m_ch)
    D = sae.D.detach()[i]                                   # [d, m]
    cos_w = (D * w_true.unsqueeze(1)).sum(0)
    comp = int(cos_w.abs().argmax()); cos_comp = float(cos_w.abs().max())
    cos_p = (D * vp1.unsqueeze(1)).sum(0)
    par = int(cos_p.abs().argmax()); cos_par = float(cos_p.abs().max())
    cos_c = (D * vcs_all[0].unsqueeze(1)).sum(0)
    chl = int(cos_c.abs().argmax()); cos_chl = float(cos_c.abs().max())
    absorbed = int(cos_comp > 0.98)
    row = dict(block=run["block"], m_children=m_ch, rho1=run["rhos"][0],
               rho2=run["rhos"][1], rho3=run["rhos"][2], sigma=run["sigma"],
               m_lat=sae.D.shape[2], seed=run["seed"], absorbed=absorbed,
               cos_comp=round(cos_comp, 4), cos_parent=round(cos_par, 4),
               cos_child=round(cos_chl, 4), comp_idx=comp, parent_idx=par,
               rec=round(float(rec_i), 4))
    rho_row = run["rhos"]; sig = run["sigma"]

    # ---- M2/M3: estimator at the run's own training rho -------------------
    g = torch.Generator(device=dev).manual_seed(9000 + run["seed"])
    x, host, chs = sample_labeled(N_EV_M2, g, Abg1, vp1, vcs_all, rho_row, sig)
    f = encode(sae, i, x)
    a = f[:, comp]
    fire = a > 1e-6
    row["comp_fire_rate"] = round(float(fire.mean()), 5)
    if fire.sum() >= 20:
        mu, wgt = gmm_fit(a[fire].astype(np.float64), k=2,
                          seed=1000 + run["seed"])
        hi = int(np.argmax(mu))
        row.update(rho_hat_gmm=round(float(wgt[hi]), 5),
                   gmm_mu_lo=round(float(mu[1 - hi]), 4),
                   gmm_mu_hi=round(float(mu[hi]), 4),
                   gmm_w_hi=round(float(wgt[hi]), 5), n_fire=int(fire.sum()))
    else:
        row.update(rho_hat_gmm=float("nan"), gmm_mu_lo=float("nan"),
                   gmm_mu_hi=float("nan"), gmm_w_hi=float("nan"),
                   n_fire=int(fire.sum()))
    sig_all = f > THETA
    has_comp = sig_all[:, comp]; has_par = sig_all[:, par]
    denom = (has_comp | has_par).mean()
    row["rho_hat_bin"] = round(float(has_comp.mean() / denom), 5) if denom > 0 \
        else float("nan")
    # labeled diagnostics (scoring only): conditional fire rates + magnitudes
    h = host.cpu().numpy() > 0.5
    c = (chs[0].cpu().numpy() > 0.5) if m_ch >= 1 else np.zeros(len(a), bool)
    row["fire|host_only"] = round(float(fire[h & ~c].mean()), 4) if (h & ~c).any() else float("nan")
    row["fire|joint"] = round(float(fire[h & c].mean()), 4) if (h & c).any() else float("nan")
    row["act|host_only"] = round(float(a[h & ~c & fire].mean()), 4) if (h & ~c & fire).any() else float("nan")
    row["act|joint"] = round(float(a[h & c & fire].mean()), 4) if (h & c & fire).any() else float("nan")

    # ---- M1: TV between frozen-SAE signature dists at rho 0.02 vs 0.20 ----
    if run["block"] in ("R", "X"):
        def sigs_at(rho, gseed):
            gg = torch.Generator(device=dev).manual_seed(gseed)
            xx, _, _ = sample_labeled(N_EV_M1, gg, Abg1, vp1, vcs_all,
                                      (rho, 0, 0), sig)
            return encode(sae, i, xx)
        fa = sigs_at(0.02, 20000 + run["seed"])
        fb = sigs_at(0.20, 30000 + run["seed"])
        fa2 = sigs_at(0.02, 40000 + run["seed"])          # null (same rho)
        row["tv_full"] = round(float(tv_dist(signatures(fa), signatures(fb))), 5)
        row["tv_null"] = round(float(tv_dist(signatures(fa), signatures(fa2))), 5)
        keep = [par, comp, chl]
        row["tv_inplane"] = round(float(tv_dist(
            signatures(fa[:, keep]), signatures(fb[:, keep]))), 5)
        row["tv_inplane_null"] = round(float(tv_dist(
            signatures(fa[:, keep]), signatures(fa2[:, keep]))), 5)
        # direct mechanism test: host-only vs host+child signatures
        gg = torch.Generator(device=dev).manual_seed(50000 + run["seed"])
        xl, hl, cl = sample_labeled(N_EV_M1, gg, Abg1, vp1, vcs_all,
                                    (0.5, 0, 0), sig)
        fl = encode(sae, i, xl)
        hm = hl.cpu().numpy() > 0.5; cm = cl[0].cpu().numpy() > 0.5
        row["tv_cond"] = round(float(tv_dist(
            signatures(fl[hm & ~cm]), signatures(fl[hm & cm]))), 5)
    # ---- H4: k=(m+1)-component fit (exploratory) --------------------------
    if run["block"] == "M" and m_ch >= 2 and fire.sum() >= 20:
        mu, wgt = gmm_fit(a[fire].astype(np.float64), k=m_ch + 1,
                          seed=2000 + run["seed"])
        order = np.argsort(mu)
        row["h4_mus"] = "|".join(f"{mu[j]:.3f}" for j in order)
        row["h4_ws"] = "|".join(f"{wgt[j]:.4f}" for j in order)
    return row

# ------------------------------------------------- main

if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for runs, m_lat, tag in [(build_runs(), M_LAT, "RSM"),
                             (build_runs_X(), M_LAT - 1, "X")]:
        sae, A_bg_s, spec_s, rec = train_program(runs, m_lat, tag)
        vp, c1, c2, c3 = spec_s
        for i, run in enumerate(runs):
            rows.append(eval_run(sae, i, run, A_bg_s[i], vp[i],
                                 [c1[i], c2[i], c3[i]], rec[i]))
            if i % 16 == 15:
                print(f"  [{tag} eval {i+1}/{len(runs)}]", flush=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    out = os.path.join(OUTDIR, "armA_runs.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows) total {time.time()-t0:.0f}s", flush=True)
