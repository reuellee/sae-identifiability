"""Experiment C: remedy validation on the L4.

C1 coherence penalty (single-child): lam=0.2, q=0.2, eps in {0, .01, .02},
   beta in {0, b*/2, b*, 2b*, 4b*}; predict 45deg -> ~90deg transition near b*.
C2 matryoshka single-child: predict absorption PERSISTS (composite wins slot 1).
C3 two-child: vanilla (predict absorbed: composites present, children absent)
   vs matryoshka geometric prefixes [1,2,4,8,16,32] (predict recovery: children
   present, composites gone).
Outputs: results_remedies.csv, done2.flag
"""
import torch, math, csv, time, json, itertools

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}", flush=True)

D_AMB, N_BG, M_LAT, BG_RATE, STEPS, BATCH = 64, 30, 32, 0.08, 15000, 2048
Q, P0 = 0.2, 0.2
PREFIXES = [1, 2, 4, 8, 16, 32]

def make_features(seed, n_special):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Qm, _ = torch.linalg.qr(torch.randn(D_AMB, D_AMB, generator=g))
    A_bg = Qm[:, :N_BG].to(dev)
    spec = [Qm[:, N_BG + i].to(dev) for i in range(n_special)]
    return A_bg, spec

def batch_single(B, A_bg, a_p, a_c, eps, gen):
    coeff = torch.empty(B, N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(B, N_BG, device=dev, generator=gen) < BG_RATE).float()
    x = (coeff * mask) @ A_bg.T
    u = torch.rand(B, device=dev, generator=gen)
    joint = (u < Q).float(); psolo = ((u >= Q) & (u < Q + P0)).float()
    csolo = ((u >= Q + P0) & (u < Q + P0 + eps)).float()
    return x + (joint + psolo).unsqueeze(1) * a_p + (joint + csolo).unsqueeze(1) * a_c

def batch_two(B, A_bg, a_p, c1, c2, eps, gen):
    coeff = torch.empty(B, N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(B, N_BG, device=dev, generator=gen) < BG_RATE).float()
    x = (coeff * mask) @ A_bg.T
    u = torch.rand(B, device=dev, generator=gen)
    j1 = (u < Q).float(); j2 = ((u >= Q) & (u < 2 * Q)).float()
    ps = ((u >= 2 * Q) & (u < 2 * Q + P0)).float()
    s1 = ((u >= 2 * Q + P0) & (u < 2 * Q + P0 + eps)).float()
    s2 = ((u >= 2 * Q + P0 + eps) & (u < 2 * Q + P0 + 2 * eps)).float()
    return (x + (j1 + j2 + ps).unsqueeze(1) * a_p
              + (j1 + s1).unsqueeze(1) * c1 + (j2 + s2).unsqueeze(1) * c2)

class SAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(D_AMB, M_LAT)
        self.dec = torch.nn.Parameter(torch.randn(D_AMB, M_LAT) / math.sqrt(D_AMB))
        self.renorm()
    def renorm(self):
        with torch.no_grad():
            self.dec.div_(self.dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

def train(sample_fn, lam, beta=0.0, matryoshka=False, seed=0):
    torch.manual_seed(seed)
    sae = SAE().to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for t in range(STEPS):
        x = sample_fn(BATCH, gen)
        f = torch.relu(sae.enc(x))
        if matryoshka:
            rec = 0.0
            for s in PREFIXES:
                xh = f[:, :s] @ sae.dec[:, :s].T
                rec = rec + ((x - xh) ** 2).sum(1).mean()
            rec = rec / len(PREFIXES)
        else:
            xh = f @ sae.dec.T
            rec = ((x - xh) ** 2).sum(1).mean()
        loss = rec + lam * f.sum(1).mean()
        if beta > 0:
            Gr = sae.dec.T @ sae.dec - torch.eye(M_LAT, device=dev)
            loss = loss + beta * (Gr ** 2).sum() / 2
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == STEPS // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
    return sae, float(loss)

def angle_single(sae, a_p, a_c):
    Dm = sae.dec.detach()
    cp, cc = Dm.T @ a_p, Dm.T @ a_c
    rho = torch.sqrt(cp ** 2 + cc ** 2)
    phi = torch.rad2deg(torch.atan2(cc, cp))
    lat = [(float(rho[i]), float(phi[i])) for i in range(M_LAT) if rho[i] > 0.5]
    cand = [(r, a) for r, a in lat if 20 < a < 120]
    phi_child = max(cand, key=lambda ra: ra[1])[1] if cand else float("nan")
    return phi_child, json.dumps(sorted(lat, key=lambda ra: -ra[0])[:4])

def classify_two(sae, a_p, c1, c2):
    Dm = sae.dec.detach()
    dirs = {"parent": a_p, "child1": c1, "child2": c2,
            "comp1": (a_p + c1) / math.sqrt(2), "comp2": (a_p + c2) / math.sqrt(2)}
    found = {}
    for name, v in dirs.items():
        found[name] = int((Dm.T @ v).max().item() > 0.9)
    label = ("faithful" if found["child1"] and found["child2"]
             else "absorbed" if found["comp1"] and found["comp2"]
             else "mixed")
    return label, json.dumps(found)

rows = []
t0 = time.time()
LAM = 0.2
BSTAR = LAM * Q * (8 - 4 * math.sqrt(2) - LAM) / 2

# ---- C1: coherence sweep, single child
grid = [(eps, mult) for eps in (0.0, 0.01, 0.02) for mult in (0.0, 0.5, 1.0, 2.0, 4.0)]
for i, ((eps, mult), seed) in enumerate(itertools.product(grid, (0, 1))):
    A_bg, (a_p, a_c) = make_features(100 + seed, 2)
    fn = lambda B, g: batch_single(B, A_bg, a_p, a_c, eps, g)
    sae, L = train(fn, LAM, beta=mult * BSTAR, seed=seed)
    phi, top = angle_single(sae, a_p, a_c)
    rows.append(dict(exp="C1", eps=eps, beta_mult=mult, seed=seed,
                     phi_child=phi, label="", detail=top, loss=L))
    print(f"[C1 {i+1}/{len(grid)*2}] eps={eps} beta={mult}b* seed={seed} "
          f"phi={phi:.1f} ({time.time()-t0:.0f}s)", flush=True)

# ---- C2: matryoshka, single child
for i, (eps, seed) in enumerate(itertools.product((0.0, 0.01), (0, 1, 2))):
    A_bg, (a_p, a_c) = make_features(100 + seed, 2)
    fn = lambda B, g: batch_single(B, A_bg, a_p, a_c, eps, g)
    sae, L = train(fn, LAM, matryoshka=True, seed=seed)
    phi, top = angle_single(sae, a_p, a_c)
    rows.append(dict(exp="C2", eps=eps, beta_mult=0, seed=seed,
                     phi_child=phi, label="", detail=top, loss=L))
    print(f"[C2 {i+1}/6] eps={eps} seed={seed} phi={phi:.1f}", flush=True)

# ---- C3: two-child, vanilla vs matryoshka
for i, (mode, eps, seed) in enumerate(itertools.product(
        ("vanilla", "matryoshka"), (0.0, 0.01), (0, 1, 2))):
    A_bg, (a_p, c1, c2) = make_features(200 + seed, 3)
    fn = lambda B, g: batch_two(B, A_bg, a_p, c1, c2, eps, g)
    sae, L = train(fn, LAM, matryoshka=(mode == "matryoshka"), seed=seed)
    label, det = classify_two(sae, a_p, c1, c2)
    rows.append(dict(exp="C3-" + mode, eps=eps, beta_mult=0, seed=seed,
                     phi_child=float("nan"), label=label, detail=det, loss=L))
    print(f"[C3 {i+1}/12] {mode} eps={eps} seed={seed} -> {label} {det}", flush=True)

with open("results_remedies.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"total {time.time()-t0:.0f}s", flush=True)
open("done2.flag", "w").write("ok\n")
