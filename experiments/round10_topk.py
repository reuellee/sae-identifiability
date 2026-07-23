"""Round 10: feature absorption under TopK vs L1 SAEs (activation-aware).

Theory: theory/topk_absorption.md (regression tests theory/verify_topk_absorption.py).
Prereg: notes/prereg-topk-absorption.md.

CHILD-RECOVERY is measured FUNCTIONALLY (not decoder geometry alone), closing
the dead-atom and abs-cosine loopholes: a run has child_recovered=1 iff some
latent has (i) SIGNED cos(D_j, v_c) >= COS_MIN, (ii) fires on child-solo
events at rate >= FIRE_MIN, and (iii) is child-selective
(fire|csolo - fire|psolo >= SEL_MIN). Binary, always defined -> no
outcome-dependent missingness. Geometric phi_child is a SECONDARY diagnostic.

Arms (all isolated pair, n_bg=0, so the pair's atom allocation = the theory's):
  A  m=2  TopK k in {1,2}: the exact two-atom theorem (P1 crossover, P2 collapse).
  B  m=16 TopK k=1: overcomplete SGD behaviour - does it find the zero-loss
     child-recovering 3-atom solution? (P3 escape).
  C  m=16 L1 (lam=0.2, no TopK): overcomplete L1 control (P4 head-to-head:
     TopK resists the absorption L1 suffers).

SMOKE=1 tiny CPU run. Weights saved per group. Outputs results/round10/r10_runs.csv.
"""
import torch, math, csv, time, os

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = bool(int(os.environ.get("SMOKE", "0")))
OUTDIR = os.path.join(HERE, "..", "results", "round10")
os.makedirs(OUTDIR, exist_ok=True)
BATCH = 2048
STEPS = 200 if SMOKE else 15000
N_EV = 4000 if SMOKE else 100_000
THETA = 0.05                          # a latent "fires" if its (post-topk) code > THETA
# frozen functional-recovery thresholds
COS_MIN, FIRE_MIN, SEL_MIN = 0.6, 0.5, 0.3
# composite diagnostic thresholds
COMP_COS_MIN, COMP_FIRE_JOINT_MIN = 0.9, 0.5

class SAE(torch.nn.Module):
    def __init__(self, R, d, m):
        super().__init__()
        bnd = 1.0 / math.sqrt(d)
        self.W = torch.nn.Parameter(torch.empty(R, m, d).uniform_(-bnd, bnd))
        self.b = torch.nn.Parameter(torch.zeros(R, m))
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

def features(d, seed):
    g = torch.Generator().manual_seed(100 + seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return Q[:, 0].contiguous().to(dev), Q[:, 1].contiguous().to(dev)   # v_p, v_c

def sample(runs, vp, vc, gen, B):
    R = len(runs)
    qv = torch.tensor([r["q"] for r in runs], device=dev).view(R, 1)
    pv = torch.tensor([r["p"] for r in runs], device=dev).view(R, 1)
    ev = torch.tensor([r["eps"] for r in runs], device=dev).view(R, 1)
    u = torch.rand(R, B, device=dev, generator=gen)
    joint = (u < qv).float()
    psolo = ((u >= qv) & (u < qv + pv)).float()
    csolo = ((u >= qv + pv) & (u < qv + pv + ev)).float()
    x = ((joint + psolo).unsqueeze(-1) * vp.unsqueeze(1)
         + (joint + csolo).unsqueeze(-1) * vc.unsqueeze(1))
    return x, joint, psolo, csolo

def train(sae, runs, vp, vc, steps, topk, lam, tag):
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(steps):
        x, *_ = sample(runs, vp, vc, gen, BATCH)
        xh, f = sae(x, topk=topk)
        rec = ((x - xh) ** 2).sum(-1).mean(-1)
        loss = (rec + lam * f.sum(-1).mean(-1)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        if t == steps // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 4000 == 0:
            print(f"  [{tag} {t}/{steps}] rec={float(rec.mean()):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return rec.detach()

def measure(sae, runs, vp, vc, topk):
    """Per-run functional child-recovery + composite/absorption + diagnostics."""
    R = len(runs)
    gen = torch.Generator(device=dev).manual_seed(9000)
    rows_extra = []
    with torch.no_grad():
        x, joint, psolo, csolo = sample(runs, vp, vc, gen, N_EV)
        _, f = sae(x, topk=topk)                       # [R, N_EV, m]
        fires = (f > THETA).float()
        D = sae.D                                       # [R, d, m]
        cos_vc = torch.einsum('rdm,rd->rm', D, vc)      # signed
        cos_vp = torch.einsum('rdm,rd->rm', D, vp)
        wcomp = (vp + vc) / math.sqrt(2)
        cos_comp = torch.einsum('rdm,rd->rm', D, wcomp)
        for i in range(R):
            cj = csolo[i] > 0.5; pj = psolo[i] > 0.5; jj = joint[i] > 0.5
            fr_c = fires[i][cj].mean(0) if cj.any() else torch.zeros(D.shape[2], device=dev)
            fr_p = fires[i][pj].mean(0) if pj.any() else torch.zeros(D.shape[2], device=dev)
            fr_j = fires[i][jj].mean(0) if jj.any() else torch.zeros(D.shape[2], device=dev)
            # functional child recovery
            childish = ((cos_vc[i] >= COS_MIN) & (fr_c >= FIRE_MIN)
                        & ((fr_c - fr_p) >= SEL_MIN))
            child_rec = int(childish.any())
            # best child-latent stats (for reporting)
            if child_rec:
                jbest = int(torch.where(childish, cos_vc[i], torch.full_like(cos_vc[i], -9)).argmax())
            else:
                jbest = int(cos_vc[i].argmax())
            # composite present & functional on joint
            comp_mask = (cos_comp[i] >= COMP_COS_MIN) & (fr_j >= COMP_FIRE_JOINT_MIN)
            comp_present = int(comp_mask.any())
            absorbed = int((not child_rec) and comp_present)
            # geometric phi (secondary): angle of the child-side in-plane latent
            rho = torch.sqrt(cos_vp[i] ** 2 + cos_vc[i] ** 2)
            phi = torch.rad2deg(torch.atan2(cos_vc[i], cos_vp[i]))
            inpl = [(float(rho[j]), float(phi[j])) for j in range(D.shape[2]) if rho[j] > 0.5]
            childside = [(r, a) for r, a in inpl if 20.0 < a < 120.0]
            phic = max(childside, key=lambda ra: ra[1])[1] if childside else float("nan")
            rows_extra.append(dict(
                child_recovered=child_rec, absorbed=absorbed, comp_present=comp_present,
                cos_child_signed=round(float(cos_vc[i][jbest]), 4),
                fire_child_csolo=round(float(fr_c[jbest]), 4),
                fire_child_psolo=round(float(fr_p[jbest]), 4),
                comp_fire_joint=round(float(fr_j[cos_comp[i].argmax()]), 4),
                cos_comp_best=round(float(cos_comp[i].max()), 4),
                phi_child=round(phic, 2) if phic == phic else float("nan")))
    return rows_extra

def run_group(runs, tag, topk, lam, d, m, rows):
    feats = [features(d, r["seed"]) for r in runs]
    vp = torch.stack([f[0] for f in feats]); vc = torch.stack([f[1] for f in feats])
    torch.manual_seed(11)
    sae = SAE(len(runs), d, m).to(dev)
    rec = train(sae, runs, vp, vc, STEPS, topk, lam, tag)
    torch.save(dict(D=sae.D.detach().cpu(), W=sae.W.detach().cpu(),
                    b=sae.b.detach().cpu(), runs=runs),
               os.path.join(OUTDIR, f"weights_r10_{tag}.pt"))
    extra = measure(sae, runs, vp, vc, topk)
    for i, r in enumerate(runs):
        row = dict(r); row.update(extra[i]); row["rec"] = round(float(rec[i]), 4)
        row["eps_star_topk"] = round(2 * r["q"], 3)
        rows.append(row)
    del sae
    if dev == "cuda": torch.cuda.empty_cache()
    print(f"[{tag}] {len(runs)} runs done", flush=True)
    write_csv(rows)

def write_csv(rows):
    if not rows: return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(os.path.join(OUTDIR, "r10_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

EPS_A = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
EPS_OC = [0.05, 0.10, 0.20, 0.40]
SEEDS = range(2 if SMOKE else 24)

if __name__ == "__main__":
    t0 = time.time(); rows = []
    # Arm A: exact two-atom, m=2, TopK k in {1,2}
    for k in (1, 2):
        for q in ([0.2] if SMOKE else [0.1, 0.2]):
            runs = [dict(arm="A", m=2, k=k, lam=0.0, q=q, p=q, eps=e, seed=s)
                    for e in (EPS_A[:3] if SMOKE else EPS_A) for s in SEEDS]
            run_group(runs, f"A_k{k}_q{int(q*100)}", topk=k, lam=0.0, d=8, m=2, rows=rows)
    # Arm B: overcomplete TopK k=1
    runs = [dict(arm="B", m=16, k=1, lam=0.0, q=0.2, p=0.2, eps=e, seed=s)
            for e in (EPS_OC[:2] if SMOKE else EPS_OC) for s in SEEDS]
    run_group(runs, "B_k1", topk=1, lam=0.0, d=16, m=16, rows=rows)
    # Arm C: overcomplete L1 control (no TopK)
    runs = [dict(arm="C", m=16, k=0, lam=0.2, q=0.2, p=0.2, eps=e, seed=s)
            for e in (EPS_OC[:2] if SMOKE else EPS_OC) for s in SEEDS]
    run_group(runs, "C_l1", topk=None, lam=0.2, d=16, m=16, rows=rows)
    write_csv(rows)
    print(f"wrote r10_runs.csv ({len(rows)} rows) total {time.time()-t0:.0f}s", flush=True)
