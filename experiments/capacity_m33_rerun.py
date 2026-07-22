"""Capacity rerun forced by the external review (R5): the round-1 grid at
m = 32 could not represent the redundant triple (30 bg + parent + child +
composite = 33 columns), so the withdrawn "dynamics gap" claim (SGD avoids the
triple even without scarcity) was untestable in that architecture.

PRE-REGISTERED PREDICTIONS (fixed before running; from §14's m=1536 evidence):
  K1: at m >= 33 (here 34 and 40) with eps > 0, the triple appears in a
      majority of runs — a separate ~90 deg child latent COEXISTS with a
      ~45 deg composite latent (round-1 criterion: child angle >= 78 deg,
      rho > 0.8, alongside composite 40-55 deg, rho > 0.8).
  K2: at m = 32 (control), triples remain absent (architecturally blocked).
  K3: the functional child transition (phi_func crossing 67.5 deg) at m >= 34
      shifts EARLIER (smaller eps) than at m = 32, since the triple lets the
      child direction win a latent without evicting the composite. Exploratory:
      quantify the shift; if the transition disappears entirely (child latent
      at all eps > 0), absorption at spare capacity is purely an optimization
      phenomenon after all — either outcome replaces the withdrawn claim.

Design: trainer/generative model = round-2/Arm A verbatim (imported).
(lam, q) = (0.2, 0.2) [eps* = 0.0486] and (0.1, 0.2) [eps* = 0.0238];
eps/eps* in {0, 0.25, 0.5, 0.75, 1.0, 1.5}; m in {32, 34, 40}; 8 seeds.
= 2 * 6 * 3 * 8 = 288 runs, one batched program per m.

Event model: round-2's u-partition (q joint, P0=0.2 parent-solo, eps child-
solo) — NOT Arm A's conditional model — to match the round-1 grid this rerun
re-adjudicates. Outputs results/capacity_m33/m33_runs.csv.
"""
import torch, math, csv, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_bimodality_armA as A
from prereg_bimodality_armA import BSAE, train_batched, make_feature_sets

dev = A.dev
SMOKE = A.SMOKE
RT2 = math.sqrt(2)
P0 = 0.2
SEEDS = 2 if SMOKE else 8
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "results", "capacity_m33")

def eps_star(lam, q):
    return lam * q * (8 - 4 * RT2 - lam) / (2 * (1 - (2 - RT2) * lam))

CELLS = [(0.2, 0.2), (0.1, 0.2)]
EPS_MULT = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5] if not SMOKE else [0.0, 1.0]
MS = (32, 34, 40) if not SMOKE else (32, 34)

def sample_cell(Bs, gen, Abg, ap, ac, q_vec, eps_vec):
    """Round-2 u-partition event model, vectorized over runs."""
    Rn = Abg.shape[0]
    coeff = torch.empty(Rn, Bs, A.N_BG, device=dev).uniform_(0.8, 1.2, generator=gen)
    mask = (torch.rand(Rn, Bs, A.N_BG, device=dev, generator=gen) < A.BG_RATE).float()
    x = torch.einsum('rbn,rdn->rbd', coeff * mask, Abg)
    u = torch.rand(Rn, Bs, device=dev, generator=gen)
    q = q_vec.view(-1, 1); e = eps_vec.view(-1, 1)
    joint = (u < q).float()
    psolo = ((u >= q) & (u < q + P0)).float()
    csolo = ((u >= q + P0) & (u < q + P0 + e)).float()
    return (x + (joint + psolo).unsqueeze(-1) * ap.unsqueeze(1)
              + (joint + csolo).unsqueeze(-1) * ac.unsqueeze(1))

if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for m_lat in MS:
        runs = [dict(lam=lam, q=q, em=em, seed=s)
                for (lam, q) in CELLS for em in EPS_MULT for s in range(SEEDS)]
        Rn = len(runs)
        A_bg_s, spec_s = make_feature_sets([100 + r["seed"] for r in runs], 2)
        ap_s, ac_s = spec_s
        q_vec = torch.tensor([r["q"] for r in runs], device=dev)
        eps_vec = torch.tensor([r["em"] * eps_star(r["lam"], r["q"])
                                for r in runs], device=dev)
        lam_vec = torch.tensor([r["lam"] for r in runs], device=dev)
        torch.manual_seed(7)
        sae = BSAE(Rn, A.D_AMB, m_lat).to(dev)
        fn = lambda Bs, g: sample_cell(Bs, g, A_bg_s, ap_s, ac_s, q_vec, eps_vec)
        rec = train_batched(sae, fn, lam_vec, A.STEPS, tag=f"m33 m={m_lat} R={Rn}")
        # ---- analysis: round-1 triple criterion + round-2 functional metric
        Dm = sae.D.detach()
        cp = torch.einsum('rdm,rd->rm', Dm, ap_s)
        cc = torch.einsum('rdm,rd->rm', Dm, ac_s)
        rho = (cp ** 2 + cc ** 2).sqrt()
        phi = torch.rad2deg(torch.atan2(cc, cp))
        fch = torch.relu(torch.einsum('rmd,rd->rm', sae.W.detach(), ac_s)
                         + sae.b.detach())
        act, idx = fch.max(dim=1)
        phi_func = phi.gather(1, idx.unsqueeze(1)).squeeze(1)
        has_child = ((phi >= 78) & (phi <= 102) & (rho > 0.8)).any(dim=1)
        has_comp = ((phi > 40) & (phi < 55) & (rho > 0.8)).any(dim=1)
        has_parent = ((phi.abs() < 20) & (rho > 0.8)).any(dim=1)
        triple = has_child & has_comp & has_parent
        for i, r in enumerate(runs):
            es = eps_star(r["lam"], r["q"])
            rows.append(dict(m_lat=m_lat, lam=r["lam"], q=r["q"],
                             eps=round(r["em"] * es, 5), eps_mult=r["em"],
                             eps_star=round(es, 5), seed=r["seed"],
                             phi_func=round(float(phi_func[i]), 2),
                             has_child=int(has_child[i]),
                             has_comp=int(has_comp[i]),
                             has_parent=int(has_parent[i]),
                             triple=int(triple[i]),
                             rec=round(float(rec[i]), 4)))
        n_tr = int(triple.sum())
        print(f"[m={m_lat}] triples {n_tr}/{Rn}", flush=True)
    out = os.path.join(OUTDIR, "m33_runs.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows) total {time.time()-t0:.0f}s", flush=True)
