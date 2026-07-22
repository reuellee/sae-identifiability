"""Amendment-2 recompute from FROZEN round-8 E1 artifacts (weights + seeds):
(1) three specificity fields the registered concept required — exact oracle-
pair flag, ANY flag touching an oracle latent, full-scan flags/SAE;
(2) shuffled-firing dependence null: permute each latent's firing column
independently (preserves decoder geometry + marginal rates, destroys pair
dependence), re-run detector v1.2, count flags.
N_EV=100k (vs 200k in the run) — flag-level metrics; disclosed.
"""
import torch, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_LO, C_HI, L_LO, L_HI, OVL_MAX = 0.45, 0.90, 0.5, 1.9, 0.9
THETA, RATE_LO, RATE_HI = 0.05, 5e-4, 0.6
D_MODEL, Q, P0, AMP, N_EV = 768, 0.2, 0.2, 5.0, 100_000

bg = torch.load(os.path.join(HERE, "..", "experiments", "activations_l6.pt")).float()
mu = bg.mean(0, keepdim=True)
bg = ((bg - mu) * math.sqrt(D_MODEL) / (bg - mu).norm(dim=1).mean())
N = bg.shape[0]

def detect(Dn, fires):
    rates = fires.mean(0)
    keep = (rates > RATE_LO) & (rates < RATE_HI)
    C = np.abs(Dn.T @ Dn)
    F32 = fires.astype(np.float32)
    Pj = (F32.T @ F32) / len(fires)
    L = Pj / np.maximum(np.outer(rates, rates), 1e-12)
    O = Pj / np.maximum(np.minimum.outer(rates, rates), 1e-12)
    flags = []
    kidx = np.where(keep)[0]
    for a in range(len(kidx)):
        for b_ in range(a + 1, len(kidx)):
            i, j = kidx[a], kidx[b_]
            if (C_LO < C[i, j] < C_HI and (L[i, j] <= L_LO or L[i, j] >= L_HI)
                    and O[i, j] < OVL_MAX):
                flags.append((int(i), int(j)))
    return flags

rng = np.random.default_rng(0)
for M in (128, 256):
    ck = torch.load(os.path.join(HERE, "..", "results", "round8",
                    f"weights_r8e1_m{M}.pt"), weights_only=False)
    W, b, D, runs = ck["W"], ck["b"], ck["D"], ck["runs"]
    res = {0.002: dict(exact=[], touch=[], total=[], shuf=[]),
           0.05: dict(exact=[], touch=[], total=[], shuf=[])}
    for i, (e, s) in enumerate(runs):
        g = torch.Generator().manual_seed(300 + s)
        Qm, _ = torch.linalg.qr(torch.randn(D_MODEL, 2, generator=g))
        ap_, ac_ = Qm.T[0], Qm.T[1]
        comp_dir = (ap_ + ac_) / math.sqrt(2)
        Di = D[i]
        par = int(torch.einsum('dm,d->m', Di, ap_).abs().argmax())
        comp = int(torch.einsum('dm,d->m', Di, comp_dir).abs().argmax())
        g2 = torch.Generator().manual_seed(9000 + s)
        chunks = []
        for c0 in range(0, N_EV, 25_000):
            n = min(25_000, N_EV - c0)
            idx = torch.randint(0, N, (n,), generator=g2)
            x = bg[idx]
            u = torch.rand(n, generator=g2)
            jj = (u < Q).float(); pp = ((u >= Q) & (u < Q + P0)).float()
            cs = ((u >= Q + P0) & (u < Q + P0 + e)).float()
            x = (x + AMP * (jj + pp).unsqueeze(-1) * ap_
                   + AMP * (jj + cs).unsqueeze(-1) * ac_)
            f = torch.relu(x @ W[i].T + b[i])
            chunks.append((f > THETA).numpy())
        fires = np.concatenate(chunks)
        flags = detect(Di.numpy(), fires)
        tp = tuple(sorted((par, comp)))
        touch = [p for p in flags if par in p or comp in p]
        res[e]["exact"].append(int(tp in flags))
        res[e]["touch"].append(len(touch))
        res[e]["total"].append(len(flags))
        # shuffled-firing dependence null
        fsh = fires.copy()
        for col in range(fsh.shape[1]):
            rng.shuffle(fsh[:, col])
        res[e]["shuf"].append(len(detect(Di.numpy(), fsh)))
        if i % 16 == 15: print(f"[m={M}] {i+1}/{len(runs)}", flush=True)
    for e in (0.002, 0.05):
        r = res[e]
        print(f"m={M} eps={e}: exact-pair flag {np.mean(r['exact']):.3f} | "
              f"oracle-TOUCH flags/SAE {np.mean(r['touch']):.2f} | "
              f"full-scan flags/SAE {np.mean(r['total']):.2f} | "
              f"SHUFFLED-null flags/SAE {np.mean(r['shuf']):.2f}", flush=True)
