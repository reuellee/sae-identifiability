"""S1 (exploratory, CPU): are Arm 2's flagged real-feature pairs seed-stable?

From weights_arm2_m{128,256}.pt: re-run detector v1.1 per SAE on a fresh eval
stream, collect flagged pairs EXCLUDING the planted (parent, composite) pair,
then match pairs across the 8 seed-SAEs of each width by decoder cosine
(both members > 0.9 to a counterpart). Seed-stable (>=4/8) pairs form the
shortlist for the queued natural-feature evaluation.

Note: all 8 seed-SAEs of a width share the SAME background activations (seeds
differ in injected-pair direction + init), so real-feature latents are
comparable across seeds. Run alongside the round-8 GPU session; CPU-only.
"""
import torch, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "prereg_pairid")

C_LO, C_HI = 0.45, 0.90
L_LO, L_HI = 0.5, 2.0
OVL_MAX = 0.9
THETA = 0.05
RATE_LO, RATE_HI = 5e-4, 0.6
N_EV = 100_000
D_MODEL, Q, P0, AMP = 768, 0.2, 0.2, 5.0

bgp = os.path.join(HERE, "..", "experiments", "activations_l6.pt")
if not os.path.exists(bgp):
    print("activations_l6.pt not present locally; S1 needs it (780MB). "
          "Fetch from dev-gpu or rerun after round 8 collection.")
    sys.exit(0)
bg = torch.load(bgp).float()
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

for M in (128, 256):
    ck = torch.load(os.path.join(RES, f"weights_arm2_m{M}.pt"),
                    weights_only=False)
    W, b, D, runs = ck["W"], ck["b"], ck["D"], ck["runs"]
    pairs = {}
    for s in range(8):
        g = torch.Generator().manual_seed(300 + s)
        Qm, _ = torch.linalg.qr(torch.randn(D_MODEL, 2, generator=g))
        pairs[s] = Qm.T
    cand = {}          # seed -> list of (i, j, Dcols)
    for i, (e, s) in enumerate(runs):
        if e != 0.002: continue          # absorbed condition SAEs only
        ap_, ac_ = pairs[s][0], pairs[s][1]
        comp_dir = (ap_ + ac_) / math.sqrt(2)
        Di = D[i]
        par = int(torch.einsum('dm,d->m', Di, ap_).abs().argmax())
        comp = int(torch.einsum('dm,d->m', Di, comp_dir).abs().argmax())
        g2 = torch.Generator().manual_seed(7700 + s)
        fires_chunks = []
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
            fires_chunks.append((f > THETA).numpy())
        fires = np.concatenate(fires_chunks)
        flags = detect(Di.numpy(), fires)
        tp = tuple(sorted((par, comp)))
        cand[s] = [(i_, j_, Di[:, i_].numpy(), Di[:, j_].numpy())
                   for (i_, j_) in flags if (i_, j_) != tp]
        print(f"[m={M} seed {s}] candidates (non-planted): {len(cand[s])}",
              flush=True)
    # cross-seed matching
    seeds = sorted(cand)
    stable = []
    used = {s: set() for s in seeds}
    base_s = seeds[0]
    for (i0, j0, di0, dj0) in cand[base_s]:
        hits = 1
        for s2 in seeds[1:]:
            found = False
            for (i2, j2, di2, dj2) in cand[s2]:
                m1 = max(abs(np.dot(di0, di2)), abs(np.dot(di0, dj2)))
                m2 = max(abs(np.dot(dj0, di2)), abs(np.dot(dj0, dj2)))
                if m1 > 0.9 and m2 > 0.9:
                    found = True; break
            hits += int(found)
        if hits >= 4:
            stable.append((i0, j0, hits))
    print(f"m={M}: seed-stable (>=4/8) candidate pairs: {len(stable)}")
    for (i0, j0, h) in stable[:10]:
        print(f"   base pair ({i0},{j0}) stable in {h}/8 seeds")
