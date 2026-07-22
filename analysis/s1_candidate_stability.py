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
        # amendment-2 exclusion: drop every pair TOUCHING a planted latent
        # (parent/background and composite/background pairs are injection-
        # affected; the previous exact-pair-only filter was insufficient)
        cand[s] = [(i_, j_, Di[:, i_].numpy(), Di[:, j_].numpy())
                   for (i_, j_) in flags
                   if i_ not in (par, comp) and j_ not in (par, comp)]
        print(f"[m={M} seed {s}] candidates (non-planted): {len(cand[s])}",
              flush=True)
    # cross-seed matching (corrected per round-8 review amendment §5):
    # bijective pair score over BOTH one-to-one assignments; clustering over
    # all candidates from all seeds (no reference-seed anchoring); at most one
    # candidate per seed per cluster.
    def pair_score(a, b):
        (di, dj), (ek, el) = a, b
        s1 = min(abs(np.dot(di, ek)), abs(np.dot(dj, el)))
        s2 = min(abs(np.dot(di, el)), abs(np.dot(dj, ek)))
        return max(s1, s2)

    nodes = [(s, i_, j_, di, dj) for s in sorted(cand)
             for (i_, j_, di, dj) in cand[s]]
    n = len(nodes)
    # union-find over match edges
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in range(n):
        for b_ in range(a + 1, n):
            if nodes[a][0] == nodes[b_][0]:
                continue                      # same seed never merges directly
            if pair_score((nodes[a][3], nodes[a][4]),
                          (nodes[b_][3], nodes[b_][4])) > 0.9:
                ra, rb = find(a), find(b_)
                if ra != rb: parent[ra] = rb
    clusters = {}
    for a in range(n):
        clusters.setdefault(find(a), []).append(nodes[a])
    stable = []
    for members in clusters.values():
        seeds_in = {}
        for (s, i_, j_, di, dj) in members:   # <=1 candidate per seed: keep first
            seeds_in.setdefault(s, (i_, j_))
        if len(seeds_in) >= 4:
            stable.append((sorted(seeds_in), members[0][1], members[0][2]))
    print(f"m={M}: seed-stable (>=4/8, clustered) candidate clusters: {len(stable)}"
          f"  (total candidates {n})")
    for (sds, i0, j0) in stable[:10]:
        print(f"   cluster incl. pair ({i0},{j0}), seeds {sds}")
    # machine-readable cluster file (amendment-2 requirement)
    import json
    out = []
    for members in clusters.values():
        if len({m_[0] for m_ in members}) < 2: continue
        out.append([{"seed": int(m_[0]), "i": int(m_[1]), "j": int(m_[2])}
                    for m_ in members])
    json.dump(out, open(os.path.join(HERE, "..", "results", "round8",
              f"s1_clusters_m{M}.json"), "w"), indent=1)
