"""S1 of notes/prereg-containment-orientation.md: non-regression check,
free/CPU-only, on FROZEN round-8 E1 weights (results/round8/
weights_r8e1_m{128,256}.pt, eps=0.002, real GPT-2 layer-6 activations).
Recomputes rarity-rule vs containment-rule orientation head-to-head on the
same fires array per run; no new training.
"""
import torch, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THETA, MARGIN, MARGIN_MAG, MIN_N = 0.05, 0.02, 0.3, 30
D_MODEL, Q, P0, AMP, N_EV = 768, 0.2, 0.2, 5.0, 100_000

bg = torch.load(os.path.join(HERE, "..", "experiments", "activations_l6.pt")).float()
mu = bg.mean(0, keepdim=True)
bg = ((bg - mu) * math.sqrt(D_MODEL) / (bg - mu).norm(dim=1).mean())
N = bg.shape[0]

def orient(fires, act, i, j):
    """Amendment 1: magnitude cross-delta primary, firing-containment
    fallback, rarity final fallback. Returns (rarity, containment, amended)."""
    fi, fj = fires[:, i], fires[:, j]
    ai, aj = act[:, i], act[:, j]
    ni, nj = fi.sum(), fj.sum()
    cij = float((fi & fj).sum() / nj) if nj > 0 else float("nan")   # P(i | j)
    cji = float((fi & fj).sum() / ni) if ni > 0 else float("nan")   # P(j | i)
    if cji > cij + MARGIN:
        cont = i
    elif cij > cji + MARGIN:
        cont = j
    else:
        cont = None   # indeterminate
    rarity = i if ni < nj else j

    i_off, i_on = ai[fi & ~fj], ai[fi & fj]
    j_off, j_on = aj[fj & ~fi], aj[fj & fi]
    if min(i_off.size, i_on.size, j_off.size, j_on.size) >= MIN_N:
        delta_i = float(i_off.mean() - i_on.mean())
        delta_j = float(j_off.mean() - j_on.mean())
        if delta_i > delta_j + MARGIN_MAG:
            amended = i
        elif delta_j > delta_i + MARGIN_MAG:
            amended = j
        else:
            amended = cont   # fall back to containment
    else:
        amended = cont
    if amended is None:
        amended = rarity   # final fallback
    return rarity, cont, amended, cij, cji

rng = np.random.default_rng(0)
for M in (128, 256):
    ck = torch.load(os.path.join(HERE, "..", "results", "round8",
                    f"weights_r8e1_m{M}.pt"), weights_only=False)
    W, b, D, runs = ck["W"], ck["b"], ck["D"], ck["runs"]
    n_flagged = n_rarity_ok = n_cont_ok = n_indet = n_amended_ok = 0
    for i, (e, s) in enumerate(runs):
        if e != 0.002:
            continue
        g = torch.Generator().manual_seed(300 + s)
        Qm, _ = torch.linalg.qr(torch.randn(D_MODEL, 2, generator=g))
        ap_, ac_ = Qm.T[0], Qm.T[1]
        comp_dir = (ap_ + ac_) / math.sqrt(2)
        Di = D[i]
        par = int(torch.einsum('dm,d->m', Di, ap_).abs().argmax())
        comp = int(torch.einsum('dm,d->m', Di, comp_dir).abs().argmax())
        cos_comp = float(torch.einsum('dm,d->m', Di, comp_dir).abs().max())
        if cos_comp <= 0.98:
            continue   # not absorbed-formed, same filter as round-8 E1
        g2 = torch.Generator().manual_seed(9000 + s)
        fire_chunks, act_chunks = [], []
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
            fire_chunks.append((f > THETA).numpy())
            act_chunks.append(f.numpy())
        fires = np.concatenate(fire_chunks)
        act = np.concatenate(act_chunks)
        tp = tuple(sorted((par, comp)))
        rarity_ch, cont_ch, amended_ch, cij, cji = orient(fires, act, *tp)
        n_flagged += 1
        n_rarity_ok += int(rarity_ch == comp)
        n_amended_ok += int(amended_ch == comp)
        if cont_ch is None:
            n_indet += 1
        else:
            n_cont_ok += int(cont_ch == comp)
        if i % 16 == 15: print(f"[m={M}] {i+1}/{len(runs)}", flush=True)
    n_det = n_flagged - n_indet
    print(f"m={M}: absorbed-formed n={n_flagged} | rarity acc "
          f"{n_rarity_ok/n_flagged:.3f} | containment acc (of determinate) "
          f"{n_cont_ok/n_det if n_det else float('nan'):.3f} "
          f"(indeterminate {n_indet}/{n_flagged}) | amended (mag+cont+rarity "
          f"fallback) acc {n_amended_ok/n_flagged:.3f}", flush=True)
