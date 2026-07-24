"""Absorbed-pair detection on a real trained SAE (round 11 analysis).

Applies the project's label-free pair detector (rounds 8-9: decoder-cosine band
+ two-sided co-firing lift + overlap veto) to a real SAE, scaled to large m by
filtering to the rate window first, then doing pairwise work only on survivors.
Reports the flagged-pair count + interpretable examples (top-activating tokens),
so an L1 SAE and a TopK SAE can be compared on absorbed-pair prevalence.

Env: SAE (path to sae_*.pt), ACTS (path to acts_*.pt), N_ANALYZE (default 100000
     tokens subsampled for firing stats), OUT (json). Detector constants inherited
     from round 8/9 (C_LO,C_HI,L_LO,L_HI,OVL_MAX,RATE_LO,RATE_HI,THETA).
"""
import os, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
SAE = os.environ["SAE"]; ACTS = os.environ["ACTS"]
N_ANALYZE = int(os.environ.get("N_ANALYZE", "50000"))
SEED = int(os.environ.get("SEED", "0"))          # SHARED across arches -> same tokens
OUT = os.environ.get("OUT", SAE.replace(".pt", "_pairs.json"))
# detector v1.1 constants (as locked in rounds 8/9). THETA=0.05 is the
# REGISTERED firing threshold (the first run wrongly used 0.0, which is
# architecture-asymmetric: L1's tiny ReLU outputs all "fire" while TopK zeros
# them -> not comparable; whole-repo review finding #1).
C_LO, C_HI, L_LO, L_HI, OVL_MAX = 0.45, 0.90, 0.5, 2.0, 0.9
RATE_LO, RATE_HI, THETA = 5e-4, 0.6, 0.05

def safe_load(p):
    try: return torch.load(p, weights_only=True)
    except Exception: return torch.load(p)

def main():
    s = safe_load(SAE); blob = safe_load(ACTS)
    W_enc = s["W_enc"].to(dev); b_enc = s["b_enc"].to(dev)
    W_dec = s["W_dec"].to(dev); b_dec = s["b_dec"].to(dev)
    mu = s["mu"].to(dev); scale = float(s["scale"]); arch = s["arch"]; k = int(s.get("k", 32))
    X = blob["acts"]; toks = blob["tokens"]
    N = X.shape[0]
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))[:min(N_ANALYZE, N)]
    Xa = ((X[idx].float().to(dev) - mu) * scale)
    toks_a = toks[idx]
    m = W_dec.shape[1]
    with torch.no_grad():
        f = torch.relu((Xa - b_dec) @ W_enc.T + b_enc)
        if arch == "topk":
            thr = f.topk(k, dim=-1).values[:, -1:].clamp_min(1e-12)
            f = f * (f >= thr).float()
        fire = (f > THETA).float()                  # [n, m]
        rates = fire.mean(0)                          # [m]
    keep = torch.where((rates > RATE_LO) & (rates < RATE_HI))[0]
    print(f"SAE {os.path.basename(SAE)}: m={m}, rate-window latents={len(keep)} "
          f"(fvu={s.get('stats',{}).get('fvu')})", flush=True)
    if len(keep) < 2:
        json.dump(dict(sae=os.path.basename(SAE), m=m, n_window=int(len(keep)),
                       n_cosine_band=0, n_flagged=0, examples=[]), open(OUT, "w"), indent=2)
        return
    # the [K,K] detector matrices are large (K can be >12k -> ~640MB each); do
    # the two matmuls on GPU, move to CPU, and do the boolean detector on CPU
    # (32GB host RAM) to avoid GPU OOM.
    fk = fire[:, keep]                                # [n, K] GPU
    fk_act = f[:, keep].cpu()                         # activations (CPU, for top_tokens)
    rk = rates[keep].cpu()
    Dk = W_dec[:, keep]                               # [d, K] GPU, unit-norm cols
    Csigned = (Dk.T @ Dk).cpu()                       # signed cos -> CPU
    C = Csigned.abs()                                 # |cos| for the band (rounds 8/9)
    n = fk.shape[0]
    Pj = (fk.T @ fk).cpu() / n                        # co-firing -> CPU
    del fire, f, fk, Dk
    if dev == "cuda": torch.cuda.empty_cache()
    L = Pj / torch.clamp(torch.outer(rk, rk), min=1e-12)
    O = Pj / torch.clamp(torch.minimum(rk[:, None], rk[None, :]), min=1e-12)
    iu = torch.triu(torch.ones_like(C, dtype=torch.bool), diagonal=1)
    band = (C > C_LO) & (C < C_HI)
    n_band = int((band & iu).sum())
    flagged = band & ((L <= L_LO) | (L >= L_HI)) & (O < OVL_MAX) & iu
    pairs = torch.nonzero(flagged)
    K = len(keep)
    n_possible = K * (K - 1) // 2                      # opportunity normalization
    n_neg = int((flagged & (Csigned < 0)).sum())       # negatively-aligned flagged pairs
    # cluster (connected-component) count of the flagged graph -> redundant clusters,
    # since a cluster of size r yields r(r-1)/2 pairs (quadratic over-count of counts)
    parent = list(range(K))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for pr in pairs.tolist():
        ra, rb = find(pr[0]), find(pr[1])
        if ra != rb: parent[ra] = rb
    involved = set()
    for pr in pairs.tolist(): involved.add(pr[0]); involved.add(pr[1])
    n_clusters = len({find(x) for x in involved})
    print(f"  window={K} band={n_band} flagged={len(pairs)} "
          f"flag_rate_per_Mpairs={1e6*len(pairs)/max(n_possible,1):.2f} "
          f"neg_aligned={n_neg} redundant_clusters={n_clusters}", flush=True)
    # interpretable examples: top-activating tokens per latent of the top few pairs
    examples = []
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(blob["model"])
    except Exception:
        tk = None
    def top_tokens(local_j, n=6):
        act = fk_act[:, local_j]                  # activation magnitude (CPU)
        top = torch.topk(act, min(n, act.shape[0])).indices
        ids = toks_a[top].tolist()
        return tk.convert_ids_to_tokens(ids) if tk else ids
    order = pairs[torch.argsort(C[flagged], descending=True)] if len(pairs) else pairs
    for p in order[:40]:                              # top 40 (was 15) - still a sample; disclosed
        a, b = int(p[0]), int(p[1])
        examples.append(dict(lat_a=int(keep[a]), lat_b=int(keep[b]),
                             cos_abs=round(float(C[a, b]), 3),
                             cos_signed=round(float(Csigned[a, b]), 3),
                             lift=round(float(L[a, b]), 3), overlap=round(float(O[a, b]), 3),
                             rate_a=round(float(rk[a]), 5), rate_b=round(float(rk[b]), 5),
                             toks_a=top_tokens(a), toks_b=top_tokens(b)))
    res = dict(sae=os.path.basename(SAE), arch=arch, m=m, theta=THETA, seed=SEED,
               n_analyze=int(n), n_window=int(K), n_possible_pairs=int(n_possible),
               n_cosine_band=n_band, n_flagged=int(len(pairs)),
               flagged_per_million_pairs=round(1e6*len(pairs)/max(n_possible, 1), 2),
               n_flagged_neg_aligned=n_neg, redundant_clusters=n_clusters,
               fvu=s.get("stats", {}).get("fvu"),
               note="EXPLORATORY; opportunity-normalize (flagged_per_million_pairs) and "
                    "compare clusters not raw pairs; examples are a top-40 sample, not all pairs",
               examples=examples)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}: flagged={len(pairs)} band={n_band} window={K} "
          f"per_Mpairs={res['flagged_per_million_pairs']} clusters={n_clusters}", flush=True)

if __name__ == "__main__":
    main()
