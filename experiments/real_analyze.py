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
import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
SAE = os.environ["SAE"]; ACTS = os.environ["ACTS"]
N_ANALYZE = int(os.environ.get("N_ANALYZE", "100000"))
OUT = os.environ.get("OUT", SAE.replace(".pt", "_pairs.json"))
# detector v1.1 constants (as locked in rounds 8/9)
C_LO, C_HI, L_LO, L_HI, OVL_MAX = 0.45, 0.90, 0.5, 2.0, 0.9
RATE_LO, RATE_HI, THETA = 5e-4, 0.6, 0.0

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
    idx = torch.randperm(N)[:min(N_ANALYZE, N)]
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
    fk = fire[:, keep]                                # [n, K]
    rk = rates[keep]
    Dk = W_dec[:, keep]                               # [d, K], unit-norm cols
    C = (Dk.T @ Dk).abs()                             # [K,K] |cos|
    Pj = (fk.T @ fk) / fk.shape[0]                    # co-firing
    L = Pj / torch.clamp(torch.outer(rk, rk), min=1e-12)
    O = Pj / torch.clamp(torch.minimum(rk[:, None], rk[None, :]), min=1e-12)
    band = (C > C_LO) & (C < C_HI)
    lift = (L <= L_LO) | (L >= L_HI)
    veto = O < OVL_MAX
    iu = torch.triu(torch.ones_like(C, dtype=torch.bool), diagonal=1)
    flagged = band & lift & veto & iu
    n_band = int((band & iu).sum())
    pairs = torch.nonzero(flagged)
    print(f"  cosine-band pairs={n_band}, flagged (band+lift+veto)={len(pairs)}", flush=True)
    # interpretable examples: top-activating tokens per latent of the top few pairs
    examples = []
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(blob["model"])
    except Exception:
        tk = None
    def top_tokens(local_j, n=6):
        act = f[:, keep[local_j]]                 # activation magnitude
        top = torch.topk(act, min(n, act.shape[0])).indices
        ids = toks_a[top.cpu()].tolist()
        return tk.convert_ids_to_tokens(ids) if tk else ids
    order = pairs[torch.argsort(C[flagged], descending=True)] if len(pairs) else pairs
    for p in order[:15]:
        a, b = int(p[0]), int(p[1])
        examples.append(dict(lat_a=int(keep[a]), lat_b=int(keep[b]),
                             cos=round(float(C[a, b]), 3), lift=round(float(L[a, b]), 3),
                             overlap=round(float(O[a, b]), 3),
                             rate_a=round(float(rk[a]), 5), rate_b=round(float(rk[b]), 5),
                             toks_a=top_tokens(a), toks_b=top_tokens(b)))
    res = dict(sae=os.path.basename(SAE), arch=arch, m=m, n_window=int(len(keep)),
               n_cosine_band=n_band, n_flagged=int(len(pairs)),
               fvu=s.get("stats", {}).get("fvu"), examples=examples)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}: n_flagged={len(pairs)} / cosine_band={n_band} / window={len(keep)}", flush=True)

if __name__ == "__main__":
    main()
