"""Natural-feature adjudication of S1 seed-stable candidates (CPU).

Locked by notes/prereg-natfeat-adjudication.md. Characterizes each seed-stable
S1 candidate cluster on PURE background GPT-2 layer-6 activations (no injected
pair) and applies the pre-registered A/B/C classification rule.

Inputs (all local):
  experiments/activations_l6.pt            (500k x 768 fp16)
  experiments/token_data.json              ({"ids":[...], "strings":[...]}, 500k)
  results/prereg_pairid/weights_arm2_m{128,256}.pt
  results/round8/s1_clusters_m{128,256}.json

Outputs:
  results/round8/natfeat_adjudication.json   (machine-readable, full evidence)
  results/round8/natfeat_report.md           (human-readable per-cluster tables)
"""
import torch, math, json, os
import numpy as np
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES_PAIRID = os.path.join(ROOT, "results", "prereg_pairid")
RES_R8 = os.path.join(ROOT, "results", "round8")

THETA = 0.05           # detector firing threshold (locked)
D_MODEL = 768
TOPK_POS = 50          # top-activating positions to show
TOPK_STR = 200         # window for dominant token-string tally
CTX = 1                # left-context tokens for readability

# ---- load token strings (exact HF alignment, regenerated on ephemeral VM) ----
td = json.load(open(os.path.join(ROOT, "experiments", "token_data.json")))
STRINGS = td["strings"]
assert len(STRINGS) == 500_000, len(STRINGS)

# ---- load + normalize activations identically to training ----
bg = torch.load(os.path.join(ROOT, "experiments", "activations_l6.pt")).float()
mu = bg.mean(0, keepdim=True)
bg = (bg - mu) * math.sqrt(D_MODEL) / (bg - mu).norm(dim=1).mean()
N = bg.shape[0]
assert N == len(STRINGS), (N, len(STRINGS))
print(f"bg {tuple(bg.shape)}  tokens {len(STRINGS)}", flush=True)


def firing(W_row, b_row):
    """f = ReLU(W x + b) over all tokens, chunked. Returns float32 (N,)."""
    out = np.empty(N, dtype=np.float32)
    for c0 in range(0, N, 65536):
        xb = bg[c0:c0 + 65536]
        fb = torch.relu(xb @ W_row + b_row)
        out[c0:c0 + xb.shape[0]] = fb.numpy()
    return out


def top_tokens(scores, k=TOPK_POS):
    idx = np.argsort(-scores)[:k]
    rows = []
    for p in idx:
        left = "".join(STRINGS[max(0, p - CTX):p])
        rows.append({"pos": int(p), "act": round(float(scores[p]), 3),
                     "tok": STRINGS[p], "ctx": (left + "|" + STRINGS[p])})
    return rows


def dominant_strings(scores, k=TOPK_STR):
    idx = np.argsort(-scores)[:k]
    c = Counter(STRINGS[p].strip().lower() for p in idx)
    return c.most_common(12)


def load_sae(M):
    ck = torch.load(os.path.join(RES_PAIRID, f"weights_arm2_m{M}.pt"),
                    weights_only=False)
    return ck["W"], ck["b"], ck["D"], ck["runs"]


def run_index_for_seed(runs, seed):
    """Row index of the absorbed-condition (eps=0.002) SAE for this seed."""
    for i, (e, s) in enumerate(runs):
        if s == seed and abs(e - 0.002) < 1e-9:
            return i
    raise KeyError(seed)


results = {"prereg": "notes/prereg-natfeat-adjudication.md", "theta": THETA,
           "widths": {}}

for M in (128, 256):
    W, b, D, runs = load_sae(M)
    clusters = json.load(open(os.path.join(RES_R8, f"s1_clusters_m{M}.json")))
    # keep only seed-stable clusters (>=4 distinct seeds) — the prereg shortlist
    stable = [c for c in clusters if len({m["seed"] for m in c}) >= 4]
    print(f"\n=== m={M}: {len(stable)} seed-stable clusters "
          f"(of {len(clusters)} raw) ===", flush=True)

    # firing cache per (seed, latent) to avoid recompute
    fcache = {}

    def get_fire(seed, lat):
        key = (seed, lat)
        if key not in fcache:
            ri = run_index_for_seed(runs, seed)
            fcache[key] = firing(W[ri][lat], b[ri][lat])
        return fcache[key]

    cluster_out = []
    for cid, members in enumerate(stable):
        seeds = sorted({m["seed"] for m in members})
        rep_seed = seeds[0]                       # locked: min seed index
        rep = next(m for m in members if m["seed"] == rep_seed)
        i, j = rep["i"], rep["j"]
        fi, fj = get_fire(rep_seed, i), get_fire(rep_seed, j)
        bi, bj = fi > THETA, fj > THETA
        ri_, rj_ = float(bi.mean()), float(bj.mean())
        # parent = higher rate, child = lower rate
        if ri_ >= rj_:
            (pi, pf, pr), (ci, cf, cr) = (i, fi, ri_), (j, fj, rj_)
            bp, bc = bi, bj
        else:
            (pi, pf, pr), (ci, cf, cr) = (j, fj, rj_), (i, fi, ri_)
            bp, bc = bj, bi
        both = int((bp & bc).sum())
        C_par_given_child = both / max(int(bc.sum()), 1)
        C_child_given_par = both / max(int(bp.sum()), 1)
        ri_sae = run_index_for_seed(runs, rep_seed)
        Dp = D[ri_sae][:, pi]
        Dc = D[ri_sae][:, ci]
        cos_ij = float((Dp * Dc).sum() / (Dp.norm() * Dc.norm()))
        # residual direction: child minus parent-projection
        r = Dc - (Dc * Dp).sum() / (Dp * Dp).sum() * Dp
        r = r / r.norm().clamp_min(1e-8)
        r_scores = (bg @ r).numpy()

        # containment-based provisional label (quantitative primary)
        asym = (C_par_given_child >= 0.80) and (C_child_given_par < 0.80)
        entry = {
            "cluster_id": cid, "n_seeds": len(seeds), "seeds": seeds,
            "rep_seed": rep_seed,
            "parent_latent": pi, "child_latent": ci,
            "rate_parent": round(pr, 5), "rate_child": round(cr, 5),
            "cos_decoder": round(cos_ij, 4),
            "C_parent_given_child": round(C_par_given_child, 4),
            "C_child_given_parent": round(C_child_given_par, 4),
            "asymmetric_nesting": bool(asym),
            "parent_top_strings": dominant_strings(pf),
            "child_top_strings": dominant_strings(cf),
            "residual_top_strings": dominant_strings(np.clip(r_scores, 0, None)),
            "parent_top_tokens": top_tokens(pf),
            "child_top_tokens": top_tokens(cf),
            "residual_top_tokens": top_tokens(r_scores),
        }
        cluster_out.append(entry)
        print(f"[m={M} c{cid}] seeds={len(seeds)} P=lat{pi}(r={pr:.4f}) "
              f"C=lat{ci}(r={cr:.4f}) cos={cos_ij:.3f} "
              f"C(P|c)={C_par_given_child:.3f} C(c|P)={C_child_given_par:.3f} "
              f"asym={asym}", flush=True)

    # 4-latent clique {51,54,107,172} at m=256: full containment matrix
    clique_out = None
    if M == 256:
        clique = [51, 54, 107, 172]
        # use seed 0 (present in all four pairwise clusters per s1 log)
        cseed = 0
        fires = {l: (get_fire(cseed, l) > THETA) for l in clique}
        rates = {l: float(fires[l].mean()) for l in clique}
        mat = {}
        for a in clique:
            for bb in clique:
                if a == bb:
                    continue
                inter = int((fires[a] & fires[bb]).sum())
                mat[f"P({a}|{bb})"] = round(inter / max(int(fires[bb].sum()), 1), 4)
        clique_out = {"latents": clique, "seed": cseed,
                      "rates": {l: round(rates[l], 5) for l in clique},
                      "containment": mat}
        print(f"[m=256 clique] rates={clique_out['rates']}", flush=True)
        print(f"[m=256 clique] containment={mat}", flush=True)

    results["widths"][str(M)] = {"clusters": cluster_out, "clique": clique_out}

json.dump(results, open(os.path.join(RES_R8, "natfeat_adjudication.json"), "w"),
          indent=1)
print("\nwrote results/round8/natfeat_adjudication.json", flush=True)
