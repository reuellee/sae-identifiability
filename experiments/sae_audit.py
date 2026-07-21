"""Direction 2: absorption-risk audit of an SAE dictionary via co-occurrence stats.

Method (lambda-free framing): run an SAE over real activations; find hierarchical
latent pairs (child j fires mostly alongside parent i); estimate per-pair
q_hat = P(co-fire), eps_hat = P(child solo), p0_hat = P(parent solo). Then:
 - critical lambda for the pair: lam_crit_pair = eps_hat/(1.17*q_hat) -- the pair
   is in the provably-absorbed regime for any sparsity coefficient above this.
 - geometry-penalty fixability: p0_hat/q_hat vs sqrt(2) (below -> no coherence
   penalty could rescue it; the no-go regime).
Audits a PUBLIC pre-trained GPT-2 SAE if downloadable; else audits the vanilla
SAE trained by natural_absorption_eval.py (method demo on our own dictionary).
CAVEAT (printed into output): latent pairs are post-absorption objects, not
ground-truth features; the audit maps risk structure, not confirmed absorption.

Outputs: results_audit.csv + audit_summary.txt.
"""
import torch, math, os, csv, json, time

dev = "cuda"

def try_public_sae():
    """Attempt to load a public GPT-2-small resid SAE from HF. Returns (W_enc[d,m],
    b_enc[m], W_dec[m,d], name) or None."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        for repo in ("jbloom/GPT2-Small-SAEs-Reformatted", "jbloom/GPT2-Small-SAEs"):
            try:
                files = api.list_repo_files(repo)
            except Exception as e:
                print(f"list {repo} failed: {e}", flush=True); continue
            cand = [f for f in files if ("6" in f and f.endswith((".pt", ".safetensors", ".npz"))
                    and ("resid" in f.lower() or "blocks" in f.lower() or "layer" in f.lower()))]
            print(f"{repo}: {len(files)} files; layer-6 candidates: {cand[:5]}", flush=True)
            for f in cand[:3]:
                try:
                    p = hf_hub_download(repo, f)
                    if f.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        sd = load_file(p)
                    else:
                        sd = torch.load(p, map_location="cpu", weights_only=False)
                    if isinstance(sd, dict):
                        keys = {k.lower().split(".")[-1]: k for k in sd.keys() if hasattr(sd[k], "shape")}
                        print(f"  {f}: keys {list(keys)[:8]}", flush=True)
                        we = next((sd[keys[k]] for k in keys if "w_enc" in k or k == "w_enc"), None)
                        wd = next((sd[keys[k]] for k in keys if "w_dec" in k), None)
                        be = next((sd[keys[k]] for k in keys if "b_enc" in k), None)
                        if we is not None and wd is not None:
                            if be is None: be = torch.zeros(we.shape[-1])
                            return we.float(), be.float(), wd.float(), f"{repo}/{f}"
                except Exception as e:
                    print(f"  load {f} failed: {e}", flush=True)
    except Exception as e:
        print(f"hub unavailable: {e}", flush=True)
    return None

def main():
    t0 = time.time()
    bg = torch.load("activations_l6.pt").float()
    pub = try_public_sae()
    source = None
    if pub is not None:
        We, be, Wd, source = pub
        # public SAEs expect RAW activations (their own normalization); use raw
        acts = bg.to(dev)
        We, be = We.to(dev), be.to(dev)
        enc = lambda X: torch.relu(X @ We + be)
        m = We.shape[1]
    else:
        # fallback: audit our own vanilla SAE -- retrain quickly (10k steps)
        source = "self-trained vanilla SAE (public SAE unavailable)"
        mu = bg.mean(0, keepdim=True)
        acts = ((bg - mu) * math.sqrt(768) / (bg - mu).norm(dim=1).mean()).to(dev)
        m = 1536
        bound = 1 / math.sqrt(768)
        W = torch.empty(m, 768, device=dev).uniform_(-bound, bound).requires_grad_()
        b = torch.zeros(m, device=dev, requires_grad=True)
        Dd = (torch.randn(768, m, device=dev) / math.sqrt(768)).requires_grad_()
        with torch.no_grad(): Dd.div_(Dd.norm(dim=0, keepdim=True))
        opt = torch.optim.Adam([W, b, Dd], lr=1e-3)
        gen = torch.Generator(device=dev).manual_seed(2)
        for t in range(10000):
            idx = torch.randint(0, acts.shape[0], (4096,), device=dev, generator=gen)
            x = acts[idx]
            f = torch.relu(x @ W.T + b)
            xh = f @ Dd.T
            loss = ((x - xh) ** 2).sum(-1).mean() + 1.0 * f.sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): Dd.div_(Dd.norm(dim=0, keepdim=True).clamp_min(1e-8))
            if t % 2500 == 0: print(f"  [audit-sae {t}/10000]", flush=True)
        enc = lambda X: torch.relu(X @ W.detach().T + b.detach())
    print(f"auditing: {source}, m={m}", flush=True)

    # firing statistics over the pool (chunked); keep top-K most active latents
    N = acts.shape[0]
    counts = torch.zeros(m, device=dev)
    for st in range(0, N, 32768):
        counts += (enc(acts[st:st + 32768].float()) > 0).float().sum(0)
    K = 1024
    top = torch.argsort(counts, descending=True)[:K]
    # co-activation matrix over top-K
    C = torch.zeros(K, K, device=dev)
    for st in range(0, N, 32768):
        A = (enc(acts[st:st + 32768].float())[:, top] > 0).float()
        C += A.T @ A
    n = torch.diag(C)
    rows = []
    for j in range(K):
        if n[j] < 200 or n[j] > 0.5 * N: continue
        cj = C[:, j] / n[j]
        cj[j] = 0
        i = int(cj.argmax())
        share = float(cj[i])
        if share < 0.7 or n[i] <= n[j]: continue      # child j under parent i
        nij = float(C[i, j]); ni = float(n[i]); nj = float(n[j])
        q_hat = nij / N
        eps_hat = (nj - nij) / N
        p0_hat = (ni - nij) / N
        lam_crit = eps_hat / (1.17 * q_hat) if q_hat > 0 else float("inf")
        rows.append(dict(child=int(top[j]), parent=int(top[i]),
                         n_child=int(nj), share=round(share, 3),
                         q_hat=round(q_hat, 5), eps_hat=round(eps_hat, 5),
                         p0_hat=round(p0_hat, 5),
                         lam_crit=round(lam_crit, 4),
                         p0_over_q=round(p0_hat / q_hat, 2) if q_hat > 0 else -1))
    rows.sort(key=lambda r: r["lam_crit"])
    with open("results_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["none"])
        w.writeheader(); w.writerows(rows)
    lcs = [r["lam_crit"] for r in rows]
    pq = [r["p0_over_q"] for r in rows]
    with open("audit_summary.txt", "w") as fh:
        fh.write(f"source: {source}\nhierarchical pairs found: {len(rows)}\n")
        if rows:
            lcs_s = sorted(lcs)
            fh.write(f"lam_crit median: {lcs_s[len(lcs_s)//2]:.4f}\n")
            fh.write(f"frac lam_crit < 0.05 (absorbed at any practical sparsity): "
                     f"{sum(1 for v in lcs if v < 0.05)/len(lcs):.2f}\n")
            fh.write(f"frac p0/q < sqrt2 (no coherence penalty could fix): "
                     f"{sum(1 for v in pq if 0 <= v < 1.414)/len(pq):.2f}\n")
        fh.write("CAVEAT: latent pairs are post-absorption objects, not ground-truth "
                 "features; this maps risk structure, not confirmed absorption.\n")
    print(open("audit_summary.txt").read(), flush=True)
    print(f"[audit] done in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
