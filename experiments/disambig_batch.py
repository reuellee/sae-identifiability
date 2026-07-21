"""Disambiguation batch (specs: results/round6/REVIEW_D12.md + CSV mining).

Part A (POC v2): rerun 3-condition injected-pair POC with the metrics that make
  claims clean: (i) REALISTIC probes (real background sample + AMP*a_c) for the
  functional/routing metric -- the bare-probe artifact killed the v1 metric;
  (ii) decoder-side triple inventory (max cos to child/comp/parent);
  (iii) routing score: P(best child-aligned latent fires | child-solo input).
Part B (natabs v2): vanilla vs residual with the firing-rate confound controls:
  global mean L0, and letter-latent SELECTIVITY (firing rate on non-letter
  tokens) so coverage-gap gains can't be explained by global densification.
Part C (audit v2): public GPT-2 SAE, statistical criteria: enrichment
  P(i|j)/P(i) > 5, directionality P(i|j) > 2*P(j|i), n_j >= 100, K=4096,
  plus activation-magnitude correlation among co-firing pairs.
Writes SUMMARY2.md. Runtime target ~40 min on L4.
"""
import torch, math, csv, time, urllib.request, json

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
D_MODEL, M_LAT, BATCH = 768, 1536, 2048
LAM, Q, P0, AMP = 1.0, 0.2, 0.2, 5.0
out_lines = ["# Disambiguation SUMMARY2\n"]

def log(s):
    print(s, flush=True); out_lines.append(s)

def load_bg():
    bg = torch.load("activations_l6.pt").float()
    mu = bg.mean(0, keepdim=True)
    return ((bg - mu) * math.sqrt(D_MODEL) / (bg - mu).norm(dim=1).mean()).half().to(dev)

def train(runs, bg, eps_of, weight_mode_of, steps=20000, warmup=4000, inject=True,
          pairs=None, seed0=41):
    Rn = len(runs)
    bound = 1 / math.sqrt(D_MODEL)
    torch.manual_seed(seed0)
    W = torch.nn.Parameter(torch.empty(Rn, M_LAT, D_MODEL, device=dev).uniform_(-bound, bound))
    b = torch.nn.Parameter(torch.zeros(Rn, M_LAT, device=dev))
    Dd = torch.nn.Parameter(torch.randn(Rn, D_MODEL, M_LAT, device=dev) / math.sqrt(D_MODEL))
    with torch.no_grad(): Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
    opt = torch.optim.Adam([W, b, Dd], lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    N = bg.shape[0]
    ap = torch.stack([pairs[s][0] for s in [r[-1] for r in runs]]) if inject else None
    ac = torch.stack([pairs[s][1] for s in [r[-1] for r in runs]]) if inject else None
    eps_vec = torch.tensor([eps_of(r) for r in runs], device=dev)
    mode = [weight_mode_of(r) for r in runs]
    is_or = torch.tensor([m == "oracle" for m in mode], device=dev)
    is_re = torch.tensor([m == "residual" for m in mode], device=dev)
    t0 = time.time()
    for t in range(steps):
        idx = torch.randint(0, N, (Rn, BATCH), device=dev, generator=gen)
        x = bg[idx].float()
        if inject:
            u = torch.rand(Rn, BATCH, device=dev, generator=gen)
            e = eps_vec.view(-1, 1)
            j = (u < Q).float(); p = ((u >= Q) & (u < Q + P0)).float()
            cs = ((u >= Q + P0) & (u < Q + P0 + e)).float()
            x = (x + AMP * (j + p).unsqueeze(-1) * ap.unsqueeze(1)
                   + AMP * (j + cs).unsqueeze(-1) * ac.unsqueeze(1))
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, W) + b.unsqueeze(1))
        xh = torch.einsum('rbm,rdm->rbd', f, Dd)
        res2 = ((x - xh) ** 2).sum(-1)
        per = res2 + LAM * f.sum(-1)
        w = torch.ones_like(per)
        if inject:
            pn = (1 - Q - P0 - e).clamp_min(1e-6)
            nn = 1 - j - p - cs
            wor = (j / Q + p / P0 + cs / e.clamp_min(1e-6) + nn / pn) / 4
            w = torch.where(is_or.view(-1, 1).expand_as(w), wor, w)
        if t >= warmup:
            wr = (res2 / res2.mean(1, keepdim=True)).clamp(1.0, 50.0).detach()
            w = torch.where(is_re.view(-1, 1).expand_as(w), wr, w)
        loss = (w * per).mean(-1).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
        if t == steps // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 4000 == 0: print(f"  [train {t}/{steps}] ({time.time()-t0:.0f}s)", flush=True)
    return W.detach(), b.detach(), Dd.detach()

def part_A(bg):
    log("\n## A. POC v2: realistic probes + routing metric")
    n_seeds = 4
    pairs = {}
    for s in range(n_seeds):
        g = torch.Generator().manual_seed(300 + s)
        Qm, _ = torch.linalg.qr(torch.randn(D_MODEL, 2, generator=g))
        pairs[s] = Qm.T.to(dev)
    runs = [(c, e, s) for c in ("vanilla", "oracle", "residual")
            for e in (0.002, 0.01, 0.05) for s in range(n_seeds)]
    W, b, Dd = train(runs, bg, lambda r: r[1], lambda r: r[0], pairs=pairs)
    N = bg.shape[0]
    gen = torch.Generator(device=dev).manual_seed(7)
    pidx = torch.randint(0, N, (256,), device=dev, generator=gen)
    for i, (c, e, s) in enumerate(runs):
        ap_, ac_ = pairs[s][0], pairs[s][1]
        comp = (ap_ + ac_) / math.sqrt(2)
        cosc = torch.einsum('dm,d->m', Dd[i], ac_)
        best = int(cosc.argmax())
        # realistic child-solo probes: real background + AMP*a_c
        xp = bg[pidx].float() + AMP * ac_
        f = torch.relu(xp @ W[i].T + b[i])
        route = float((f[:, best] > 0).float().mean())          # routing score
        arg_lat = f.argmax(1)
        arg_cos = float(cosc[arg_lat].mean())                    # avg child-cos of argmax latent
        log(f"- {c} eps={e} s{s}: dec[child={float(cosc.max()):.2f} "
            f"comp={float(torch.einsum('dm,d->m', Dd[i], comp).max()):.2f}] "
            f"route(best-child fires|csolo)={route:.2f} argmax_latent_child_cos={arg_cos:.2f}")

def part_B(bg):
    log("\n## B. natabs v2 with firing-rate controls")
    from transformers import GPT2Tokenizer
    urls = ["https://www.gutenberg.org/files/2600/2600-0.txt",
            "https://www.gutenberg.org/files/1661/1661-0.txt"]
    text = ""
    for u in urls:
        try: text += urllib.request.urlopen(u, timeout=60).read().decode("utf-8", "ignore")
        except Exception: pass
        if len(text) > 3_500_000: break
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ids = tok(text, return_tensors=None)["input_ids"]
    chunks = [ids[i:i+512] for i in range(0, len(ids)-512, 512)]
    flat = [t for ch in chunks for t in ch][:bg.shape[0]]
    letters = [(tok.decode([t]).strip().lower() or "_")[0] for t in flat]
    letters = [c if c.isalpha() else "_" for c in letters]
    from collections import Counter
    freq = Counter(letters); freq.pop("_", None)
    top = [l for l, _ in freq.most_common(8)]
    bgf = bg.float()
    gmean = bgf.mean(0)
    probes = {}
    for L in top:
        m = torch.tensor([c == L for c in letters])
        v = bgf[m].mean(0) - gmean
        probes[L] = (v / v.norm()).to(dev)
    lab = {L: torch.tensor([c == L for c in letters], device=dev) for L in top}
    runs = [(c, s) for c in ("vanilla", "residual") for s in range(2)]
    W, b, Dd = train(runs, bg, lambda r: 0.0, lambda r: r[0], inject=False,
                     pairs=None, seed0=31)
    N = bg.shape[0]
    for i, (c, s) in enumerate(runs):
        L0_sum, cov, sel = 0.0, {}, {}
        main = {L: int(torch.einsum('dm,d->m', Dd[i], probes[L]).argmax()) for L in top}
        fire_in = {L: 0 for L in top}; fire_out = {L: 0 for L in top}
        n_out = {L: 0 for L in top}
        for st in range(0, N, 65536):
            xb = bg[st:st+65536].float()
            fb = torch.relu(xb @ W[i].T + b[i])
            L0_sum += float((fb > 0).sum())
            for L in top:
                lm = lab[L][st:st+65536]
                fire_in[L] += int((fb[lm][:, main[L]] > 0).sum())
                fire_out[L] += int((fb[~lm][:, main[L]] > 0).sum())
                n_out[L] += int((~lm).sum())
        gaps, sels = [], []
        for L in top:
            tot = int(lab[L].sum())
            gaps.append(1 - fire_in[L] / max(tot, 1))
            sels.append(fire_out[L] / max(n_out[L], 1))
        log(f"- {c} s{s}: global_L0={L0_sum/N:.1f}  cov_gap_mean={sum(gaps)/len(gaps):.3f}  "
            f"offletter_fire_mean={sum(sels)/len(sels):.4f}")

def part_C():
    log("\n## C. audit v2 (statistical criteria) on public GPT-2 SAE")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    p = hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted",
                        "blocks.6.hook_resid_pre/sae_weights.safetensors")
    sd = load_file(p)
    We = sd["W_enc"].float().to(dev); be = sd["b_enc"].float().to(dev)
    bg = torch.load("activations_l6.pt").float().to(dev)
    N = bg.shape[0]
    m = We.shape[1]
    counts = torch.zeros(m, device=dev)
    for st in range(0, N, 16384):
        counts += (torch.relu(bg[st:st+16384] @ We + be) > 0).float().sum(0)
    K = 4096
    top = torch.argsort(counts, descending=True)[:K]
    C = torch.zeros(K, K, device=dev)
    Msum = torch.zeros(K, K, device=dev)   # sum of act_i*act_j for corr proxy
    for st in range(0, N, 16384):
        F = torch.relu(bg[st:st+16384] @ We + be)[:, top]
        A = (F > 0).float()
        C += A.T @ A
        Msum += F.T @ F
    n = torch.diag(C).clamp_min(1)
    pairs = []
    base = n / N
    for jj in range(K):
        nj = float(n[jj])
        if nj < 100 or nj > 0.3 * N: continue
        pij = C[:, jj] / nj                      # P(i | j)
        enrich = pij / base.clamp_min(1e-9)      # P(i|j)/P(i)
        pij[jj] = 0; enrich[jj] = 0
        cand = torch.nonzero((enrich > 5) & (pij > 0.4) & (n > n[jj] * 2)).flatten()
        for ii in cand.tolist():
            pji = float(C[ii, jj] / n[ii])
            if pij[ii] < 2 * pji: continue       # directionality
            nij = float(C[ii, jj]); ni = float(n[ii])
            q_hat = nij / N; eps_hat = (nj - nij) / N; p0_hat = (ni - nij) / N
            pairs.append(dict(parent=int(top[ii]), child=int(top[jj]),
                              share=round(float(pij[ii]), 2),
                              enrich=round(float(enrich[ii]), 1),
                              lam_crit=round(eps_hat / (1.17 * q_hat), 4),
                              p0_over_q=round(p0_hat / q_hat, 2)))
    pairs.sort(key=lambda r: r["lam_crit"])
    with open("results_audit_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()) if pairs else ["none"])
        w.writeheader(); w.writerows(pairs)
    log(f"- hierarchical pairs found: {len(pairs)}")
    if pairs:
        lcs = sorted(r["lam_crit"] for r in pairs)
        pqs = [r["p0_over_q"] for r in pairs]
        log(f"- lam_crit median={lcs[len(lcs)//2]:.4f}  frac<0.05={sum(1 for v in lcs if v<0.05)/len(lcs):.2f}")
        log(f"- frac p0/q < sqrt2 (coherence-penalty-unfixable regime)={sum(1 for v in pqs if v<1.414)/len(pqs):.2f}")
        log(f"- examples: {pairs[:5]}")

if __name__ == "__main__":
    t0 = time.time()
    bg = load_bg()
    part_A(bg)
    part_B(bg)
    del bg; torch.cuda.empty_cache()
    part_C()
    log(f"\ntotal {time.time()-t0:.0f}s")
    open("SUMMARY2.md", "w").write("\n".join(out_lines))
    open("disambig_done.flag", "w").write("ok\n")
