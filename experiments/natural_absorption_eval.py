"""Direction 1 (lite benchmark): does residual-weighting reduce NATURAL absorption?

No injected features. Train vanilla vs residual-weighted SAEs on real GPT-2
layer-6 activations; measure first-letter feature absorption Chanin-style-lite:
for each frequent first letter L, probe direction = class-mean difference;
main latent = decoder column most aligned with the probe; absorption proxy =
coverage gap 1 - P(main latent fires | token starts with L) plus alignment.
Prediction (pre-registered): weighted SAEs show smaller coverage gaps (rare
L-token variants less absorbed into token-specific latents). This is a proxy
metric, not SAEBench; stated as such.

Outputs: results_natabs.csv (terse).
"""
import torch, math, csv, os, time, urllib.request

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
D_MODEL, M_LAT, BATCH, STEPS, WARMUP, LAM = 768, 1536, 2048, 20000, 4000, 1.0
N_SEEDS = 2

def get_letters():
    """Regenerate token-aligned first-letter labels (same corpus/chunking as
    extract_activations.py, deterministic)."""
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
    chunks = [ids[i:i + 512] for i in range(0, len(ids) - 512, 512)]
    flat = [t for ch in chunks for t in ch][:500_000]
    letters = []
    for t in flat:
        s = tok.decode([t]).strip().lower()
        letters.append(s[0] if s and s[0].isalpha() else "_")
    return letters

def main():
    bg = torch.load("activations_l6.pt").float()
    mu = bg.mean(0, keepdim=True)
    bg = (bg - mu) * math.sqrt(D_MODEL) / (bg - mu).norm(dim=1).mean()
    letters = get_letters()
    assert len(letters) == bg.shape[0]
    from collections import Counter
    freq = Counter(letters); freq.pop("_", None)
    top_letters = [l for l, _ in freq.most_common(8)]
    print("letters:", [(l, freq[l]) for l in top_letters], flush=True)
    probes = {}
    gmean = bg.mean(0)
    for L in top_letters:
        m = torch.tensor([c == L for c in letters])
        v = bg[m].mean(0) - gmean
        probes[L] = (v / v.norm()).to(dev)
    bg = bg.half().to(dev)
    N = bg.shape[0]
    lab = {L: torch.tensor([c == L for c in letters], device=dev) for L in top_letters}

    runs = [(c, s) for c in ("vanilla", "residual") for s in range(N_SEEDS)]
    Rn = len(runs)
    bound = 1 / math.sqrt(D_MODEL)
    torch.manual_seed(31)
    W = torch.nn.Parameter(torch.empty(Rn, M_LAT, D_MODEL, device=dev).uniform_(-bound, bound))
    b = torch.nn.Parameter(torch.zeros(Rn, M_LAT, device=dev))
    Dd = torch.nn.Parameter(torch.randn(Rn, D_MODEL, M_LAT, device=dev) / math.sqrt(D_MODEL))
    with torch.no_grad(): Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
    is_res = torch.tensor([c == "residual" for c, _ in runs], device=dev)
    opt = torch.optim.Adam([W, b, Dd], lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(STEPS):
        idx = torch.randint(0, N, (Rn, BATCH), device=dev, generator=gen)
        x = bg[idx].float()
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, W) + b.unsqueeze(1))
        xh = torch.einsum('rbm,rdm->rbd', f, Dd)
        res2 = ((x - xh) ** 2).sum(-1)
        per = res2 + LAM * f.sum(-1)
        w = torch.ones_like(per)
        if t >= WARMUP:
            wr = (res2 / res2.mean(1, keepdim=True)).clamp(1.0, 50.0).detach()
            w = torch.where(is_res.view(-1, 1).expand_as(w), wr, w)
        loss = (w * per).mean(-1).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
        if t == STEPS // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 4000 == 0:
            print(f"  [natabs {t}/{STEPS}] res={float(res2.mean()):.1f} ({time.time()-t0:.0f}s)", flush=True)
    # metrics: full-data latent firing per run (chunked)
    rows = []
    for i, (c, s) in enumerate(runs):
        fire = {L: 0 for L in top_letters}
        tot = {L: int(lab[L].sum()) for L in top_letters}
        # main latent per letter
        main = {}
        for L in top_letters:
            cosv = torch.einsum('dm,d->m', Dd[i].detach(), probes[L])
            main[L] = (int(cosv.argmax()), float(cosv.max()))
        # count firing on letter tokens, plus number of aligned latents (splitting)
        for st in range(0, N, 65536):
            xb = bg[st:st + 65536].float()
            fb = torch.relu(xb @ W[i].detach().T + b[i].detach())
            for L in top_letters:
                lm = lab[L][st:st + 65536]
                if lm.any():
                    fire[L] += int((fb[lm][:, main[L][0]] > 0).sum())
        for L in top_letters:
            n_aligned = int((torch.einsum('dm,d->m', Dd[i].detach(), probes[L]) > 0.35).sum())
            rows.append(dict(cond=c, seed=s, letter=L,
                             cov_gap=round(1 - fire[L] / max(tot[L], 1), 3),
                             align=round(main[L][1], 3), n_split=n_aligned))
    with open("results_natabs.csv", "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"[natabs] done in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
