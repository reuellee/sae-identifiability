"""Arm 2 (real data, GATED on Arm 1 D1+D2 pass) of the pair-identification
pre-registration: does the label-free detector find the absorbed pair in
capacity-limited SAEs on real GPT-2 activations?

Setup mirrors experiments/capacity_limited_test.py exactly (same normalization,
AMP=5, Q=0.2, P0=0.2, LAM=1.0, STEPS=20000) but VANILLA ONLY:
  m in {128, 256} x eps in {0.002 (absorbed regime), 0.05 (faithful control)}
  x 8 seeds = 32 runs (two batched programs).
Detector, thresholds, theta, rate window: IDENTICAL to Arm 1 (locked in
notes/prereg-pair-identification.md) — no re-tuning for real data; transfer is
the test. Oracle directions are used ONLY to score (parent latent, composite
latent, absorbed criterion cos_comp > 0.98), per the registered scope.

Registered readouts (from the prereg decision rule):
  R1: fraction of absorbed-formed eps=0.002 runs whose oracle pair is flagged.
  R2: flags in eps=0.05 (faithful) runs touching the oracle pair (should be ~0).
  R3: audit-v3 descriptive scan — total flags/SAE on the real-feature
      background (correlated real features may legitimately flag; reported,
      not thresholded).
  R4 (descriptive): rho_hat from the detected pair vs true child-given-parent
      rate Q/(Q+P0) = 0.5.

Needs activations_l6.pt (created by extract_activations.py if absent).
Outputs: results/prereg_pairid/arm2_runs.csv
"""
import torch, math, csv, time, os, sys, subprocess

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

# ---- detector constants: MUST match Arm 1 / the prereg note ----------------
C_LO, C_HI = 0.45, 0.90
L_LO, L_HI = 0.5, 2.0        # two-sided lift rule, locked from the pilot
THETA = 0.05
RATE_LO, RATE_HI = 5e-4, 0.6
N_EV = 200_000
# ----------------------------------------------------------------------------

D_MODEL, BATCH, STEPS = 768, 2048, 20000
LAM, Q, P0, AMP = 1.0, 0.2, 0.2, 5.0
SEEDS = 8
OUTDIR = os.path.join(HERE, "..", "results", "prereg_pairid")

ACT = os.path.join(HERE, "activations_l6.pt")
if not os.path.exists(ACT):
    if not os.path.exists("activations_l6.pt"):
        print("extracting activations (needs transformers)...", flush=True)
        try:
            import transformers  # noqa
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--user",
                            "-q", "transformers"], check=True)
        import extract_activations
        extract_activations.main()
    ACT = "activations_l6.pt"

bg = torch.load(ACT).float()
mu = bg.mean(0, keepdim=True)
bg = ((bg - mu) * math.sqrt(D_MODEL) / (bg - mu).norm(dim=1).mean()).half().to(dev)
N = bg.shape[0]
print(f"bg {bg.shape} device={dev}", flush=True)

pairs = {}
for s in range(SEEDS):
    g = torch.Generator().manual_seed(300 + s)   # seeds 0-3 = same pairs as §15
    Qm, _ = torch.linalg.qr(torch.randn(D_MODEL, 2, generator=g))
    pairs[s] = Qm.T.to(dev)

def detect(Dn, fires):
    rates = fires.mean(0)
    keep = (rates > RATE_LO) & (rates < RATE_HI)
    C = np.abs(Dn.T @ Dn)
    F32 = fires.astype(np.float32)
    Pj = (F32.T @ F32) / len(fires)
    L = Pj / np.maximum(np.outer(rates, rates), 1e-12)
    m = Dn.shape[1]
    flags = []
    for i in range(m):
        if not keep[i]: continue
        for j in range(i + 1, m):
            if not keep[j]: continue
            if C_LO < C[i, j] < C_HI and (L[i, j] <= L_LO or L[i, j] >= L_HI):
                flags.append((i, j))
    return flags, C, L, rates

rows_csv = []
for M_LAT in (128, 256):
    runs = [(e, s) for e in (0.002, 0.05) for s in range(SEEDS)]
    Rn = len(runs)
    ap = torch.stack([pairs[s][0] for _, s in runs])
    ac = torch.stack([pairs[s][1] for _, s in runs])
    eps_vec = torch.tensor([e for e, _ in runs], device=dev)
    bound = 1 / math.sqrt(D_MODEL)
    torch.manual_seed(51)
    W = torch.nn.Parameter(torch.empty(Rn, M_LAT, D_MODEL, device=dev).uniform_(-bound, bound))
    b = torch.nn.Parameter(torch.zeros(Rn, M_LAT, device=dev))
    Dd = torch.nn.Parameter(torch.randn(Rn, D_MODEL, M_LAT, device=dev) / math.sqrt(D_MODEL))
    with torch.no_grad(): Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
    opt = torch.optim.Adam([W, b, Dd], lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(1)
    t0 = time.time()
    for t in range(STEPS):
        idx = torch.randint(0, N, (Rn, BATCH), device=dev, generator=gen)
        x = bg[idx].float()
        u = torch.rand(Rn, BATCH, device=dev, generator=gen)
        e = eps_vec.view(-1, 1)
        j = (u < Q).float(); p = ((u >= Q) & (u < Q + P0)).float()
        cs = ((u >= Q + P0) & (u < Q + P0 + e)).float()
        x = (x + AMP * (j + p).unsqueeze(-1) * ap.unsqueeze(1)
               + AMP * (j + cs).unsqueeze(-1) * ac.unsqueeze(1))
        f = torch.relu(torch.einsum('rbd,rmd->rbm', x, W) + b.unsqueeze(1))
        xh = torch.einsum('rbm,rdm->rbd', f, Dd)
        loss = (((x - xh) ** 2).sum(-1) + LAM * f.sum(-1)).mean(-1).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Dd.div_(Dd.norm(dim=1, keepdim=True).clamp_min(1e-8))
        if t == STEPS // 2:
            for gp in opt.param_groups: gp["lr"] = 1e-3 / 3
        if t % 4000 == 0:
            print(f"  [m={M_LAT} {t}/{STEPS}] ({time.time()-t0:.0f}s)", flush=True)
    torch.save(dict(W=W.detach().cpu(), b=b.detach().cpu(), D=Dd.detach().cpu(),
                    runs=runs), os.path.join(OUTDIR, f"weights_arm2_m{M_LAT}.pt"))
    # ------------------------------------------------------------- evaluate
    for i, (e, s) in enumerate(runs):
        ap_, ac_ = pairs[s][0], pairs[s][1]
        comp_dir = (ap_ + ac_) / math.sqrt(2)
        Di = Dd[i].detach()
        par = int(torch.einsum('dm,d->m', Di, ap_).abs().argmax())
        comp = int(torch.einsum('dm,d->m', Di, comp_dir).abs().argmax())
        cos_comp = float(torch.einsum('dm,d->m', Di, comp_dir).abs().max())
        cos_child = float(torch.einsum('dm,d->m', Di, ac_).abs().max())
        # labeled eval stream (same generative process, fresh draws)
        g2 = torch.Generator(device=dev).manual_seed(9000 + s)
        fires_chunks, labels = [], []
        for c0 in range(0, N_EV, 50_000):
            n = min(50_000, N_EV - c0)
            idx = torch.randint(0, N, (n,), device=dev, generator=g2)
            x = bg[idx].float()
            u = torch.rand(n, device=dev, generator=g2)
            jj = (u < Q).float(); pp = ((u >= Q) & (u < Q + P0)).float()
            cc = ((u >= Q + P0) & (u < Q + P0 + e)).float()
            x = (x + AMP * (jj + pp).unsqueeze(-1) * ap_
                   + AMP * (jj + cc).unsqueeze(-1) * ac_)
            f = torch.relu(x @ W[i].detach().T + b[i].detach())
            fires_chunks.append((f > THETA).cpu().numpy())
            labels.append(torch.stack([jj, pp, cc]).cpu().numpy())
        fires = np.concatenate(fires_chunks)
        jj, pp, cc = np.concatenate(labels, axis=1)
        flags, C, L, rates = detect(Di.cpu().numpy(), fires)
        tp = tuple(sorted((par, comp)))
        tp_flagged = int(tp in flags and par != comp)
        row = dict(m=M_LAT, eps=e, seed=s, absorbed=int(cos_comp > 0.98),
                   cos_comp=round(cos_comp, 4), cos_child=round(cos_child, 4),
                   par_idx=par, comp_idx=comp, distinct=int(par != comp),
                   tp_cos=round(float(C[tp]), 4), tp_lift=round(float(L[tp]), 4),
                   rate_par=round(float(rates[par]), 5),
                   rate_comp=round(float(rates[comp]), 5),
                   n_flagged=len(flags), tp_flagged=tp_flagged)
        if tp_flagged:
            lo, hi = (tp if rates[tp[0]] < rates[tp[1]] else (tp[1], tp[0]))
            u_res = Di[:, lo] - (Di[:, lo] * Di[:, hi]).sum() * Di[:, hi]
            u_res = u_res / u_res.norm().clamp_min(1e-8)
            row["child_res_cos"] = round(float((u_res * ac_).sum().abs()), 4)
            either = fires[:, lo] | fires[:, hi]
            row["rho_hat"] = round(float(fires[:, lo].mean()
                                         / max(either.mean(), 1e-9)), 5)
        # diagnostics: gating on real data (composite fire by event class)
        row["comp_fire_joint"] = round(float(fires[:, comp][jj > 0.5].mean()), 4)
        row["comp_fire_psolo"] = round(float(fires[:, comp][pp > 0.5].mean()), 4)
        row["par_fire_joint"] = round(float(fires[:, par][jj > 0.5].mean()), 4)
        row["par_fire_psolo"] = round(float(fires[:, par][pp > 0.5].mean()), 4)
        rows_csv.append(row)
        print(f"[m={M_LAT} eps={e} s{s}] abs={row['absorbed']} "
              f"tp_flagged={tp_flagged} tp_cos={row['tp_cos']} "
              f"tp_lift={row['tp_lift']} n_flags={len(flags)}", flush=True)
    del W, b, Dd, opt
    if dev == "cuda": torch.cuda.empty_cache()

os.makedirs(OUTDIR, exist_ok=True)
fields = []
for r in rows_csv:
    for k in r:
        if k not in fields: fields.append(k)
out = os.path.join(OUTDIR, "arm2_runs.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader(); w.writerows(rows_csv)
print(f"wrote {out} ({len(rows_csv)} rows)", flush=True)
