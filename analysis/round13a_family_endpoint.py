"""Round 13a (FROZEN AT LOCK): family-based first-letter absorption endpoint.

Pre-registration: notes/prereg-round13a-family-endpoint.md. This file is frozen at
the same commit. Do not tune anything here after seeing a result.

Question: does the round-12 first-letter absorption signal survive when the letter
is scored against its whole SPLIT FAMILY instead of one designated main latent?

Re-scores the 16 EXISTING round-12 SAEs on the SAME held-out words, changing only
the endpoint. Everything else (theta, probes, present/retained, sel rule) is copied
from experiments/real_firstletter.py so the baseline reproduces exactly (gate 4).

Endpoints, per letter L:
  sel_i      = P(fire_i | L) - P(fire_i | not L)
  SINGLE     j = argmax_i sel_i, scored iff sel_j >= TAU; absorbed iff
               present & retained & not fire_j                  [round-12 registered]
  FAMILY     F_L = {i : sel_i >= TAU}, capped at 32 by sel; absorbed iff
               present & retained & NO i in F_L fires           [new]

Env: WORDS=words.pt  SAES=<comma-separated .pt paths>  OUT=<json path>
     THETA=0.0  TAU=0.30  MIN_WORDS=30  PROBE_C=1.0  FAM_CAP=32
CPU-only. Probes are SAE-independent and are fit ONCE, then reused.
"""
import os, json, hashlib
import numpy as np
import torch

THETA = float(os.environ.get("THETA", "0.0"))
TAU = float(os.environ.get("TAU", "0.30"))
MIN_WORDS = int(os.environ.get("MIN_WORDS", "30"))
PROBE_C = float(os.environ.get("PROBE_C", "1.0"))
FAM_CAP = int(os.environ.get("FAM_CAP", "32"))
BOOT = 10_000


def safe_load(p):
    try:
        return torch.load(p, weights_only=True, map_location="cpu")
    except Exception:
        return torch.load(p, map_location="cpu")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def probe_oof(X, y, C=1.0, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(y)); w = np.zeros(X.shape[1]); nf = 0
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        if len(set(y[tr])) < 2:
            continue
        lr = LogisticRegression(C=C, max_iter=200, class_weight="balanced")
        lr.fit(X[tr], y[tr]); oof[te] = lr.predict_proba(X[te])[:, 1]
        w += lr.coef_[0]; nf += 1
    return oof, w / max(nf, 1)


def build_probes(Xr, letters):
    """SAE-INDEPENDENT: present-mask (out-of-fold) + the full-fit probe applied
    later to each SAE's reconstruction for the retention check."""
    from sklearn.linear_model import LogisticRegression
    probes = {}
    for L in sorted(set(letters)):
        yL = (letters == L).astype(int)
        if yL.sum() < MIN_WORDS or (1 - yL).sum() < MIN_WORDS:
            continue
        oof, _ = probe_oof(Xr, yL, C=PROBE_C)
        lrf = LogisticRegression(C=PROBE_C, max_iter=200,
                                 class_weight="balanced").fit(Xr, yL)
        probes[L] = (yL, oof > 0.5, lrf)
        print(f"  probe {L}: n={int(yL.sum())}", flush=True)
    return probes


def encode(s, Xr):
    mu = s["mu"].float(); scale = float(s["scale"])
    arch = s["arch"]; k = int(s.get("k", 32))
    Xn = torch.tensor((Xr - mu.numpy()) * scale)
    W_enc = s["W_enc"].float(); b_enc = s["b_enc"].float()
    W_dec = s["W_dec"].float(); b_dec = s["b_dec"].float()
    with torch.no_grad():
        f = torch.relu((Xn - b_dec) @ W_enc.T + b_enc)
        if arch == "topk":
            thr = f.topk(k, dim=-1).values[:, -1:].clamp_min(1e-12)
            f = f * (f >= thr).float()
        Xhat = ((f @ W_dec.T + b_dec) / scale + mu).numpy().astype("float32")
    return f.numpy(), Xhat, arch, k


def score_one(path, Xr, letters, probes):
    s = safe_load(path)
    F, Xhat, arch, k = encode(s, Xr)
    fires = F > THETA
    out = {}
    tot = dict(present=0, abs_single=0, abs_fam=0)
    for L, (yL, present_all, lrf) in probes.items():
        retained_all = lrf.predict_proba(Xhat)[:, 1] > 0.5
        sel = fires[yL == 1].mean(0) - fires[yL == 0].mean(0)
        j = int(sel.argmax())
        if sel[j] < TAU:
            out[L] = dict(clean_latent=False, sel=round(float(sel[j]), 3))
            continue
        fam = np.where(sel >= TAU)[0]
        if len(fam) > FAM_CAP:                      # registered cap, by sel
            fam = fam[np.argsort(sel[fam])[::-1][:FAM_CAP]]
        Lw = np.where(yL == 1)[0]
        present = present_all[Lw]; retained = retained_all[Lw]
        miss_single = present & (~fires[Lw, j]) & retained
        miss_fam = present & (~fires[np.ix_(Lw, fam)].any(axis=1)) & retained
        npres = int(present.sum())
        out[L] = dict(clean_latent=True, latent=j, sel=round(float(sel[j]), 3),
                      fam_size=int(len(fam)),
                      fam_max_sel=round(float(sel[fam].max()), 3),
                      letter_present=npres,
                      absorbed_single=int(miss_single.sum()),
                      absorbed_family=int(miss_fam.sum()),
                      rate_single=round(float(miss_single.sum() / max(npres, 1)), 4),
                      rate_family=round(float(miss_fam.sum() / max(npres, 1)), 4))
        tot["present"] += npres
        tot["abs_single"] += int(miss_single.sum())
        tot["abs_fam"] += int(miss_fam.sum())
    return dict(sae=os.path.basename(path), sha256=sha256(path), arch=arch,
                k=k, lam=s.get("lam"), m=int(s["W_enc"].shape[0]),
                seed=s.get("seed"), theta=THETA, tau=TAU, fam_cap=FAM_CAP,
                model=s.get("model"), layer=s.get("layer"),
                rate_single=round(tot["abs_single"] / max(tot["present"], 1), 4),
                rate_family=round(tot["abs_fam"] / max(tot["present"], 1), 4),
                per_letter=out)


def boot_ci(x, reps=BOOT, seed=1):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    ms = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def main():
    W = safe_load(os.environ["WORDS"])
    Xr = W["acts"].numpy().astype("float32")
    letters = np.array(W["letters"])
    print(f"words={Xr.shape} letters={len(set(letters))}", flush=True)
    print("fitting SAE-independent probes once...", flush=True)
    probes = build_probes(Xr, letters)

    rows = []
    for p in os.environ["SAES"].split(","):
        p = p.strip()
        if not p:
            continue
        r = score_one(p, Xr, letters, probes)
        rows.append(r)
        print(f"  {r['sae']}: single={r['rate_single']:.4f} family={r['rate_family']:.4f} "
              f"arch={r['arch']} seed={r['seed']}", flush=True)

    out = os.environ.get("OUT", "round13a_family.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out} ({len(rows)} SAEs)")


if __name__ == "__main__":
    main()
