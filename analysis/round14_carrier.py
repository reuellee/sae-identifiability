"""Round 14 (FROZEN AT LOCK): does the "absorbed" state have a carrier?

Pre-registration: notes/prereg-round14-carrier.md, committed at the same lock.
Do not tune anything here after seeing a result.

The absorbed set is reproduced EXACTLY as round13b_scorer.py defines it (same
probes, theta, tau, FAM_CAP, present/retained rules -- the relevant code is copied
verbatim), so gate 1 can check `rate_family` against frozen round-13b values. What
is new is only the carrier decomposition on top of that identical set.

Per token, latent i contributes to the letter direction u_L:
    c_i = f_i * (W_dec[:, i] . u_L)
(the scorer's 1/scale is common to all latents, so it cannot change an argmax or a
share, and is omitted). The carrier is argmax_i c_i over i not in F_L.

Trial sets within a letter, all requiring the letter present:
    A  absorbed      retained,     no F_L latent fires   <- round 13b's numerator
    C  control-fired an F_L latent fires
    N  lost          not retained, no F_L latent fires

Env: WORDS=words.pt SAES=<comma-sep .pt> OUT=<json> THETA=0.0 TAU=0.30
     FAM_CAP=32 MIN_WORDS=30 PROBE_C=1.0 MIN_A=20 NULL_DRAWS=32 SEED=0
CPU-only.
"""
import os, json, hashlib
import numpy as np
import torch

THETA = float(os.environ.get("THETA", "0.0"))
TAU = float(os.environ.get("TAU", "0.30"))
MIN_WORDS = int(os.environ.get("MIN_WORDS", "30"))
PROBE_C = float(os.environ.get("PROBE_C", "1.0"))
FAM_CAP = int(os.environ.get("FAM_CAP", "32"))
MIN_A = int(os.environ.get("MIN_A", "20"))          # prereg gate 2
NULL_DRAWS = int(os.environ.get("NULL_DRAWS", "32"))
SEED = int(os.environ.get("SEED", "0"))


# ---------------------------------------------------------------- copied verbatim
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
    return f.numpy(), Xhat, arch, k, W_dec.numpy()


# ---------------------------------------------------------------- round 14
def carrier_stats(F, Wdec, fam, rows_A, rows_C, u, rng):
    """Carrier = argmax_i f_i * (d_i . u) over i NOT in fam. Returns modal carrier
    on A, its activation on A and C, top-1 share, and the random-direction null."""
    align = Wdec.T @ u                       # (m,) latent alignment with u_L
    mask = np.ones(len(align), bool); mask[list(fam)] = False
    idx = np.where(mask)[0]

    def carriers(rows, a):
        c = F[np.ix_(rows, idx)] * a[idx]
        best = c.argmax(axis=1)
        return idx[best]

    cA = carriers(rows_A, align)
    vals, cnts = np.unique(cA, return_counts=True)
    kappa = int(vals[cnts.argmax()])
    share = float(cnts.max() / len(cA))

    # random-direction null: same trials, arbitrary direction (prereg P2)
    null = []
    for _ in range(NULL_DRAWS):
        r = rng.normal(size=u.shape); r /= np.linalg.norm(r)
        cR = carriers(rows_A, Wdec.T @ r)
        _, cn = np.unique(cR, return_counts=True)
        null.append(cn.max() / len(cR))

    return dict(
        kappa=kappa,
        act_A=float(F[rows_A, kappa].mean()),
        act_C=float(F[rows_C, kappa].mean()) if len(rows_C) else float("nan"),
        share=share,
        share_null=float(np.mean(null)),
        rate_kappa=float((F[:, kappa] > THETA).mean()),
        rate_fam=float(np.mean([(F[:, i] > THETA).mean() for i in fam])),
        # P4 concentration: kappa's share of total positive letter-direction mass
        conc_A=float(np.mean(_share_of(F[rows_A], align, kappa))),
        conc_C=float(np.mean(_share_of(F[rows_C], align,
                                       max(fam, key=lambda i: align[i]))))
        if len(rows_C) else float("nan"),
    )


def _share_of(Fr, align, i):
    c = Fr * align
    pos = np.clip(c, 0, None).sum(axis=1)
    return np.clip(c[:, i], 0, None) / np.maximum(pos, 1e-9)


def score_one(path, Xr, letters, probes, rng):
    s = safe_load(path)
    F, Xhat, arch, k, Wdec = encode(s, Xr)
    fires = F > THETA
    out = {}
    tot = dict(present=0, abs_fam=0)
    for L, (yL, present_all, lrf) in probes.items():
        retained_all = lrf.predict_proba(Xhat)[:, 1] > 0.5
        sel = fires[yL == 1].mean(0) - fires[yL == 0].mean(0)
        j = int(sel.argmax())
        if sel[j] < TAU:
            out[L] = dict(clean_latent=False)
            continue
        fam = np.where(sel >= TAU)[0]
        if len(fam) > FAM_CAP:
            fam = fam[np.argsort(sel[fam])[::-1][:FAM_CAP]]
        Lw = np.where(yL == 1)[0]
        present = present_all[Lw]; retained = retained_all[Lw]
        famfire = fires[np.ix_(Lw, fam)].any(axis=1)

        miss_fam = present & (~famfire) & retained          # set A (13b numerator)
        ctrl = present & famfire                            # set C
        lost = present & (~famfire) & (~retained)           # set N
        npres = int(present.sum())
        tot["present"] += npres; tot["abs_fam"] += int(miss_fam.sum())

        rec = dict(clean_latent=True, fam_size=int(len(fam)),
                   n_A=int(miss_fam.sum()), n_C=int(ctrl.sum()),
                   n_N=int(lost.sum()), letter_present=npres,
                   rate_family=round(float(miss_fam.sum() / max(npres, 1)), 4))
        if int(miss_fam.sum()) >= MIN_A and int(ctrl.sum()) > 0:
            u = lrf.coef_[0].astype(np.float64)
            u = u / max(np.linalg.norm(u), 1e-12)
            rec.update(carrier_stats(F, Wdec, fam, Lw[miss_fam], Lw[ctrl], u, rng))
            rec["scored"] = True
        else:
            rec["scored"] = False                           # prereg gate 2 exclusion
        out[L] = rec
    return dict(sae=os.path.basename(path), sha256=sha256(path), arch=arch,
                m=int(s["W_enc"].shape[0]), seed=s.get("seed"),
                expansion=int(s["W_enc"].shape[0] // s["W_dec"].shape[0]),
                theta=THETA, tau=TAU, min_a=MIN_A,
                rate_family=round(tot["abs_fam"] / max(tot["present"], 1), 4),
                per_letter=out)


def main():
    # Loader COPIED VERBATIM from round13b_scorer.py. The first version of this
    # file invented its own (d["X"]) and died instantly with KeyError: 'X' -- the
    # words file's key is "acts". Copying rather than paraphrasing is the whole
    # reason the 13a->13b counters transferred cleanly; the same discipline applies
    # to the loader.
    W = safe_load(os.environ["WORDS"])
    Xr = W["acts"].numpy().astype("float32")
    letters = np.array(W["letters"])
    print(f"words={Xr.shape} letters={len(set(letters))}", flush=True)
    probes = build_probes(Xr, letters)
    rng = np.random.default_rng(SEED)
    rows = []
    for p in os.environ["SAES"].split(","):
        p = p.strip()
        if not p:
            continue
        rows.append(score_one(p, Xr, letters, probes, rng))
        print(f"  scored {os.path.basename(p)} "
              f"rate_family={rows[-1]['rate_family']}", flush=True)
    json.dump(rows, open(os.environ.get("OUT", "round14_results.json"), "w"))
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
