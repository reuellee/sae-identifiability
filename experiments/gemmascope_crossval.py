"""Round 15 (FROZEN AT LOCK): Gemma Scope 2 cross-validation harness.

Pre-registration: notes/prereg-round15-gemmascope-crossval.md. This file is frozen
at the same commit. Do not tune anything here after seeing a result.

Ports the round-13a first-letter machinery (probes, present/retained, sel rule,
SINGLE + FAMILY endpoints; theta=0, tau=0.30, FAM_CAP=32, MIN_WORDS=30) to the
public Gemma Scope 2 JumpReLU suite on Gemma 3 1B. No training.

Two entry points via MODE env:
  MODE=words -> build the whole-word token set from the Gemma tokenizer, extract
                each word-token's residual at hidden_states[LAYER+1] (BOS+token;
                Gemma Scope's hf_hook_point_in = model.layers.<LAYER>.output),
                registered stratified subsample to WORD_CAP, cache to words .pt.
  MODE=score -> SAE_DIRS=<comma-separated local dirs, each with params.safetensors
                + config.json> WORDS=<words.pt> OUT=<json>: probes fit ONCE
                (SAE-independent), then each SAE is scored streaming (fires kept
                as bool; magnitudes kept fp16 only when width <= GRID_MAX_WIDTH,
                for the descriptive theta grid).

JumpReLU fire definition (prereg): f = relu(pre) * (pre > threshold); fire <=> f>0.

Env: MODEL=unsloth/gemma-3-1b-pt  LAYER=13  WORD_CAP=24000  THETA=0.0  TAU=0.30
     FAM_CAP=32  MIN_WORDS=30  PROBE_C=1.0  GRID=0.0,0.01,0.05,0.1
     GRID_MAX_WIDTH=65536  BATCH=2048  SMOKE=0
"""
import os, sys, re, json, hashlib
import numpy as np
import torch

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
MODEL = os.environ.get("MODEL", "unsloth/gemma-3-1b-pt")
# All registered hook layers are extracted in ONE forward pass (GPT pre-lock
# review P2.1: scoring a layer-l SAE against layer-13 activations is garbage).
LAYERS = [int(x) for x in os.environ.get("LAYERS", "7,13,17,22").split(",")]
MODE = os.environ.get("MODE", "words")
WORD_CAP = int(os.environ.get("WORD_CAP", "24000" if not SMOKE else "1500"))
THETA = float(os.environ.get("THETA", "0.0"))
TAU = float(os.environ.get("TAU", "0.30"))
FAM_CAP = int(os.environ.get("FAM_CAP", "32"))
MIN_WORDS = int(os.environ.get("MIN_WORDS", "30" if not SMOKE else "5"))
PROBE_C = float(os.environ.get("PROBE_C", "1.0"))
GRID = [float(x) for x in os.environ.get("GRID", "0.0,0.01,0.05,0.1").split(",")]
GRID_MAX_WIDTH = int(os.environ.get("GRID_MAX_WIDTH", "65536"))
BATCH = int(os.environ.get("BATCH", "1024"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("OUTDIR", os.path.join(HERE, "..", "results", "real"))
os.makedirs(OUTDIR, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def safe_load(p):
    # words files are produced by this harness and contain only tensors/primitives
    return torch.load(p, weights_only=True, map_location="cpu")


# ------------------------------------------------------------------ MODE=words
def build_words():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, output_hidden_states=True).eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pat = re.compile(r"^ [a-z]{3,}$")
    words = []                                    # (token_id, first_letter)
    n_ids = len(tok) if not SMOKE else 40000
    for tid in range(n_ids):
        s = tok.decode([tid])
        if pat.match(s):
            words.append((tid, s.strip()[0]))
    print(f"word tokens (raw): {len(words)}", flush=True)
    # Registered stratified subsample (prereg Gate 3): cap at WORD_CAP, seed 0.
    if len(words) > WORD_CAP:
        rng = np.random.default_rng(0)
        letters_all = np.array([l for _, l in words])
        keep = []
        uniq = sorted(set(letters_all))
        per = WORD_CAP // len(uniq)
        for L in uniq:
            idx = np.where(letters_all == L)[0]
            take = idx if len(idx) <= per else rng.choice(idx, size=per, replace=False)
            keep.extend(take.tolist())
        # top up to WORD_CAP from the remainder, uniformly
        rest = np.setdiff1d(np.arange(len(words)), np.array(keep))
        extra = WORD_CAP - len(keep)
        if extra > 0 and len(rest) > 0:
            keep.extend(rng.choice(rest, size=min(extra, len(rest)), replace=False).tolist())
        keep = sorted(keep)
        words = [words[i] for i in keep]
    print(f"word tokens (kept): {len(words)}", flush=True)
    ids = torch.tensor([[bos, t] for t, _ in words])
    acts = {ly: [] for ly in LAYERS}
    with torch.no_grad():
        for i in range(0, len(ids), 256):
            hs = model(ids[i:i+256]).hidden_states
            for ly in LAYERS:
                acts[ly].append(hs[ly + 1][:, 1, :].float().cpu())
            if i % 2048 == 0:
                print(f"  acts {i}/{len(ids)}", flush=True)
    letters = [l for _, l in words]
    for ly in LAYERS:
        out = os.path.join(OUTDIR, f"words_gemma-3-1b_L{ly}.pt")
        torch.save(dict(acts=torch.cat(acts[ly]), token_ids=[t for t, _ in words],
                        letters=letters, model=MODEL, layer=ly,
                        hook=f"model.layers.{ly}.output (hidden_states[{ly+1}])",
                        tok_len=len(tok)), out)
        print(f"saved {out}", flush=True)


# ------------------------------------------------------------------ MODE=score
def probe_oof(X, y, C=1.0, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(y)); nf = 0
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        if len(set(y[tr])) < 2:
            continue
        lr = LogisticRegression(C=C, max_iter=200, class_weight="balanced")
        lr.fit(X[tr], y[tr]); oof[te] = lr.predict_proba(X[te])[:, 1]
        nf += 1
    return oof


def build_probes(Xr, letters):
    """SAE-independent: present mask (out-of-fold) + full-fit retention probe."""
    from sklearn.linear_model import LogisticRegression
    probes = {}
    for L in sorted(set(letters)):
        yL = (letters == L).astype(int)
        if yL.sum() < MIN_WORDS or (1 - yL).sum() < MIN_WORDS:
            continue
        oof = probe_oof(Xr, yL, C=PROBE_C)
        lrf = LogisticRegression(C=PROBE_C, max_iter=200,
                                 class_weight="balanced").fit(Xr, yL)
        probes[L] = (yL, oof > 0.5, lrf)
        print(f"  probe {L}: n={int(yL.sum())}", flush=True)
    return probes


def load_sae(d):
    """Load a Gemma Scope 2 params.safetensors + config.json; orient weights."""
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(d, "config.json")))
    P = load_file(os.path.join(d, "params.safetensors"))
    keys = {k.lower().split(".")[-1]: k for k in P}
    W_enc = P[keys.get("w_enc", keys.get("wenc"))].float()
    W_dec = P[keys.get("w_dec", keys.get("wdec"))].float()
    b_enc = P[keys.get("b_enc", keys.get("benc"))].float()
    b_dec = P[keys.get("b_dec", keys.get("bdec"))].float()
    thr = P[keys.get("threshold", keys.get("log_threshold"))].float()
    if "log_threshold" in keys and "threshold" not in keys:
        thr = thr.exp()
    m = int(cfg["width"]); dm = int(b_dec.shape[0])
    if W_enc.shape != (dm, m):
        W_enc = W_enc.T.contiguous()
    if W_dec.shape != (m, dm):
        W_dec = W_dec.T.contiguous()
    assert W_enc.shape == (dm, m) and W_dec.shape == (m, dm) and thr.shape[0] == m, \
        f"orientation failure in {d}: {W_enc.shape} {W_dec.shape} {thr.shape}"
    return dict(W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec, thr=thr,
                cfg=cfg, m=m, d=dm, sha=sha256(os.path.join(d, "params.safetensors")))


def rates_from_fires(fires, letters, probes, retained_by_L):
    """13a-semantics SINGLE + FAMILY endpoints from a boolean fire matrix.
    sel is computed from column sums (algebraically identical to
    fires[yL==1].mean(0) - fires[yL==0].mean(0)) to avoid materialising the
    ~N x m complement subset at width 262k."""
    per_letter, tot = {}, dict(present=0, abs_single=0, abs_fam=0)
    N = fires.shape[0]
    sum_all = fires.sum(0, dtype=np.float64)
    for L, (yL, present_all, _lrf) in probes.items():
        nL = int(yL.sum())
        sum_L = fires[yL == 1].sum(0, dtype=np.float64)
        sel = sum_L / nL - (sum_all - sum_L) / (N - nL)
        j = int(sel.argmax())
        Lw = np.where(yL == 1)[0]
        present = present_all[Lw]; retained = retained_by_L[L][Lw]
        npres = int(present.sum())
        if sel[j] < TAU:
            # tau-WAIVED sensitivity fields (GPT pre-lock P1.1 / evaluator D4):
            # the argmax singleton stands in for the family so a letter that
            # loses its clean latent in one cell still yields a defined rate.
            missw = present & (~fires[Lw, j]) & retained
            per_letter[L] = dict(clean_latent=False, sel=round(float(sel[j]), 3),
                                 letter_present=npres,
                                 rate_family_waived=round(float(missw.sum() / max(npres, 1)), 4))
            continue
        fam_uncapped = np.where(sel >= TAU)[0]
        fam = fam_uncapped
        if len(fam) > FAM_CAP:
            fam = fam[np.argsort(sel[fam])[::-1][:FAM_CAP]]
        miss_single = present & (~fires[Lw, j]) & retained
        miss_fam = present & (~fires[np.ix_(Lw, fam)].any(axis=1)) & retained
        per_letter[L] = dict(clean_latent=True, latent=int(j),
                             sel=round(float(sel[j]), 3), fam_size=int(len(fam)),
                             fam_size_uncapped=int(len(fam_uncapped)),
                             cap_hit=bool(len(fam_uncapped) > FAM_CAP),
                             letter_present=npres,
                             absorbed_single=int(miss_single.sum()),
                             absorbed_family=int(miss_fam.sum()),
                             rate_single=round(float(miss_single.sum() / max(npres, 1)), 4),
                             rate_family=round(float(miss_fam.sum() / max(npres, 1)), 4))
        tot["present"] += npres
        tot["abs_single"] += int(miss_single.sum())
        tot["abs_fam"] += int(miss_fam.sum())
    return per_letter, tot


def pick_encoder_variant(s, X0):
    """BLIND infra rule (prereg amendment-safe): Gemma Scope 1's published
    encoder takes RAW inputs (pre = x@W_enc + b_enc); Anthropic-style SAEs
    center by b_dec first. Scope 2's convention is not documented in config.
    Select the variant whose L0 on one batch is closest to the config's trained
    l0 — decided by sparsity conformance only, blind to every endpoint."""
    outs = {}
    with torch.no_grad():
        for name, xb in (("raw", X0), ("centered", X0 - s["b_dec"])):
            pre = xb @ s["W_enc"] + s["b_enc"]
            f = torch.relu(pre) * (pre > s["thr"]).float()
            outs[name] = float((f > 0).float().sum(1).mean())
    cfg_l0 = float(s["cfg"].get("l0") or 0)
    pick = min(outs, key=lambda k: abs(outs[k] - cfg_l0)) if cfg_l0 > 0 else "raw"
    return pick, outs


def score_one(d, Xr, letters, probes):
    s = load_sae(d)
    N = Xr.shape[0]; m = s["m"]
    keep_mag = m <= GRID_MAX_WIDTH
    fires = np.zeros((N, m), dtype=bool)
    Fmag = np.zeros((N, m), dtype=np.float16) if keep_mag else None
    Xhat = np.zeros((N, s["d"]), dtype=np.float32)
    l0_sum = 0.0; sse = 0.0
    X = torch.tensor(Xr)
    variant, l0_probe = pick_encoder_variant(s, X[:min(1024, N)])
    print(f"  encoder variant={variant} (batch L0 raw={l0_probe['raw']:.1f} "
          f"centered={l0_probe['centered']:.1f} cfg={s['cfg'].get('l0')})", flush=True)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            xb = X[i:i+BATCH]
            xe = xb - s["b_dec"] if variant == "centered" else xb
            pre = xe @ s["W_enc"] + s["b_enc"]
            f = torch.relu(pre) * (pre > s["thr"]).float()
            fires[i:i+BATCH] = (f > 0).numpy()
            if keep_mag:
                Fmag[i:i+BATCH] = f.numpy().astype(np.float16)
            xh = f @ s["W_dec"] + s["b_dec"]      # reconstruction of the RAW input
            Xhat[i:i+BATCH] = xh.numpy()
            l0_sum += float((f > 0).sum())
            sse += float(((xb - xh) ** 2).sum())  # FVU always in raw space
    measured_l0 = l0_sum / N
    ss_tot = float(((Xr - Xr.mean(0)) ** 2).sum())
    fvu = sse / max(ss_tot, 1e-9)
    # retention probes on THIS SAE's reconstruction (13a semantics)
    retained_by_L = {L: lrf.predict_proba(Xhat)[:, 1] > 0.5
                     for L, (_y, _p, lrf) in probes.items()}
    per_letter, tot = rates_from_fires(fires, letters, probes, retained_by_L)
    # descriptive theta grid (widths <= GRID_MAX_WIDTH only; prereg D3)
    grid = {}
    if keep_mag:
        for t in GRID:
            pl_t, tot_t = rates_from_fires(Fmag > np.float16(t),
                                           letters, probes, retained_by_L)
            grid[f"{t:g}"] = round(tot_t["abs_fam"] / max(tot_t["present"], 1), 4)
    name = os.path.basename(os.path.normpath(d))
    row = dict(sae=name, sha256=s["sha"], arch="jumprelu", enc_variant=variant,
               width=m, layer=LAYER_FROM_NAME(name), l0_tag=L0_TAG(name),
               config_l0=s["cfg"].get("l0"), measured_l0=round(measured_l0, 2),
               fvu=round(fvu, 4), model=s["cfg"].get("model_name"),
               hook=s["cfg"].get("hf_hook_point_in"),
               theta=THETA, tau=TAU, fam_cap=FAM_CAP,
               rate_single=round(tot["abs_single"] / max(tot["present"], 1), 4),
               rate_family=round(tot["abs_fam"] / max(tot["present"], 1), 4),
               grid_family=grid, per_letter=per_letter)
    del fires, Fmag, Xhat
    return row


def LAYER_FROM_NAME(n):
    m_ = re.search(r"layer_(\d+)", n)
    return int(m_.group(1)) if m_ else None


def L0_TAG(n):
    m_ = re.search(r"l0_(small|medium|big)", n)
    return m_.group(1) if m_ else None


def score_all():
    # Registered theta is 0 and the fire matrix is computed as f > 0; refuse to
    # run with any other value rather than record a theta the code ignores
    # (GPT pre-lock P3.2).
    assert THETA == 0.0, f"locked run requires THETA=0, got {THETA}"
    words_dir = os.environ["WORDS_DIR"]
    cache = {}   # layer -> (Xr, letters, probes)

    def layer_ctx(ly):
        if ly not in cache:
            W = safe_load(os.path.join(words_dir, f"words_gemma-3-1b_L{ly}.pt"))
            Xr = W["acts"].numpy().astype("float32")
            letters = np.array(W["letters"])
            print(f"layer {ly}: words={Xr.shape} letters={len(set(letters))}; "
                  f"fitting SAE-independent probes...", flush=True)
            cache[ly] = (Xr, letters, build_probes(Xr, letters))
        return cache[ly]

    rows = []
    for d in os.environ["SAE_DIRS"].split(","):
        d = d.strip()
        if not d:
            continue
        ly = LAYER_FROM_NAME(os.path.basename(os.path.normpath(d)))
        assert ly is not None, f"cannot parse layer from {d}"
        Xr, letters, probes = layer_ctx(ly)
        print(f"scoring {d} (layer {ly} activations)", flush=True)
        r = score_one(d, Xr, letters, probes)
        rows.append(r)
        print(f"  {r['sae']}: single={r['rate_single']:.4f} family={r['rate_family']:.4f} "
              f"L0={r['measured_l0']:.1f}/cfg{r['config_l0']} fvu={r['fvu']:.3f}", flush=True)
    out = os.environ.get("OUT", os.path.join(OUTDIR, "round15_rows.json"))
    json.dump(rows, open(out, "w"), indent=1)
    print(f"wrote {out} ({len(rows)} SAEs)", flush=True)


if __name__ == "__main__":
    if MODE == "words":
        build_words()
    else:
        score_all()
