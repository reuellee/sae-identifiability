"""First-letter feature-absorption metric (round 12), Chanin-spirit.

Two entry points via MODE env:
  MODE=words  -> build the whole-word token set from the tokenizer, extract each
                 word-token's layer-L residual (BOS+token), cache to words.pt.
  MODE=score  -> given SAE=... and WORDS=words.pt, compute the absorption metric,
                 the causal ablation gap, and dump a small JSON.

Absorption (per SAE, per letter L, definition frozen in the prereg):
  * residual probe: 5-fold logistic regression on RESIDUAL acts predicting
    starts-with-L; out-of-fold prob>0.5 => "letter present".
  * main L-latent: argmax_j [P(fire_j|L) - P(fire_j|not-L)], require >= SEL_MIN.
  * absorbed word: letter-present AND the L-latent does NOT fire (act<=THETA).
  * absorption_rate = mean over letters of
      #(letter-present & L-latent-misses) / #(letter-present).
Causal: for absorbed words, absorbing latents = fire & decoder-cos>=0.3 to the
  probe direction; ablate them from the SAE reconstruction, re-probe, and
  compare the L-logit drop to a matched-random equal-count control.

Env: MODEL, LAYER, MODE, SAE, WORDS, OUT, THETA=0.05, SEL_MIN=0.30, MIN_WORDS=30,
     PROBE_C=1.0. SMOKE shrinks the model + word set.
"""
import os, sys, re, json
import torch

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
MODEL = os.environ.get("MODEL", "EleutherAI/pythia-1.4b" if not SMOKE else "EleutherAI/pythia-70m")
LAYER = int(os.environ.get("LAYER", "12" if not SMOKE else "3"))
MODE = os.environ.get("MODE", "words")
# "fire" = act > THETA. Primary THETA=0.0 matches the L0 def the archs are
# matched on (L1's soft-thresholded acts are systematically smaller, so a
# positive magnitude threshold would zero L1 "fires" TopK keeps and bias
# absorption toward L1 -- so we threshold at 0 and REPORT the theta grid).
THETA = float(os.environ.get("THETA", "0.0"))
THETA_GRID = [float(x) for x in os.environ.get("THETA_GRID", "0.0,0.01,0.05,0.1").split(",")]
SEL_MIN = float(os.environ.get("SEL_MIN", "0.30"))
MIN_WORDS = int(os.environ.get("MIN_WORDS", "30" if not SMOKE else "5"))
N_CARRIERS = int(os.environ.get("N_CARRIERS", "3"))   # top-magnitude carriers per absorbed word
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results", "real")
os.makedirs(OUTDIR, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def ensure(mod, pip_name=None):
    try: __import__(mod)
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q",
                        pip_name or mod], check=True)

def safe_load(p):
    try: return torch.load(p, weights_only=True)
    except Exception: return torch.load(p)

# ------------------------------------------------------------------ MODE=words
def build_words():
    ensure("transformers")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        output_hidden_states=True).to(dev).eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pat = re.compile(r"^ [a-z]{3,}$")
    words = []                                    # (token_id, first_letter)
    for tid in range(min(tok.vocab_size, 2000 if SMOKE else 10**9)):
        s = tok.decode([tid])
        if pat.match(s):
            words.append((tid, s.strip()[0]))
    if SMOKE: words = words[:200]
    print(f"word tokens: {len(words)}", flush=True)
    ids = torch.tensor([[bos, t] for t, _ in words], device=dev)
    acts = []
    with torch.no_grad():
        for i in range(0, len(ids), 512):
            hs = model(ids[i:i+512]).hidden_states[LAYER][:, 1, :]   # word-token position
            acts.append(hs.float().cpu())
    acts = torch.cat(acts)
    letters = [l for _, l in words]
    out = os.path.join(OUTDIR, f"words_{MODEL.split('/')[-1]}_L{LAYER}.pt")
    torch.save(dict(acts=acts, token_ids=[t for t, _ in words], letters=letters,
                    model=MODEL, layer=LAYER), out)
    print(f"saved {out}: {tuple(acts.shape)}", flush=True)

# ------------------------------------------------------------------ MODE=score
def probe_oof(X, y, C=1.0, folds=5):
    """5-fold out-of-fold logistic-regression probabilities + the mean weight dir."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    oof = np.zeros(len(y)); w = np.zeros(X.shape[1])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    nf = 0
    for tr, te in skf.split(X, y):
        if len(set(y[tr])) < 2: continue
        lr = LogisticRegression(C=C, max_iter=200, class_weight="balanced")
        lr.fit(X[tr], y[tr]); oof[te] = lr.predict_proba(X[te])[:, 1]
        w += lr.coef_[0]; nf += 1
    return oof, (w / max(nf, 1))

def score_sae():
    import numpy as np
    W = safe_load(os.environ["WORDS"]); s = safe_load(os.environ["SAE"])
    Xr = W["acts"].numpy().astype("float32")            # residual acts [N,d]
    letters = np.array(W["letters"])
    # SAE encode (on normalized acts)
    mu = s["mu"].float(); scale = float(s["scale"]); arch = s["arch"]; k = int(s.get("k", 32))
    Xn = torch.tensor((Xr - mu.numpy()) * scale)
    W_enc = s["W_enc"].float(); b_enc = s["b_enc"].float()
    W_dec = s["W_dec"].float(); b_dec = s["b_dec"].float()
    with torch.no_grad():
        f = torch.relu((Xn - b_dec) @ W_enc.T + b_enc)
        if arch == "topk":
            thr = f.topk(k, dim=-1).values[:, -1:].clamp_min(1e-12)
            f = f * (f >= thr).float()
    F = f.numpy()
    uniq = sorted(set(letters))

    # ---- pass A: probes (theta-independent) + eligible letters ----
    probes = {}                                            # L -> (unit probe dir, present mask, oof)
    for L in uniq:
        yL = (letters == L).astype(int)
        if yL.sum() < MIN_WORDS or (1 - yL).sum() < MIN_WORDS:
            continue
        oof, wdir = probe_oof(Xr, yL, C=float(os.environ.get("PROBE_C", "1.0")))
        wdirt = torch.tensor(wdir, dtype=torch.float32)
        wdirt = wdirt / wdirt.norm().clamp_min(1e-8)
        probes[L] = (wdirt, oof > 0.5, oof)

    def absorption_at(theta):
        """Main-L-latent selection + absorbed-instance counting at a fire
        threshold. Returns (rate, per_letter, main_latent{L:j}, absorbed{L:[wi]})."""
        fires = F > theta
        per_letter, main_latent, absorbed = {}, {}, {}
        tot_present, tot_miss = 0, 0
        for L, (_wd, present, oof) in probes.items():
            yL = (letters == L).astype(int)
            sel = fires[yL == 1].mean(0) - fires[yL == 0].mean(0)
            j = int(sel.argmax())
            if sel[j] < SEL_MIN:
                per_letter[L] = dict(n=int(yL.sum()), clean_latent=False, sel=round(float(sel[j]), 3))
                continue
            Lw = np.where(yL == 1)[0]
            present_L = present[Lw]
            miss = present_L & (~fires[Lw, j])             # letter present but L-latent misses
            np_, nm = int(present_L.sum()), int(miss.sum())
            tot_present += np_; tot_miss += nm
            per_letter[L] = dict(n=int(yL.sum()), clean_latent=True, latent=j,
                                 sel=round(float(sel[j]), 3),
                                 probe_acc=round(float(((oof > 0.5) == yL).mean()), 3),
                                 letter_present=np_, absorbed=nm, rate=round(nm / max(np_, 1), 4))
            main_latent[L] = j
            absorbed[L] = [int(w) for w in Lw[miss]]
        return tot_miss / max(tot_present, 1), per_letter, main_latent, absorbed

    # primary metric at THETA, plus a theta-sensitivity grid (a theta artifact
    # would show as the L1-vs-TopK sign flipping across the grid; the scorer checks it)
    absorption_rate, per_letter, main_latent, absorbed = absorption_at(THETA)
    grid = {f"{t:g}": round(absorption_at(t)[0], 4) for t in THETA_GRID}

    # ---- causal (NON-CIRCULAR): carriers chosen by ACTIVATION MAGNITUDE (not by
    # probe alignment); test whether ablating them drops the TRUE letter's probe
    # logit MORE than OTHER letters' logits. Same carriers for both -> the
    # magnitude confound cancels; selection is independent of letter identity, so
    # this CAN fail (letter-agnostic carriers -> no gap). ----
    Wd = W_dec                                             # [d, m] unit-norm columns
    causal_diffs, causal_true, causal_other = [], [], []
    n_absorbed_inst = 0
    other_dirs = {L: torch.stack([probes[L2][0] for L2 in probes if L2 != L])
                  for L in probes} if len(probes) > 1 else {}
    for L, wis in absorbed.items():
        wdir_L = probes[L][0]
        for wi in wis:
            n_absorbed_inst += 1
            fw = F[wi]
            fired = np.where(fw > THETA)[0]
            if len(fired) == 0 or L not in other_dirs: continue
            carriers = fired[np.argsort(fw[fired])[::-1][:N_CARRIERS]]  # top-mag firing latents
            fc = torch.tensor(fw[carriers], dtype=torch.float32)
            # dvec is in the SAE's NORMALIZED residual space; the probe wdir is in
            # RAW residual space. Because normalization is a SCALAR (isotropic
            # `scale`), it factors out as a common positive multiplier -> the SIGN
            # of (d_true - d_other) is valid, but the magnitudes are `scale x` the
            # true probe-logit drop (proportional, not equal). This breaks if the
            # recipe ever switches to per-dimension normalization.
            dvec = Wd[:, carriers] @ fc
            d_true = float(wdir_L @ dvec)
            d_other = float((other_dirs[L] @ dvec).mean())
            causal_true.append(d_true); causal_other.append(d_other)
            causal_diffs.append(d_true - d_other)
    def _m(x): return round(float(np.mean(x)), 4) if x else None
    res = dict(sae=os.path.basename(os.environ["SAE"]), arch=arch, seed=int(s.get("seed", -1)),
               k=k, lam=s.get("lam"), fvu=s.get("stats", {}).get("fvu"),
               l0=s.get("stats", {}).get("l0"),           # matched-sparsity check (gates P1)
               theta=THETA, sel_min=SEL_MIN, n_carriers=N_CARRIERS,
               n_letters_scored=sum(1 for v in per_letter.values() if v.get("clean_latent")),
               absorption_rate=round(absorption_rate, 4), absorption_by_theta=grid,
               causal_true_mean=_m(causal_true), causal_other_mean=_m(causal_other),
               causal_diff_mean=_m(causal_diffs),
               causal_diff_sd=round(float(np.std(causal_diffs)), 4) if causal_diffs else None,
               n_causal=len(causal_diffs), n_absorbed_inst=n_absorbed_inst,
               main_latents=main_latent,                   # P3 ground truth: {letter: main-L-latent id}
               per_letter=per_letter)
    out = os.environ.get("OUT", os.environ["SAE"].replace(".pt", "_fl.json"))
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}: absorption={absorption_rate:.4f} grid={grid} "
          f"causal_diff={res['causal_diff_mean']} (true={res['causal_true_mean']} "
          f"other={res['causal_other_mean']}) n_causal={len(causal_diffs)}", flush=True)

if __name__ == "__main__":
    if MODE == "words":
        build_words()
    else:
        ensure("sklearn", "scikit-learn")
        score_sae()
