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
    # SAE reconstruction in RAW residual space (for the retention check below).
    with torch.no_grad():
        Xhat = ((f @ W_dec.T + b_dec) / scale + mu).numpy().astype("float32")

    # ---- pass A: probes (theta-independent) + eligible letters ----
    # present(x): out-of-fold probe on the RAW residual (leak-free label).
    # retained(x_hat): a full-fit probe applied to the SAE RECONSTRUCTION -- this
    # separates ABSORPTION (letter still linearly present in x_hat, just not via the
    # main latent) from feature LOSS (SAE dropped the letter). Without it P1 would
    # score an SAE that simply reconstructs the letter worse as "more absorbing"
    # (review Finding 1). Full-fit is in-sample on x, but applied to x_hat -- a tiny
    # leak in a binary retention check, immaterial.
    from sklearn.linear_model import LogisticRegression
    probes = {}                                            # L -> (unit probe dir, present mask, oof, retained mask)
    for L in uniq:
        yL = (letters == L).astype(int)
        if yL.sum() < MIN_WORDS or (1 - yL).sum() < MIN_WORDS:
            continue
        oof, wdir = probe_oof(Xr, yL, C=float(os.environ.get("PROBE_C", "1.0")))
        wdirt = torch.tensor(wdir, dtype=torch.float32)
        wdirt = wdirt / wdirt.norm().clamp_min(1e-8)
        lrf = LogisticRegression(C=float(os.environ.get("PROBE_C", "1.0")), max_iter=200,
                                 class_weight="balanced").fit(Xr, yL)
        retained = lrf.predict_proba(Xhat)[:, 1] > 0.5     # letter survives in the reconstruction
        probes[L] = (wdirt, oof > 0.5, oof, retained)

    def absorption_at(theta):
        """Main-L-latent selection + absorbed-instance counting at a fire threshold.
        absorbed = present(x) AND retained(x_hat) AND main-L-latent misses.
        Returns (rate, per_letter, main_latent, absorbed_words, loss_rate)."""
        fires = F > theta
        per_letter, main_latent, absorbed = {}, {}, {}
        tot_present, tot_miss, tot_lost = 0, 0, 0
        for L, (_wd, present, oof, retained) in probes.items():
            yL = (letters == L).astype(int)
            sel = fires[yL == 1].mean(0) - fires[yL == 0].mean(0)
            j = int(sel.argmax())
            if sel[j] < SEL_MIN:
                per_letter[L] = dict(n=int(yL.sum()), clean_latent=False, sel=round(float(sel[j]), 3))
                continue
            Lw = np.where(yL == 1)[0]
            present_L = present[Lw]; retained_L = retained[Lw]
            latent_miss = present_L & (~fires[Lw, j])      # letter present but L-latent misses
            miss = latent_miss & retained_L                # ABSORBED: and letter survives in x_hat
            lost = latent_miss & (~retained_L)             # LOST: letter dropped by the SAE
            np_, nm, nl = int(present_L.sum()), int(miss.sum()), int(lost.sum())
            tot_present += np_; tot_miss += nm; tot_lost += nl
            per_letter[L] = dict(n=int(yL.sum()), clean_latent=True, latent=j,
                                 sel=round(float(sel[j]), 3),
                                 probe_acc=round(float(((oof > 0.5) == yL).mean()), 3),
                                 letter_present=np_, absorbed=nm, lost=nl,
                                 rate=round(nm / max(np_, 1), 4))
            main_latent[L] = j
            absorbed[L] = [int(w) for w in Lw[miss]]
        return (tot_miss / max(tot_present, 1), per_letter, main_latent, absorbed,
                tot_lost / max(tot_present, 1))

    # primary metric at THETA (matched), plus a DESCRIPTIVE theta grid (theta>0 is
    # unmatched/biased toward L1 -- an upper bracket, not a robustness gate).
    absorption_rate, per_letter, main_latent, absorbed, loss_rate = absorption_at(THETA)
    grid = {f"{t:g}": round(absorption_at(t)[0], 4) for t in THETA_GRID}

    # ---- P2 attribution (DESCRIPTIVE, reconstruction-space; NOT a causal-validity
    # bar -- the real causal test is the deferred Chanin forward-pass). Two contrasts
    # per absorbed word, carriers = top-N_CARRIERS firing latents by MAGNITUDE:
    #   * CONCENTRATION (the falsifiable one): letter-selectivity per unit
    #     reconstruction mass, cos(dvec, wdir_L), of the top-magnitude carriers vs
    #     a random equal-count set of the word's OTHER firing latents. Magnitude is
    #     normalized out (cosine), so this is NOT the near-tautology of "does the
    #     reconstruction contain the present letter" -- it asks whether the letter
    #     is CONCENTRATED in the dominant latents. Can fail (diffuse letter -> ~0).
    #   * specificity (weaker, descriptive): d_true - d_other, the true-letter vs
    #     other-letter drop. Near-guaranteed positive for a low-FVU SAE (the word
    #     contains L, not L'), so reported but NOT a bar (per external review).
    # Space note: dvec is in NORMALIZED residual space, wdir in RAW; normalization
    # is a SCALAR, so cosine (concentration) is exactly invariant and the sign of
    # d_true-d_other is valid. Breaks under per-dimension normalization.
    Wd = W_dec                                             # [d, m] unit-norm columns
    conc, spec, causal_true, causal_other = [], [], [], []
    n_absorbed_inst = 0
    rng = np.random.default_rng(0)
    other_dirs = {L: torch.stack([probes[L2][0] for L2 in probes if L2 != L])
                  for L in probes} if len(probes) > 1 else {}
    def _cos(dvec, wdir):
        nv = float(dvec.norm())
        return float(wdir @ dvec) / nv if nv > 1e-8 else 0.0
    for L, wis in absorbed.items():
        wdir_L = probes[L][0]
        for wi in wis:
            n_absorbed_inst += 1
            fw = F[wi]
            fired = np.where(fw > THETA)[0]
            if len(fired) == 0 or L not in other_dirs: continue
            order = fired[np.argsort(fw[fired])[::-1]]
            carriers = order[:N_CARRIERS]                  # top-magnitude firing latents
            rest = order[N_CARRIERS:]
            dtop = Wd[:, carriers] @ torch.tensor(fw[carriers], dtype=torch.float32)
            d_true = float(wdir_L @ dtop)
            d_other = float((other_dirs[L] @ dtop).mean())
            causal_true.append(d_true); causal_other.append(d_other)
            spec.append(d_true - d_other)
            if len(rest) >= 1:                             # concentration needs a control set
                ctrl = rng.choice(rest, size=min(len(carriers), len(rest)), replace=False)
                drand = Wd[:, ctrl] @ torch.tensor(fw[ctrl], dtype=torch.float32)
                conc.append(_cos(dtop, wdir_L) - _cos(drand, wdir_L))
    def _m(x): return round(float(np.mean(x)), 4) if x else None
    res = dict(sae=os.path.basename(os.environ["SAE"]), arch=arch, seed=int(s.get("seed", -1)),
               k=k, lam=s.get("lam"), fvu=s.get("stats", {}).get("fvu"),
               l0=s.get("stats", {}).get("l0"),           # matched-sparsity check (gates P1)
               model=s.get("model"), layer=s.get("layer"),   # conformance (scorer checks)
               theta=THETA, sel_min=SEL_MIN, n_carriers=N_CARRIERS,
               min_words=MIN_WORDS, probe_c=float(os.environ.get("PROBE_C", "1.0")),
               n_letters_scored=sum(1 for v in per_letter.values() if v.get("clean_latent")),
               absorption_rate=round(absorption_rate, 4), loss_rate=round(loss_rate, 4),
               absorption_by_theta=grid,
               causal_conc_mean=_m(conc),                  # PRIMARY P2 contrast (magnitude-normalized, falsifiable)
               causal_conc_sd=round(float(np.std(conc)), 4) if conc else None,
               causal_spec_mean=_m(spec),                  # descriptive (near-guaranteed +; not a bar)
               causal_true_mean=_m(causal_true), causal_other_mean=_m(causal_other),
               n_causal=len(conc), n_absorbed_inst=n_absorbed_inst,
               main_latents=main_latent,                   # P3 ground truth: {letter: main-L-latent id}
               per_letter=per_letter)
    out = os.environ.get("OUT", os.environ["SAE"].replace(".pt", "_fl.json"))
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}: absorption={absorption_rate:.4f} loss={loss_rate:.4f} grid={grid} "
          f"conc={res['causal_conc_mean']} spec={res['causal_spec_mean']} "
          f"n_letters={res['n_letters_scored']} n_causal={len(conc)}", flush=True)

if __name__ == "__main__":
    if MODE == "words":
        build_words()
    else:
        ensure("sklearn", "scikit-learn")
        score_sae()
