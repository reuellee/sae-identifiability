"""Extract residual-stream activations from a real LM for SAE training.

Real-model scaling of the absorption program (round 11 infra): the toy/GPT-2
work graduates to a real model. Model-agnostic via env vars; default
Pythia-1.4B (open, no license gate, d_model=2048, 24 layers).

Env: MODEL (default EleutherAI/pythia-1.4b), LAYER (default 12),
     N_TOKENS (default 2_000_000; SMOKE tiny), SEQ (default 512),
     OUT (default results/real/acts_<model>_L<layer>.pt).
Saves fp16 activations [N, d], the token ids [N], and metadata. Token ids are
kept so feature/first-letter analyses can label latents later.

Corpus: Project Gutenberg English text (dependency-light, as extract_activations.py).
"""
import os, sys, urllib.request, time
import torch

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
MODEL = os.environ.get("MODEL", "EleutherAI/pythia-1.4b" if not SMOKE else "EleutherAI/pythia-70m")
LAYER = int(os.environ.get("LAYER", "12" if not SMOKE else "3"))
N_TOKENS = int(os.environ.get("N_TOKENS", "20000" if SMOKE else "2000000"))
SEQ = int(os.environ.get("SEQ", "512"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results", "real")
os.makedirs(OUTDIR, exist_ok=True)
_safe = MODEL.split("/")[-1]
OUT = os.environ.get("OUT", os.path.join(OUTDIR, f"acts_{_safe}_L{LAYER}.pt"))

def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q", pkg], check=True)

def get_corpus(min_chars):
    urls = ["https://www.gutenberg.org/files/2600/2600-0.txt",   # War and Peace
            "https://www.gutenberg.org/files/1661/1661-0.txt",   # Sherlock Holmes
            "https://www.gutenberg.org/files/98/98-0.txt",       # Tale of Two Cities
            "https://www.gutenberg.org/files/1342/1342-0.txt"]   # Pride and Prejudice
    text = ""
    for u in urls:
        try:
            text += urllib.request.urlopen(u, timeout=60).read().decode("utf-8", "ignore")
            print(f"fetched {u}: total {len(text)} chars", flush=True)
        except Exception as e:
            print(f"fetch failed {u}: {e}", flush=True)
        if len(text) > min_chars:
            break
    return text

def main():
    ensure("transformers")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={MODEL} layer={LAYER} n_tokens={N_TOKENS} device={dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        output_hidden_states=True).to(dev).eval()
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    assert 0 <= LAYER <= n_layers, f"layer {LAYER} out of range (0..{n_layers})"
    text = get_corpus(min_chars=max(500_000, N_TOKENS * 6))
    ids = tok(text, return_tensors=None)["input_ids"]
    print(f"corpus tokens: {len(ids)}; hidden_size={d}; layers={n_layers}", flush=True)
    chunks = [ids[i:i + SEQ] for i in range(0, len(ids) - SEQ, SEQ)]
    acts_buf, tok_buf, got, t0 = [], [], 0, time.time()
    bs = 8
    with torch.no_grad():
        for c0 in range(0, len(chunks), bs):
            batch = chunks[c0:c0 + bs]
            x = torch.tensor(batch, device=dev)
            hs = model(x).hidden_states[LAYER]           # [b, SEQ, d]
            acts_buf.append(hs.reshape(-1, d).to(torch.float16).cpu())
            tok_buf.append(x.reshape(-1).cpu())
            got += x.numel()
            if (c0 // bs) % 20 == 0:
                print(f"  {got}/{N_TOKENS} tokens ({time.time()-t0:.0f}s)", flush=True)
            if got >= N_TOKENS:
                break
    acts = torch.cat(acts_buf)[:N_TOKENS]
    toks = torch.cat(tok_buf)[:N_TOKENS]
    torch.save(dict(acts=acts, tokens=toks, model=MODEL, layer=LAYER,
                    d=d, tokenizer=MODEL), OUT)
    print(f"saved {OUT}: acts {tuple(acts.shape)} fp16 "
          f"({acts.numel()*2/1e9:.1f} GB), {time.time()-t0:.0f}s total", flush=True)

if __name__ == "__main__":
    main()
