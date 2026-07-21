"""Extract GPT-2-small layer-6 residual-stream activations for the weighting POC.
Corpus: Project Gutenberg text (no HF datasets dependency). Saves fp16 tensor.
"""
import torch, urllib.request, os, sys

N_TOKENS = 500_000
LAYER = 6
OUT = "activations_l6.pt"

def main():
    from transformers import GPT2Tokenizer, GPT2Model
    urls = ["https://www.gutenberg.org/files/2600/2600-0.txt",     # War and Peace
            "https://www.gutenberg.org/files/1661/1661-0.txt"]     # Sherlock Holmes
    text = ""
    for u in urls:
        try:
            text += urllib.request.urlopen(u, timeout=60).read().decode("utf-8", "ignore")
            print(f"fetched {u}: total chars {len(text)}", flush=True)
        except Exception as e:
            print(f"fetch failed {u}: {e}", flush=True)
        if len(text) > 3_500_000: break
    if len(text) < 500_000:
        print("FATAL: no corpus"); sys.exit(1)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").cuda().eval()
    ids = tok(text, return_tensors=None)["input_ids"]
    print(f"corpus tokens: {len(ids)}", flush=True)
    chunks = [ids[i:i + 512] for i in range(0, len(ids) - 512, 512)]
    acts = []
    got = 0
    with torch.no_grad():
        for i in range(0, len(chunks), 16):
            batch = torch.tensor(chunks[i:i + 16]).cuda()
            hs = model(batch, output_hidden_states=True).hidden_states[LAYER]
            acts.append(hs.reshape(-1, 768).half().cpu())
            got += acts[-1].shape[0]
            if got >= N_TOKENS: break
            if i % 160 == 0: print(f"  {got}/{N_TOKENS} tokens", flush=True)
    A = torch.cat(acts)[:N_TOKENS]
    torch.save(A, OUT)
    print(f"saved {A.shape} fp16 -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
