"""Round 15 Amendment 1: in-distribution implementation oracle.

Rationale (pilot, 2026-07-28, BEFORE any registered number): the SMOKE pilot
measured word-set FVU 0.503 for the 16k cell while this in-distribution check
gives FVU 0.0419 on real corpus sequences — the word-set FVU conflates domain
shift with implementation error. Gate 1's implementation half therefore runs on
in-distribution sequences (from the suite's own examples.safetensors token
corpus); the word-set FVU is reported descriptively.

Evaluates the three width-series SAEs (layer 13) on N_SEQ corpus sequences and
writes {sae_name: {l0_indist, fvu_indist}} to OUT. Blind to every endpoint.

Env: SAE_ROOT=~/r15/sae  OUT=~/r15/results/indist.json  N_SEQ=4
     MODEL=unsloth/gemma-3-1b-pt
"""
import os, json
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

MODEL = os.environ.get("MODEL", "unsloth/gemma-3-1b-pt")
N_SEQ = int(os.environ.get("N_SEQ", "4"))
ROOT = os.path.expanduser(os.environ.get("SAE_ROOT", "~/r15/sae"))
OUT = os.path.expanduser(os.environ.get("OUT", "~/r15/results/indist.json"))
CELLS = ["layer_13_width_16k_l0_medium", "layer_13_width_65k_l0_medium",
         "layer_13_width_262k_l0_medium"]

p = hf_hub_download("google/gemma-scope-2-1b-pt",
                    "resid_post/layer_13_width_16k_l0_medium/examples.safetensors")
with safe_open(p, framework="pt") as f:
    toks = f.get_slice("tokens")[:N_SEQ]
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, output_hidden_states=True).eval()
with torch.no_grad():
    hs = model(torch.as_tensor(toks).long()).hidden_states[14]  # layers.13.output
X = hs[:, 1:, :].reshape(-1, hs.shape[-1]).float()

res = {}
for c in CELLS:
    P = load_file(os.path.join(ROOT, c, "params.safetensors"))
    k = {n.lower(): n for n in P}
    W_enc = P[k["w_enc"]].float(); W_dec = P[k["w_dec"]].float()
    b_enc = P[k["b_enc"]].float(); b_dec = P[k["b_dec"]].float()
    thr = P[k["threshold"]].float()
    if W_enc.shape[0] != X.shape[1]:
        W_enc = W_enc.T.contiguous()
    if W_dec.shape[1] != X.shape[1]:
        W_dec = W_dec.T.contiguous()
    fvu_num = 0.0
    l0_sum = 0.0
    with torch.no_grad():
        for i in range(0, X.shape[0], 512):
            xb = X[i:i+512]
            pre = xb @ W_enc + b_enc
            fq = torch.relu(pre) * (pre > thr).float()
            xh = fq @ W_dec + b_dec
            fvu_num += float(((xb - xh) ** 2).sum())
            l0_sum += float((fq > 0).sum())
    fvu = fvu_num / float(((X - X.mean(0)) ** 2).sum())
    res[c] = dict(l0_indist=round(l0_sum / X.shape[0], 2), fvu_indist=round(fvu, 4))
    print(c, res[c], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
