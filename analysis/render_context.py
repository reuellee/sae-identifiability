"""Render readable context windows for the natfeat top-activating positions.

Local GPT-2 byte-level detokenizer (no transformers dependency): uses the saved
token ids (experiments/token_data.json) + gpt2_encoder.json to reconstruct
multi-byte characters that individual-token decode renders as U+FFFD. Also
quantifies each candidate latent's non-ASCII / byte-fragment firing fraction.

Reads results/round8/natfeat_adjudication.json (top positions per latent),
writes results/round8/natfeat_context.md.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
EXP = os.path.join(ROOT, "experiments")
R8 = os.path.join(ROOT, "results", "round8")


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


enc = json.load(open(os.path.join(EXP, "gpt2_encoder.json")))
id2piece = {v: k for k, v in enc.items()}
u2b = bytes_to_unicode()
td = json.load(open(os.path.join(EXP, "token_data.json")))
IDS = td["ids"]


def detok(id_list):
    pieces = "".join(id2piece[i] for i in id_list)
    bys = bytes(u2b[c] for c in pieces)
    return bys.decode("utf-8", errors="replace")


def window(pos, left=6, right=4):
    lo, hi = max(0, pos - left), min(len(IDS), pos + right + 1)
    pre = detok(IDS[lo:pos])
    cur = detok([IDS[pos]])
    post = detok(IDS[pos + 1:hi])
    return f"...{pre}⟦{cur}⟧{post}..."


def nonascii_frac(top_tokens):
    """Fraction of top positions whose single token is non-ASCII / byte-frag."""
    n = 0
    for t in top_tokens:
        s = t["tok"]
        if any(ord(ch) > 127 for ch in s) or "�" in s:
            n += 1
    return round(n / max(len(top_tokens), 1), 3)


d = json.load(open(os.path.join(R8, "natfeat_adjudication.json")))
lines = ["# Natural-feature adjudication — readable context windows",
         "",
         "Reconstructed with a local GPT-2 byte-level detokenizer from saved",
         "token ids (individual-token decode renders multi-byte chars as �).",
         "`⟦tok⟧` marks the firing token; `naf` = non-ASCII/byte-fragment fraction",
         "of the top-50 activating tokens.", ""]

for M in ("128", "256"):
    lines.append(f"## m = {M}\n")
    for c in d["widths"][M]["clusters"]:
        cid = c["cluster_id"]
        naf_p = nonascii_frac(c["parent_top_tokens"])
        naf_c = nonascii_frac(c["child_top_tokens"])
        lines.append(
            f"### c{cid}  P=lat{c['parent_latent']} (r={c['rate_parent']}, "
            f"naf={naf_p})  C=lat{c['child_latent']} (r={c['rate_child']}, "
            f"naf={naf_c})  cos={c['cos_decoder']}  "
            f"C(P|c)={c['C_parent_given_child']} C(c|P)={c['C_child_given_parent']}")
        for name, key in (("PARENT", "parent_top_tokens"),
                          ("CHILD", "child_top_tokens"),
                          ("RESID", "residual_top_tokens")):
            lines.append(f"- **{name}**:")
            for t in c[key][:6]:
                lines.append(f"    - `{window(t['pos'])}`  (act {t['act']})")
        lines.append("")

out = os.path.join(R8, "natfeat_context.md")
open(out, "w").write("\n".join(lines))
print("wrote", out)
# also print a compact naf summary for the run log
print("\nnon-ASCII/byte-fragment fraction (top-50) per latent:")
for M in ("128", "256"):
    for c in d["widths"][M]["clusters"]:
        print(f"  m{M} c{c['cluster_id']}: P lat{c['parent_latent']} "
              f"naf={nonascii_frac(c['parent_top_tokens'])}  "
              f"C lat{c['child_latent']} naf={nonascii_frac(c['child_top_tokens'])}")
