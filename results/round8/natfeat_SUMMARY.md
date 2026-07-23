# Natural-feature adjudication of S1 seed-stable candidates — result

Pre-registered in `notes/prereg-natfeat-adjudication.md` (pre-results lock
`0603d38`). Queue item #1 of `RESEARCH_PLAN.md` ("the program's path to its
first wild-caught absorption pair"). CPU-only on the orchestrator; token
strings regenerated on a throwaway VM with the exact HF GPT-2 tokenizer and
the corpus/chunking of `extract_activations.py`, then all analysis local.
Code: `analysis/natfeat_adjudicate.py` (+ `analysis/render_context.py` for the
byte-level detokenized context windows). Raw evidence:
`natfeat_adjudication.json`, `natfeat_context.md`.

## Headline (registered R4)

**0 of 15 seed-stable wild candidate clusters (3 at m=128, 12 at m=256) are
natural-absorption candidates.** None meets the pre-registered asymmetric-nesting
condition `C(parent|child) ≥ 0.80 AND C(child|parent) < 0.80`. The **maximum
child→parent containment observed anywhere is 0.46** (clique latent 51|172),
far below the 0.80 absorption threshold; most are ≤ 0.013 (mutually exclusive).
This is the registered "0 survives" branch: the S1 candidates are correlated
real-feature families, not absorption — confirming, on wild features, the paper's
disclosed limitation that co-firing + cosine cannot separate absorption from the
CDX equivalence class.

## The candidates split cleanly into two non-absorption families

The naf column = fraction of a latent's top-50 tokens that are non-ASCII /
byte-fragment. The split is perfect: sign of the decoder cosine ⟺ family.

### Family 1 — typographic byte-fragment features (9 clusters; cos > 0, naf = 1.0)

Positive decoder cosine (0.46–0.63), symmetric moderate co-firing
(containments 0.22–0.46, no nesting). Every member fires on a **byte-fragment of
a multi-byte curly-typography character** abundant in the Gutenberg corpus:

| latent | fires on | interpretation |
|---|---|---|
| 51 | `Napoleon’⟧s`, `Alexander’⟧s` | possessive `’s` after a proper noun |
| 107 | `don’⟧t`, `wasn’⟧t` | apostrophe inside a contraction |
| 54 | `“⟧Well`, `“⟧Oh` | opening `“` starting dialogue |
| 172 | `honor!”⟧`, `frightful!”⟧` | closing `”` after end punctuation |
| 108/117 | `Emperor’⟧s`, `England’⟧s` | possessive (alt byte split) |
| 226/91 | `it!”⟧`, `. ”⟧Yes` | closing / opening quote |
| 12/76 (m128) | paragraph-initial `“`, `won’⟧t` | quote / apostrophe |

The **4-latent clique {51, 54, 107, 172}** (m=256, the S1 highlight) is exactly
this family: {possessive-’s, open-quote, contraction-’, close-quote}. Its full
4×4 containment matrix is **flat and symmetric** (all P(a|b) ∈ 0.24–0.46, no
value ≥ 0.80) — a single correlated punctuation family, **not** an absorption
hierarchy. These co-fire because curly typography clusters in dialogue-dense
passages, and share decoder geometry because they are all "interior byte of a
multi-byte punctuation token." A real dependence structure (the shuffled-firing
null already showed these flags are dependence-driven, not geometric), but a
**tokenizer × non-ASCII-typography artifact**, not linguistic absorption.

### Family 2 — anti-correlated distinct linguistic features (6 clusters; cos < 0, naf ≈ 0)

Negative decoder cosine (−0.56 to **−0.83**), near-zero co-firing (0.0001–0.014):
mutually exclusive real features, each coherent but **distinct** (no subset
relation):

| cluster | latent A | latent B | relation |
|---|---|---|---|
| m128 c1 / m256 c1 | mid-word continuation (`forehead`, `rigging`) | different mid-word set (`sofa`, `cardboard`) | anti-correlated |
| m256 c0 | common noun after determiner (`the ground`) | sentence/paragraph-initial word (`Nor`, `The regiment`) | anti-correlated |
| m256 c11 | proper-noun word-start (`Napoleon`, `Nat‑ásha`) | other continuation tokens | cos −0.83, exclusive |

These flag the detector only through the **low side of the two-sided lift**
(lift ≤ 0.5 = mutual exclusion), which is orthogonal to absorption (absorption
is the high, positive-cosine side).

## Classification (locked rule applied)

All 15 clusters → **category B** (correlated / anti-correlated feature family =
the disclosed CDX equivalence class). **Zero A** (natural absorption). **Zero C**
(all interpretable once byte fragments are reconstructed). No cluster is
Unresolved: the quantitative containment discriminator is unambiguous for every
one.

Independent Gemini semantic adjudication (prereg §"semantic-judgment protocol")
was attempted twice via `gemx` and **timed out both times** (the known
`gemx`-flakiness caveat); disclosed. It was confirmatory-only here — the
pre-registered *primary* discriminator is quantitative containment, which decides
A on its own, so Gemini's absence does not change the verdict. The author
semantic read from `natfeat_context.md` is recorded above.

## What this buys the program (tiered honestly)

- **The wild-absorption hunt is a null (tier-1, pre-registered).** No wild-caught
  absorption pair exists among the seed-stable S1 candidates. Nothing to escalate
  to causal / cross-corpus confirmation (the queued follow-up fires only on a
  surviving A; none survived).
- **Detector-refinement finding (tier-2).** On real data every flag is a
  non-absorption structure of two identifiable kinds. A wild absorption hunt
  should therefore (i) drop the low-lift/mutual-exclusion branch, (ii) require
  **positive cosine AND asymmetric containment ≥ 0.80**, and (iii) run on an
  ASCII-clean / monolingual corpus to suppress the byte-fragment typography
  family (a Gutenberg-multilingual artifact). This is direct evidence for the
  already-queued "containment-based orientation" stage: asymmetric containment is
  the exact metric that rejects all 15 wild candidates.
- **Consistency with prior claims.** Reinforces, on wild features, the dictionary-
  vs-code and CDX-equivalence-class disclosures already in the paper (§16/§17):
  co-firing structure is abundant and real, absorption structure is not implied
  by it.

## Cost

$0 GPU. One `e2-standard-2` ephemeral VM for ~6 min (tokenizer only, ~$0.01),
deleted. Analysis ~35 s CPU on the orchestrator.

**Environment.** Analysis: orchestrator CPU, Debian, Python 3.11, torch 1.13.0a0,
numpy 1.24.2 (deterministic; no RNG in the adjudication). Token regeneration:
ephemeral Debian 12 VM, Python 3.11, `transformers` (tokenizer-only, no torch),
HF `gpt2` `GPT2Tokenizer`; corpus + chunking + 500k truncation identical to
`experiments/extract_activations.py`. Regenerable inputs `token_data.json` and
`gpt2_encoder.json` are gitignored (rebuild: `analysis/render_context.py` header
+ the ephemeral-VM step documented here).
