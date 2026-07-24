# Round 11 artifact manifest (reproducibility)

*Added 2026-07-24 (whole-repo review finding #12). The large binaries live in
GCS, not git (>100 MB). This manifest pins them and the pipeline so another
group can reproduce or verify.*

## GCS artifacts (`gs://sae-identifiability-artifacts-ebd5a273/round11/`)

| object | bytes | md5 (base64) | crc32c (base64) |
|---|---|---|---|
| `acts_pythia-1.4b_L12.pt` | 5,734,307,494 | `jyTaU2gHsYuN6hLzMf4o/Q==` | `LNjaCA==` |
| `sae_pythia-1.4b_L12_topk_x8.pt` | 268,520,064 | `wRwJ3NEwIpmTudFB3aTdVw==` | `m9uvCQ==` |
| `sae_pythia-1.4b_L12_l1_x8.pt` | 268,520,046 | `TNE/Q1Rp9Vns2s/Uqjmf3A==` | `SmEWkQ==` |

(md5/crc32c are the GCS object hashes; verify a download with
`gcloud storage objects describe <obj> --format='value(md5_hash)'`.)

## Pipeline provenance

- **Repo commit:** run from `experiments/real_extract.py` + `real_train_sae.py`
  at the round-11 commits (see `git log -- experiments/real_*.py`).
- **Base model:** `EleutherAI/pythia-1.4b` (open, no gate). Loaded via
  `transformers.AutoModelForCausalLM`, fp16, `output_hidden_states=True`.
  Pin the HF model revision when re-running (add `revision=` to
  `from_pretrained`); the first run used the default `main`.
- **Layer:** 12 (residual stream, `hidden_states[12]`). `d_model = 2048`.
- **Corpus:** Project Gutenberg plain text — #2600 (War and Peace), #1661
  (Sherlock Holmes), #98 (Two Cities), #1342 (Pride and Prejudice), fetched at
  extraction time; ~1.4 M tokens extracted (`SEQ=512`). *No document-level
  train/test split* (a known limitation; the confirmatory experiment adds a
  held-out corpus split).
- **SAE:** m = 16384 (x8), unit-norm decoder, pre-decoder bias, ReLU;
  **TopK** k=32 / **L1** λ=5; 30k Adam steps, batch 4096, lr 4e-4 (→/3 at
  half), Anthropic-style dead-latent resampling. Activation normalization:
  per-batch, stats from a 200k-row subsample.
- **Environment:** spot L4 (`g2-standard-8`), image `dev-gpu-img-tmp`
  (torch 2.5.1+cu121, per `ENVIRONMENT.md`); `transformers` pip-installed at
  run time (pin the version when reproducing).

## Known gaps (to close for a confirmatory result — review §12)

The committed detector JSONs (`*_pairs.json`) hold only a top-N example sample,
not all flagged pairs. The confirmatory experiment must add: persisted
evaluation-token index files, complete pair-level (or aggregate-distribution)
outputs, training curves, a package lockfile/container, corpus snapshot hashes,
and the exact model+tokenizer revision hashes. The first detector pass also
used a non-registered firing threshold and unseeded subsample (both fixed in
`real_analyze.py`; see `results/real/SUMMARY.md`).
