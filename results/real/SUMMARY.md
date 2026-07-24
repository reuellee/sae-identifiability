# Round 11: real-model SAEs (Pythia-1.4B) — infrastructure + first absorption look

*Started 2026-07-24. Graduates the absorption program from the toy model +
semi-synthetic GPT-2-small injections to **real SAEs trained on a real
model's activations** — the credibility gap every reviewer has flagged, and
the natural next stage of the north star (SAE geometry → **identifiable
codes** → causal features → reusable abstractions → novel-task adaptation).*

## Why Pythia-1.4B on an L4 (not Gemma on an A100)

- **A100/H100 quota is 0** on this account across every region (on-demand and
  spot); the GPU ceiling is a single L4. An A100 needs a quota-increase
  request (owner-filed, slow, often denied for low-usage accounts).
- **The experiment doesn't need an A100.** SAE work = extract activations once
  (a 2B model fits on the 24 GB L4), cache, then train SAEs cheaply on the
  cached vectors. The A100's 80 GB only matters for 8B+ models or much faster
  iteration.
- **Model:** Gemma-2-2B (the SAEBench/Gemma-Scope absorption standard) is
  license-gated (needs an HF token on the box). **Pythia-1.4B** (EleutherAI,
  open, d_model = 2048, 24 layers) is a legitimate real model, a clean ~3×
  scale-up from the GPT-2-small (d = 768) used so far, and unblocked.

## Pipeline (committed)

- `experiments/real_extract.py` — model-agnostic residual-stream extraction
  (Pythia-1.4B layer 12; keeps token ids for later feature labeling).
- `experiments/real_train_sae.py` — standard wide SAE (pre-decoder bias,
  unit-norm decoder, ReLU, **L1 or TopK**, Anthropic-style dead-latent
  resampling); reports FVU / L0 / dead%. (fp16 activations, per-batch
  normalization — an early full-float32 copy OOM-killed the first attempt and
  was fixed.)
- `experiments/real_analyze.py` — scales the project's own rounds-8/9
  label-free pair detector (decoder-cosine band + two-sided co-firing lift +
  overlap veto) to large m by filtering to the rate window first; reports
  flagged-pair counts + interpretable examples (top-activating tokens).
- Ops: `ops/l4_real_pipeline.sh` / `ops/l4_retrain.sh` run self-contained on a
  spot L4 (git-clone the public repo, upload big artifacts to
  `gs://sae-identifiability-artifacts-ebd5a273/round11/`). Weights (~270 MB/SAE)
  and the ~5.7 GB activation cache exceed GitHub's 100 MB limit, so **only
  small stats/findings go to git; big binaries live in GCS.**

## Infrastructure milestone (achieved)

A real Pythia-1.4B layer-12 SAE (m = 16384, x8) trains to high quality on the
L4: **FVU ≈ 0.048** (95% of activation variance reconstructed), **L0 = 32**
(TopK budget), with dead-latent resampling recovering latents online. This
proves the real-model SAE infrastructure works end to end on the available
hardware. Activations (~1.4 M tokens) and weights are in GCS.

## First L1-vs-TopK absorption look (exploratory)

*Exploratory (not pre-registered): the first pass to see whether the detector
finds sane absorbed-pair structure on real SAEs and to calibrate a
confirmatory comparison. A pre-registered L1-vs-TopK comparison (matched
seeds, bars) follows once these numbers are in hand.*

**RESULTS PENDING** — TopK and L1 SAEs training; the detector runs on both.
The round-10 question this addresses at real scale: with automatic background
(a real model has thousands of features always present), does TopK have more
or fewer absorbed pairs than L1? Round 10 found isolated L1 doesn't absorb at
all (absorption is background-driven); a real model is the background-rich
regime where the L1-vs-TopK contrast is actually meaningful.

*(This section is filled from the frozen detector output —
`gs://.../round11/*_pairs.json` — when the run completes.)*
